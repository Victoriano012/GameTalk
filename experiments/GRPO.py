import torch.optim as optim
import torch

from omegaconf import OmegaConf
import random
from types import SimpleNamespace
from copy import deepcopy
from time import time
import wandb
import hydra
import sys
import os
import re
import gc

from game_utils import masked_call, one_turn, parse_last, estimate_strategy, kl_div
from llm_utils import LLM, one_turn_stop_criteria
from games import get_game



def create_batch(llm_1, llm_2, train_llm_num, config, metrics):
    """
    Creates a batch of episodes where two competing LLMs interact.
    Args:
        llm_1, llm_2: The two competing LLMs.
        train_llm_num (int): 1 to train llm_1, 2 to train llm_2, any other value to not train
            root conversation will be replicated before trained llm's turn

    Returns:
        - list of conversations (List[str])
        - list of winners (List[int])
        - list of errored conversations (List[bool])
        - data for training : [
            list of conversations (List[str]) : from the point of view of trained llm
            attention indices (List[pair(int,int)]) : start and end indices of interaction to train
            group indices (List[int]) : groups for GRPO
        ]
    """

    Game = get_game(config.game_name)
    
    swapped = False
    conversation = [""]

    with open(config.prompts.folder + config.prompts.initial, "r") as file:
        initial_prompt = file.read()
    
    player_1 = SimpleNamespace(
        llm = llm_1,
        name = config.player_1_name,
        query = [initial_prompt.format(my_name=config.player_1_name, other_name=config.player_2_name) + "<think>"],
        train = train_llm_num == 1,
        play = [None],
        generation_config = {
            "max_new_tokens" : config.train.max_new_tokens,
            "do_sample" : True,
            "top_p" : config.train.trained_top_p if train_llm_num == 1 else config.train.opponent_top_p,
            "temperature" : config.train.trained_temperature if train_llm_num == 1 else config.train.opponent_temperature
        }
    )
    player_2 = SimpleNamespace(
        llm = llm_2,
        name = config.player_2_name,
        query = [initial_prompt.format(my_name=config.player_2_name, other_name=config.player_1_name)],
        train = train_llm_num == 2,
        play = [None],
        generation_config = {
            "max_new_tokens" : config.train.max_new_tokens,
            "do_sample" : True,
            "top_p" : config.train.trained_top_p if train_llm_num == 2 else config.train.opponent_top_p,
            "temperature" : config.train.trained_temperature if train_llm_num == 2 else config.train.opponent_temperature
        }
    )

    must_play = [False]
    group_indices = [-1]
    attention_indices = [(0,0)]
    num_interactions = [-1] # unfinished := -1

    print("    Creating batch", flush=True)
    for interaction_idx in range(2*config.train.max_interactions):

        # check if both players played in all games
        if min(num_interactions) != -1:
            break

        # replicate root conversation if it's not over and training
        if player_1.train and num_interactions[0] == -1:
            conversation += [conversation[0]]*config.train.group_size
            player_1.query += [player_1.query[0]]*config.train.group_size
            player_2.query += [player_2.query[0]]*config.train.group_size
            player_1.play += [player_1.play[0]]*config.train.group_size
            player_2.play += [player_2.play[0]]*config.train.group_size
            
            must_play += [must_play[0]]*config.train.group_size
            group_indices += [group_indices[-1] + 1]*config.train.group_size
            attention_indices += [len(player_1.query[0])]*config.train.group_size # end index added later
            num_interactions += [-1]*config.train.group_size

        # generate actions
        actions = masked_call(
                    lambda x: one_turn(player_1.llm, x, **player_1.generation_config),
                    player_1.query,
                    [(x == -1) for x in num_interactions]
                )
        for idx, action in enumerate(actions):
            if action == "":
                continue

            try:
                parsed_action = parse_last(action)
            except AssertionError as e:
                think_tag = "<think>"
                player_1.query[idx] += action[len(think_tag):]
                conversation[idx] += player_1.name + " did a formating error, their response could not be parsed.\n"

                if isinstance(attention_indices[idx], int):
                    attention_indices[idx] = (attention_indices[idx], len(player_1.query[idx]))

                player_1.play[idx] = Game("error")
                if player_2.play[idx] == None:
                    player_2.play[idx] = Game.default()
                num_interactions[idx] = interaction_idx+1

                continue

            # check if player played
            if 'play' in parsed_action:
                player_1.play[idx] = Game(parsed_action['play'].lower().strip())
                must_play[idx] = True
            elif must_play[idx]:
                player_1.play[idx] = Game("error")

            if player_1.play[idx] and player_2.play[idx]:
                num_interactions[idx] = interaction_idx+1
                metrics["num_interactions"] += interaction_idx+1

            ### add last action to queries
            # player_1 query
            player_1.query[idx] += parsed_action['think'] + "</think>\n"
            if 'talk' in parsed_action:
                player_1.query[idx] += "<talk>" + parsed_action['talk'] + "</talk> \n"
            if 'play' in parsed_action:
                player_1.query[idx] += "<play>" + parsed_action['play'] + "</play> \n"
            
            if isinstance(attention_indices[idx], int):
                attention_indices[idx] = (attention_indices[idx], len(player_1.query[idx]))
            player_1.query[idx] += player_2.name + ": " 
            
            # player_2 query
            if 'talk' in parsed_action:
                player_2.query[idx] += parsed_action['talk'].strip() + "\n"
            if 'play' in parsed_action:
                with open(config.prompts.folder + config.prompts.other_moved, "r") as file:
                    player_2.query[idx] += file.read().format(other_name = player_1.name)
            player_2.query[idx] += player_2.name + ": <think>" 

            # conversation
            conversation[idx] += player_1.name + ":\n    <think>" + parsed_action['think'] + "</think>\n"
            if 'talk' in parsed_action:
                conversation[idx] += "    <talk>" + parsed_action['talk'] + "</talk> \n"
            if 'play' in parsed_action:
                conversation[idx] += "    <play>" + parsed_action['play'] + "</play> \n"

        # swap players for next round
        swapped = not swapped
        player_1, player_2 = player_2, player_1

    if swapped:
        player_1, player_2 = player_2, player_1

    # error if any player didn't play
    for idx in range(len(player_1.play)):
        if player_1.play[idx] is None:
            player_1.play[idx] = Game("error")
        if player_2.play[idx] is None:
            player_2.play[idx] = Game("error")
        # if player_1.play[idx] is None or player_2.play[idx] is None:
        #     errors[idx] = AssertionError(f"Players didn't play : {player_1.name} -> {player_1.play[idx]}, {player_2.name} -> {player_2.play[idx]}")
        #     player_1.play[idx] = player_2.play[idx] = Game("error")

    # concat moves
    moves = list(zip(player_1.play, player_2.play))

    training_conversation = player_1.query if player_1.train else player_2.query
    metrics["conversation_length (tokens)"] = len((player_1 if player_1.train else player_2).llm.tokenizer(training_conversation[0])['input_ids'])
    
    # return conversation, moves, errors, (training_conversation, attention_indices, group_indices, num_interactions)
    return conversation, moves, (training_conversation, attention_indices, group_indices, num_interactions)

def eval_batch(llm_1, llm_2, eval_llm_num, config, metrics, metrics_prefix=""):
    """
    Evaluates two competing LLMs through the creation of a batch.
    Returns nothing, evaluations are added to metrics
    """

    metrics[metrics_prefix+"internal_state_loss"] = 0.0
    metrics[metrics_prefix+"word_based_loss"] = 0.0
    word_based_count = 0
    internal_state_count = 0

    Game = get_game(config.game_name)
    
    swapped = False
    conversation = ["" for _ in range(config.eval.num_episodes)]

    with open(config.prompts.folder + config.prompts.initial, "r") as file:
        initial_prompt = file.read()
    
    player_1 = SimpleNamespace(
        llm = llm_1,
        name = config.player_1_name,
        query = [initial_prompt.format(my_name=config.player_1_name, other_name=config.player_2_name) + "<think>" for _ in range(config.eval.num_episodes)],
        eval = eval_llm_num == 1,
        play = [None for _ in range(config.eval.num_episodes)],
        generation_config = {
            "max_new_tokens" : config.train.max_new_tokens,
            "do_sample" : True,
            "top_p" : config.eval.trained_top_p if eval_llm_num == 1 else config.eval.opponent_top_p,
            "temperature" : config.eval.trained_temperature if eval_llm_num == 1 else config.eval.opponent_temperature
        }
    )
    player_2 = SimpleNamespace(
        llm = llm_2,
        name = config.player_2_name,
        query = [initial_prompt.format(my_name=config.player_2_name, other_name=config.player_1_name) for _ in range(config.eval.num_episodes)],
        eval = eval_llm_num == 2,
        play = [None for _ in range(config.eval.num_episodes)],
        generation_config = {
            "max_new_tokens" : config.train.max_new_tokens,
            "do_sample" : True,
            "top_p" : config.eval.trained_top_p if eval_llm_num == 2 else config.eval.opponent_top_p,
            "temperature" : config.eval.trained_temperature if eval_llm_num == 2 else config.eval.opponent_temperature
        }
    )

    must_play = [False for _ in range(config.eval.num_episodes)]
    errors = [None for _ in range(config.eval.num_episodes)]

    print("    Evaluating", flush=True)
    for _ in range(2*config.train.max_interactions):

        # check if both players played in all games
        game_over = [x and y for x,y in zip(player_1.play, player_2.play)]
        if all(game_over):
            break

        # evaluate player_1 if needed
        if player_1.eval:
            ### internal state evaluation
            player_1_estimation = masked_call(
                lambda q : estimate_strategy(player_1.llm, q, Game, other_name=player_2.name),
                player_1.query,
                [not x for x in game_over],
                unpack = False
            )
            player_2_strategy = masked_call(
                lambda q : estimate_strategy(player_2.llm, q, Game, other_name=None),
                player_2.query,
                [not x for x in game_over],
                unpack = False
            )

            curr_kl = 0.0
            for x in range(len(player_1_estimation)):
                curr_kl += kl_div(player_1_estimation[x], player_2_strategy[x])
            curr_kl /= len(player_1_estimation)
            metrics[metrics_prefix+"internal_state_loss"] += curr_kl
            internal_state_count += 1

        # generate actions
        actions = masked_call(
                    lambda x: one_turn(player_1.llm, x, **player_1.generation_config),
                    player_1.query,
                    [not x for x in game_over]
                )
        for idx, action in enumerate(actions):
            if action == "":
                continue

            try:
                parsed_action = parse_last(action)
            except AssertionError as e:
                errors[idx] = e
                player_1.play[idx] = player_2.play[idx] = Game("error")
                continue

            # check if player played
            if 'play' in parsed_action:
                player_1.play[idx] = Game(parsed_action['play'].lower().strip())
            elif must_play[idx]:
                player_1.play[idx] = Game("error")

            ### add last action to queries
            # player_1 query
            player_1.query[idx] += parsed_action['think'] + "</think>\n"
            if 'talk' in parsed_action:
                player_1.query[idx] += "<talk>" + parsed_action['talk'] + "</talk> \n"
            if 'play' in parsed_action:
                player_1.query[idx] += "<play>" + parsed_action['play'] + "</play> \n"
            
            player_1.query[idx] += player_2.name + ": " 
            
            # player_2 query
            if 'talk' in parsed_action:
                player_2.query[idx] += parsed_action['talk'].strip() + "\n"
            if 'play' in parsed_action:
                with open(config.prompts.folder + config.prompts.other_moved, "r") as file:
                    player_2.query[idx] += file.read().format(other_name = player_1.name)
                must_play[idx] = True
            player_2.query[idx] += player_2.name + ": <think>" 

            # conversation
            conversation[idx] += player_1.name + ":\n    <think>" + parsed_action['think'] + "</think>\n"
            if 'talk' in parsed_action:
                conversation[idx] += "    <talk>" + parsed_action['talk'] + "</talk> \n"
            if 'play' in parsed_action:
                conversation[idx] += "    <play>" + parsed_action['play'] + "</play> \n"

            if player_1.eval and 'talk' in parsed_action:
                metrics[metrics_prefix+"word_based_loss"] += sum(len(re.findall(r'\b' + re.escape(word) + r'\b', parsed_action['talk'])) for word in config.bad_words)
                word_based_count += 1

        # swap players for next round
        swapped = not swapped
        player_1, player_2 = player_2, player_1

    metrics[metrics_prefix+"word_based_loss"] /= word_based_count if word_based_count != 0 else 1
    metrics[metrics_prefix+"internal_state_loss"] /= internal_state_count if internal_state_count != 0 else 1

def batch_step(train_llm, ref_llm, optimizer, batch, config, metrics, metrics_prefix):
    input, padding, attention_mask, advantage = batch

    # random shuffle the batch
    indices = torch.randperm(input.shape[0])
    input, padding, attention_mask, advantage = input[indices], padding[indices], attention_mask[indices], advantage[indices]

    mini_size = config.train.minibatch_size
    mini_batches = [(input[i:i+mini_size], padding[i:i+mini_size], attention_mask[i:i+mini_size], advantage[i:i+mini_size]) for i in range(0, len(input), mini_size)]

    train_llm.train()
    optimizer.zero_grad()

    print("    Processing minibatches", metrics_prefix, flush=True)

    for mini_batch in mini_batches:
        input_batch, padding_batch, attention_mask_batch, advantage_batch = mini_batch
        first_one_idx = padding_batch.to(torch.int).argmax(dim=1).min()
        input_batch = input_batch[:,first_one_idx:]
        padding_batch = padding_batch[:,first_one_idx:]
        attention_mask_batch = attention_mask_batch[:,first_one_idx:]
    
    ref_llm.to('cuda')
    train_llm.to('cpu')
    ref_log_probs = []
    with torch.no_grad():
        for mini_batch in mini_batches:
            input_batch, padding_batch, _, _ = mini_batch
            input_batch = input_batch.clone().to('cuda')
            padding_batch = padding_batch.clone().to('cuda')

            # compute log_probs and ref_log_probs
            ref_log_probs.append(
                ref_llm.get_log_probs(input_batch, padding_batch).to('cpu')
            )

    ref_llm.to('cpu')
    train_llm.to('cuda')
    for mini_batch, ref_log_prob in zip(mini_batches, ref_log_probs):
        input_batch, padding_batch, attention_mask_batch, advantage_batch = mini_batch
        input_batch = input_batch.clone().to('cuda')
        padding_batch = padding_batch.clone().to('cuda')
        attention_mask_batch = attention_mask_batch.clone().to('cuda')
        advantage_batch = advantage_batch.clone().to('cuda')
        ref_log_prob = ref_log_prob.clone().to('cuda')

        if not config.train.gradient_accumulation:
            optimizer.zero_grad()

        # compute log_prob
        log_prob = train_llm.get_log_probs(input_batch, padding_batch)

        # compute loss
        ratio = torch.exp(log_prob - ref_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - config.train.ppo_eps, 1 + config.train.ppo_eps)

        advantage_batch = advantage_batch.unsqueeze(1)
        grpo_loss = torch.min(ratio * advantage_batch, clipped_ratio * advantage_batch)

        kl = ref_log_prob - log_prob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1)
        
        masked_loss = (grpo_loss - config.train.kl_coef * kld) * attention_mask_batch / attention_mask_batch.sum(dim=-1, keepdim=True)

        loss = - masked_loss.sum() / input_batch.shape[0]

        # backpropagate
        loss.backward()
        if not config.train.gradient_accumulation:
            optimizer.step()

        metrics[metrics_prefix + "kl"] += (kld * attention_mask_batch).sum().item()
        metrics[metrics_prefix + "total_loss"] += -masked_loss.sum().item()
    
    if config.train.gradient_accumulation:
        optimizer.step()


def train_loop(train_llm, opponent_llm, config):

    logger = wandb.init(
        config=OmegaConf.to_container(config, resolve=True),
        name = config.run_name,
        project = config.wandb.project
    )

    conversation_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.conversations}", "w")
    metrics_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.metrics}", "w")
    print(OmegaConf.to_container(config, resolve=True), file=metrics_file, flush=True)

    Game = get_game(config.game_name)
    buffer = []

    ref_llm = LLM(config.train_llm_name, stopping_criteria=one_turn_stop_criteria, lora_config=config.lora, unsloth=config.unsloth)
    # ref_llm = deepcopy(train_llm)
    ref_llm.eval()
    opponent_llm.eval()

    trainable_parameters = [p for p in train_llm.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_parameters, lr=config.train.lr)

    for epoch in range(config.train.epochs):
        print(f"EPOCH {epoch}", flush=True)
        if config.lora is None:
            ref_llm.model.load_state_dict(train_llm.model.state_dict())
        else:
            lora_state_dict = {k: v for k, v in train_llm.model.state_dict().items() if 'lora' in k}
            ref_llm.model.load_state_dict(lora_state_dict, strict=False)
        metrics = {metric : 0 for metric in config.tracked_metrics}
        # no metric is divided by num_samples until just before logging

        for batch_idx in range(config.train.batches_per_epoch):
            train_llm.eval()

            ### compute the batch

            if config.trained_player == 1 or (config.trained_player == "both" and batch_idx % 2 == 1):
                llm_1, llm_2 = train_llm, opponent_llm
                train_llm_num = 1
            else:
                llm_1, llm_2 = opponent_llm, train_llm
                train_llm_num = 2

            ref_llm.to('cpu')
            opponent_llm.to('cuda')
            torch.cuda.empty_cache()

            time_pre_generation = time()
            torch.cuda.reset_peak_memory_stats()

            # conversation, moves, errors, train_data = create_batch(llm_1, llm_2, train_llm_num, config, metrics)
            conversation, moves, train_data = create_batch(llm_1, llm_2, train_llm_num, config, metrics)
            training_conversation, att_idx, group_indices, num_interactions = train_data

            metrics["time_generation (s)"] = time() - time_pre_generation
            metrics["memory_usage_generation (GB)"] = torch.cuda.max_memory_allocated() / 1024**3
            # metrics["memory_usage_generation (GB)"] = torch.cuda.memory_allocated() / 1024**3
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            print(f"\n\nEPOCH {epoch} - BATCH {batch_idx} :\n", file=conversation_file)
            print(conversation[0], file=conversation_file, flush=True)

            opponent_llm.to('cpu')

            ### prepare batch for training
            
            """
            # get rid of errors
            metrics["num_error_conversations"] = len(errors) - errors.count(None)
            conversation = [c for c, e in zip(conversation, errors) if e is None]
            moves = [w for w, e in zip(moves, errors) if e is None]
            training_conversation = [c for c, e in zip(training_conversation, errors) if e is None]
            att_idx = [a for a, e in zip(att_idx, errors) if e is None]
            group_indices = [g for g, e in zip(group_indices, errors) if e is None]
            num_interactions = [n for n, e in zip(num_interactions, errors) if e is None]
            """

            # compute groups for GRPO
            group_indices = torch.tensor(group_indices)
            unique_groups, counts = torch.unique(group_indices, return_counts=True)
            # get rid of root conversation & avoid std error in some niche cases
            unique_groups = unique_groups[counts > 1]
            group_mask = torch.isin(group_indices, unique_groups)
            group_indices = group_indices[group_mask]
            conversation = [conversation[i] for i in range(len(group_mask)) if group_mask[i]]
            moves = [moves[i] for i in range(len(group_mask)) if group_mask[i]]
            training_conversation = [training_conversation[i] for i in range(len(group_mask)) if group_mask[i]]
            att_idx = [att_idx[i] for i in range(len(group_mask)) if group_mask[i]]
            num_interactions = [num_interactions[i] for i in range(len(group_mask)) if group_mask[i]]

            # moves -> rewards
            if train_llm_num == 2:
                moves = [(w[1], w[0]) for w in moves]
            rewards = torch.tensor([Game.score(w[0], w[1]) for w in moves])
            opponent_rewards = torch.tensor([Game.score(w[1], w[0]) for w in moves])
            metrics["win_rate"] += (rewards == 2.).sum().item()
            metrics["draw_rate"] += (rewards == 1.).sum().item()
            metrics["loss_rate"] += (rewards <= 0.).sum().item()
            metrics["reward"] += rewards.sum().item()
            metrics["opponent_reward"] += opponent_rewards.sum().item()
            metrics["lost_by_error"] = sum(1 for w in moves if w[0].is_error())
            metrics["won_by_error"] = sum(1 for w in moves if w[1].is_error() and not w[0].is_error())

            # compute advantages
            means = {group.item() : rewards[group_indices == group].mean() for group in unique_groups}
            stds = {group.item() : rewards[group_indices == group].std()+config.train.grpo_std_eps for group in unique_groups}
            group_indices = group_indices.tolist()
            advantage = torch.tensor([(rewards[i] - means[group_indices[i]]) / stds[group_indices[i]] for i in range(len(rewards))])

            # multiply by discount factor
            discount_factor = torch.full(advantage.shape, config.train.gamma) ** ((torch.tensor(num_interactions)-train_llm_num)//2 - torch.tensor(group_indices))
            advantage = (advantage*discount_factor)

            # with the advantages, we can forget everything after the evaluated turn
            training_conversation = [c[:idx[1]] for c, idx in zip(training_conversation, att_idx)]

                # this happens when one player doesn't ever play
                # so it should NEVER happen
            if len(training_conversation) == 0:
                print("Empty training conversation")
                print("This is not normal")
                continue

            # tokenize training_conversation
            tokenized = train_llm.tokenizer(training_conversation, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True)
            input = tokenized['input_ids']
            padding = tokenized['attention_mask']

            # att_idx -> attention_mask
            attention_mask = []
            for idx in range(len(training_conversation)):
                offsets = tokenized['offset_mapping'][idx].tolist()  # Get the offsets for the current text
                mask = [not (token_end <= att_idx[idx][0] or token_start >= att_idx[idx][1]) for token_start, token_end in offsets]
                attention_mask.append(mask)
            attention_mask = torch.tensor(attention_mask)
            
            curr_batch = (input, padding, attention_mask, advantage)
            buffer.append(curr_batch)

            # train current batch
            time_pre_minibatches = time()
            batch_step(train_llm, ref_llm, optimizer, curr_batch, config, metrics, "")

            metrics["num_samples"] += len(input)
            metrics["memory_usage_minibatch (GB)"] = torch.cuda.max_memory_allocated() / 1024**3
            metrics["time_total_minibatches (s)"] = time() - time_pre_minibatches
            metrics["normalized_relative_advantage"] = (metrics["reward"] - metrics["opponent_reward"])/(metrics["reward"] + metrics["opponent_reward"] + 1e-8)

        buffer = buffer[-config.train.buffer_size_limit:]

        # replay
        time_pre_replay = time()
        for _ in range(config.train.replay_batches_per_epoch):
            replay_batch = random.choice(buffer)
            batch_step(train_llm, ref_llm, optimizer, replay_batch, config, metrics, "replay_")
        metrics["time_total_replay (s)"] = time() - time_pre_replay
        
        # eval if needed
        if config.eval.eval_every is not None and epoch % config.eval.eval_every == 0:
            ref_llm.to('cpu')
            opponent_llm.to('cuda')
            if config.trained_player in [1, "both"]:
                eval_batch(train_llm, opponent_llm, 1, config, metrics, "player1_" if config.trained_player == "both" else "")
            if config.trained_player in [2, "both"]:
                eval_batch(opponent_llm, train_llm, 2, config, metrics, "player2_" if config.trained_player == "both" else "")
            opponent_llm.to('cpu')

        # log metrics
        for metric in config.normalized_metrics:
            metrics[metric] /= max(metrics["num_samples"], 1)
        logger.log(metrics)

        print(f"\nEPOCH {epoch}", file=metrics_file)
        print(metrics, file=metrics_file, flush=True)

        if epoch % config.train.save_every == 0 and config.lora is not None:
            train_llm.model.save_pretrained(f"{config.logs.folder}{config.run_name}/{config.logs.model}") # save LoRA model

    conversation_file.close()
    metrics_file.close()

@hydra.main(config_path='config', config_name='config', version_base=None)
def __main__(config):
    assert torch.cuda.is_available() , "This script is designed to run on GPU, please make sure cuda is available"
    torch.autograd.set_detect_anomaly(True)

    os.makedirs(f"{config.logs.folder}{config.run_name}", exist_ok=True)
    general_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.general}", "w")
    sys.stdout = general_file
    sys.stderr = general_file

    train_llm = LLM(config.train_llm_name, stopping_criteria=one_turn_stop_criteria, lora_config=config.lora, unsloth=config.unsloth).to('cuda')
    opponent_llm = LLM(config.opponent_llm_name, stopping_criteria=one_turn_stop_criteria, unsloth=config.unsloth)

    print("\nStart training")
    train_loop(train_llm, opponent_llm, config)
    print("\nTraining is done")

    if config.lora is not None:
        train_llm.model.save_pretrained(f"{config.logs.folder}{config.run_name}/{config.logs.model}") # save LoRA model

if __name__ == "__main__":
    __main__()
