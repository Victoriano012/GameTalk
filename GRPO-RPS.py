from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from types import SimpleNamespace
from time import time
from enum import Enum
from tqdm import tqdm
from bs4 import BeautifulSoup
from omegaconf import OmegaConf
import wandb
import hydra

device = 'cuda'

class LLM(nn.Module):
    def __init__(self, llm_name, stopping_criteria=None, lora_config=None):
        super().__init__()
        self.stopping_criteria = stopping_criteria

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(llm_name).to(device)
        if lora_config is not None:
            self.model = get_peft_model(self.model, lora_config)

    def generate(self, prompt, max_length=2000):
        inputs = self.tokenizer(prompt,padding=True,truncation=True, return_tensors="pt").to(device)
        output = self.model.generate(
            **inputs,
            max_length=max_length,
            stopping_criteria=self.stopping_criteria(self.tokenizer),
            pad_token_id=128009 # llm.tokenizer.eos_token
        )
        output = output[:, inputs['input_ids'].shape[-1]:]
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)
    
    def get_log_probs(self, input):
        output = self.model(input)
        log_prob = output.logits.log_softmax(dim=-1)
        log_prob = log_prob[torch.arange(input.shape[0]).unsqueeze(1), torch.arange(input.shape[1]), input].reshape(input.shape)
        return log_prob
    
default_stopping_text = ["</talk>", "</play>"]

# stopping criteria to force generation stopping after each player's turn
class one_player_generation(StoppingCriteria):
    def __init__(self, tokenizer, stopping_text = default_stopping_text, max_generation_len = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.stopping_text = stopping_text
        self.initial_length = None
        self.max_generation_len = max_generation_len

    def __call__(self, input_ids, scores):
        if self.initial_length is None:
            self.initial_length = input_ids.shape[1]
            self.text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            return False
        
        if self.max_generation_len is not None and \
                input_ids.shape[1] - self.initial_length > self.max_generation_len:
            return True
        
        stop = []
        for i in range(input_ids.shape[0]):
            self.text[i] += self.tokenizer.decode(input_ids[i,-1:])
            stop.append(any(
                self.text[i].strip().endswith(st) for st in self.stopping_text
            ))

        return torch.tensor(stop).to(device)

stop_criteria = lambda tokenizer: StoppingCriteriaList([one_player_generation(tokenizer = tokenizer, max_generation_len = 200)])

def parse_last(text):
    # Regular expression to find all tags with content
    soup = BeautifulSoup(text, 'html.parser')
    matches = [(tag.name, tag.get_text()) for tag in soup.find_all()]
    
    parsed_text = {}
    for tag, content in matches:
        if tag == "think":
            parsed_text = {}
        parsed_text[tag] = content
    
    if len(parsed_text) < 2:
        raise AssertionError("Format error", text)
    
    return parsed_text


# solve format errors from early stopping 
def rps_turn(llm, query):
    text = llm.generate(query)

    # find unfinished thinks
    reindex = []
    query_again = []
    for i, t in enumerate(text):
        think_end = t.rfind("</think>")
        if think_end == -1:
            reindex.append(i)
            query_again.append(query[i] + t + " </think> ")

    # complete unfinished thinks
    if len(reindex) > 0:
        text_2 = llm.generate(query_again)
        for idx, t in zip(reindex, text_2):
            text[idx] += " </think> " + t

    # finish unfinished talks
    for i, t in enumerate(text):
        talk_start = t.rfind("<talk>")
        talk_end = t.rfind("</talk>")
        if talk_end == -1 and talk_start > -1:
            text[i] += " </talk>\n"
    
    return ["<think>" + t for t in text]


def masked_call(cls, queries, mask):
    filtered_inputs = [q for q, m in zip(queries, mask) if m] # Extract elements where mask is 1
    filtered_outputs = cls(filtered_inputs)                   # Call cls once with all necessary elements
    output_iter = iter(filtered_outputs)                      # Iterator to retrieve processed elements

    return [next(output_iter) if m else "" for m in mask]


class RPS(Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"

    def compete(move1, move2) -> int:
        """
        move1 wins -> 1, move2 wins -> 2, tie -> 0
        """
        mapping = {"rock": 0, "paper": 1, "scissors": 2}
        return (mapping[move1.value] - mapping[move2.value] +3) % 3


def create_batch(llm_1, llm_2, train_llm_num, config):
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
    
    swapped = False
    conversation = [""]

    player_1 = SimpleNamespace(llm=llm_1, name=config.player_1_name)
    player_2 = SimpleNamespace(llm=llm_2, name=config.player_2_name)

    with open(config.prompts.folder + config.prompts.initial, "r") as file:
        initial_prompt = file.read()
    player_1.query = [initial_prompt.format(my_name=player_1.name, other_name=player_2.name) + "<think>"]
    player_2.query = [initial_prompt.format(my_name=player_2.name, other_name=player_1.name)]

    player_1.train = train_llm_num == 1
    player_2.train = train_llm_num == 2

    player_1.play = [None]
    player_2.play = [None]

    errors = [None]
    group_indices = [-1]
    attention_indices = [(0,0)]

    print("    Creating batch", flush=True)
    for t in range(2*config.train.max_interactions):

        # check if both players played in all games
        game_over = [x and y for x,y in zip(player_1.play, player_2.play)]
        if all(game_over):
            break

        # replicate root conversation if it's not over and training
        if player_1.train and not game_over[0]:
            conversation += [conversation[0]]*config.train.group_size
            player_1.query += [player_1.query[0]]*config.train.group_size
            player_2.query += [player_2.query[0]]*config.train.group_size
            player_1.play += [player_1.play[0]]*config.train.group_size
            player_2.play += [player_2.play[0]]*config.train.group_size
            errors += [None]*config.train.group_size
            group_indices += [group_indices[-1] + 1]*config.train.group_size
            attention_indices += [len(player_1.query[0])]*config.train.group_size # end index added later
            game_over += [False]*config.train.group_size

        # generate actions
        actions = masked_call(
                    lambda x: rps_turn(player_1.llm, x),
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
                player_1.play[idx] = player_2.play[idx] = RPS.ROCK
                continue

            # check if player played
            if 'play' in parsed_action:
                try:
                    player_1.play[idx] = RPS(parsed_action['play'].lower().strip())
                except ValueError as e:
                    errors[idx] = e
                    player_1.play[idx] = player_2.play[idx] = RPS.ROCK
                    continue


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
            conversation[idx] += player_1.name + ": <think>" + parsed_action['think'] + "</think>\n"
            if 'talk' in parsed_action:
                conversation[idx] += "<talk>" + parsed_action['talk'] + "</talk> \n"
            if 'play' in parsed_action:
                conversation[idx] += "<play>" + parsed_action['play'] + "</play> \n"

        # swap players for next round
        swapped = not swapped
        player_1, player_2 = player_2, player_1

    if swapped:
        player_1, player_2 = player_2, player_1

    # error if any player didn't play
    for idx in range(len(player_1.play)):
        if player_1.play[idx] is None or player_2.play[idx] is None:
            errors[idx] = AssertionError(f"Players didn't play : {player_1.name} -> {player_1.play[idx]}, {player_2.name} -> {player_2.play[idx]}")
            player_1.play[idx] = player_2.play[idx] = RPS.ROCK

    # find winner
    winner = map(RPS.compete, player_1.play, player_2.play)
    mapping = {1: player_1.name, 2: player_2.name, 0: "Tie"}
    winner = [mapping[w] for w in winner]

    training_conversation = player_1.query if player_1.train else player_2.query
    return conversation[1:], winner[1:], errors[1:], (training_conversation[1:], attention_indices[1:], group_indices[1:])
    # root conversation is not returned


def train_loop(train_llm, opponent_llm, config):

    logger = wandb.init(
        config=OmegaConf.to_container(config, resolve=True),
        **config.wandb,
    )

    ref_llm = deepcopy(train_llm)
    ref_llm.eval()
    opponent_llm.eval()

    trainable_parameters = [p for p in train_llm.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_parameters, lr=config.train.lr)

    for epoch in range(config.train.epochs):
        print(f"EPOCH {epoch}", flush=True)
        ref_llm.load_state_dict(train_llm.state_dict())
        metrics = {"total_loss": 0, "kl": 0, "win_rate": 0, "draw_rate": 0, "loss_rate": 0, "num_samples": 0}
        # no metric is divided by num_samples until just before logging

        for batch_idx in range(config.train.batches_per_epoch):
            train_llm.eval()

            ### compute the batch

            if config.trained_player == 1 or (config.trained_player == "both" and batch_idx % 2 == 1):
                llm_1, llm_2 = train_llm, opponent_llm
                train_llm_num = 1
                player_name = "Player-1"
            else:
                llm_1, llm_2 = opponent_llm, train_llm
                train_llm_num = 2
                player_name = "Player-2"

            for attempt in range(5):
                try:
                    conversation, winner, errors, train_data = create_batch(llm_1, llm_2, train_llm_num, config)
                    break
                except MemoryError:
                    print(f"Batch creation, attempt {attempt} failed due to memory limits.", flush=True)
            training_conversation, att_idx, group_indices = train_data


            ### train llm with computed batch

            train_llm.train()
            
            # get rid of errors
            conversation = [c for c, e in zip(conversation, errors) if e is None]
            winner = [w for w, e in zip(winner, errors) if e is None]
            training_conversation = [c for c, e in zip(training_conversation, errors) if e is None]
            att_idx = [a for a, e in zip(att_idx, errors) if e is None]
            group_indices = [g for g, e in zip(group_indices, errors) if e is None]

            # winner string -> rewards
            winner_to_reward = {player_name : 1., "Tie": 0.}
            rewards = torch.tensor([winner_to_reward.get(w, -1.) for w in winner]).to(device)
            metrics["win_rate"] += (rewards == 1.).sum().item()
            metrics["draw_rate"] += (rewards == 0.).sum().item()
            metrics["loss_rate"] += (rewards == -1.).sum().item()

            # compute advantages
            group_indices = torch.tensor(group_indices)
            unique_groups = torch.unique(group_indices)
            means = torch.tensor([rewards[group_indices == group].mean() for group in unique_groups]).to(device)
            stds = torch.tensor([rewards[group_indices == group].std()+config.train.grpo_std_eps for group in unique_groups]).to(device)
            advantage = (rewards - means[group_indices]) / stds[group_indices]

            # # since we already have the advantages, we can forget about everything after the turn we are evaluating
            training_conversation = [c[:idx[1]] for c, idx in zip(training_conversation, att_idx)]

            # tokenize training_conversation
            tokenized = train_llm.tokenizer(training_conversation, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True)
            input = tokenized['input_ids'].to(device)

            # att_idx -> attention_mask
            attention_mask = []
            for idx in range(len(training_conversation)):
                offsets = tokenized['offset_mapping'][idx].tolist()  # Get the offsets for the current text
                mask = [not (token_end <= att_idx[idx][0] or token_start >= att_idx[idx][1]) for token_start, token_end in offsets]
                attention_mask.append(mask)
            attention_mask = torch.tensor(attention_mask).to(device)


            mini_size = config.train.minibatch_size
            print("    Processing minibatches", flush=True)
            for i in range(0, len(training_conversation), mini_size):
                optimizer.zero_grad()

                input_batch = input[i:i+mini_size]
                attention_mask_batch = attention_mask[i:i+mini_size]
                advantage_batch = advantage[i:i+mini_size]

                # compute log_probs and ref_log_probs
                log_probs = train_llm.get_log_probs(input_batch)
                ref_log_probs = ref_llm.get_log_probs(input_batch)

                # compute loss
                ratio = torch.exp(log_probs - ref_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - config.train.ppo_eps, 1 + config.train.ppo_eps)

                advantage_batch = advantage_batch.unsqueeze(1)
                loss = torch.min(ratio * advantage_batch, clipped_ratio * advantage_batch)

                kl = ref_log_probs - log_probs
                ratio = torch.exp(kl)
                kld = (ratio - kl - 1)
                
                masked_loss = (loss - config.train.kl_coef * kld) * attention_mask_batch

                loss = - masked_loss.sum() / input_batch.shape[0]

                # backpropagate
                loss.backward()
                optimizer.step()

                metrics["kl"] += (kld * attention_mask_batch).sum().item()
                metrics["total_loss"] += -masked_loss.sum().item()
                metrics["num_samples"] += input_batch.shape[0]
        
        for metric in metrics:
            if metric != "num_samples":
                metrics[metric] /= metrics["num_samples"]
        logger.log(metrics)

@hydra.main(config_path='config', config_name='config', version_base=None)
def __main__(config):

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        **config.lora
    ) if config.lora is not None else None

    train_llm = LLM(config.train_llm_name, stopping_criteria=stop_criteria, lora_config=lora_config)
    opponent_llm = LLM(config.opponent_llm_name, stopping_criteria=stop_criteria)

    train_loop(train_llm, opponent_llm, config)

__main__()