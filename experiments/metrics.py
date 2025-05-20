import numpy as np
import random
import torch
import math
import re

from itertools import accumulate
from functools import partial

from game_dataset import finish_conversations
from games import RPS, BertrandCompetition, SizePrizeGame
from utils import simple_cache
from copy import deepcopy

def get_eval_metrics(train_llm, opponent_llm):
    return {
        "word_based_loss" : wordBasedLoss,
        "internal_state_evaluation" : partial(internalStateEvaluation, train_llm=train_llm, opponent_llm=opponent_llm),
        "state_relative_performance" : partial(stateRelativePerformance, train_llm=train_llm, opponent_llm=opponent_llm),
        "leverage_opportunity" : partial(leverageOpportunity, train_llm=train_llm, opponent_llm=opponent_llm),
        "internal_state_evaluation (lastTurn)" : partial(internalStateEvaluation, train_llm=train_llm, opponent_llm=opponent_llm, lastTurn=True),
        "state_relative_performance (lastTurn)" : partial(stateRelativePerformance, train_llm=train_llm, opponent_llm=opponent_llm, lastTurn=True),
        "leverage_opportunity (all turns)" : partial(leverageOpportunity, train_llm=train_llm, opponent_llm=opponent_llm, lastTurn=False),
    }


########### Word Based Loss ###########

bad_words = []
def inividual_wordBasedLoss(conversation, train_llm_num):
    word_based_loss = 0.0
    player = conversation.player_1 if train_llm_num%2 == 1 else conversation.player_2
    for parsed_action in player.parsed_actions:
        if 'talk' not in parsed_action:
            continue
        talk = parsed_action['talk']
        word_based_loss += sum(len(re.findall(r'\b' + re.escape(word) + r'\b', talk)) for word in bad_words)
    return word_based_loss

def wordBasedLoss(conversations, train_llm_num):
    return [inividual_wordBasedLoss(c, train_llm_num) for c in conversations]


########### Previous common part ###########

# other_name=None indicates that the strategy to estimate is llm's strategy
# player_num is that of the player whose strategy is being estimated
def estimate_strategy(llm, queries, Game, player_num, other_name=None, return_queries=False):

    possible_moves = Game.get_possible_moves(player_num)
    possible_tokens = llm.tokenizer([" " + name for name in list(possible_moves)], return_tensors='pt')['input_ids'][:,-1]
    
    sample_move = possible_moves[0]
    sample_token = possible_tokens[0]

    queries = [q.removesuffix("<think>") + "<play> " + sample_move for q in queries] if other_name == None else \
        [q + f" I think {other_name} will play " + sample_move for q in queries]
            
    tokenized = llm.tokenizer(queries, padding=True, truncation=True, return_tensors='pt').to('cuda')
    assert (sample_token == tokenized['input_ids'][:,-1]).all()

    probs_list = []
    with torch.no_grad():
        for i in range(tokenized["input_ids"].shape[0]):  # Iterate over sentences (memory issues)
            output = llm.model(
                input_ids=tokenized["input_ids"][i].unsqueeze(0),
                attention_mask=tokenized["attention_mask"][i].unsqueeze(0),
            )
            logits = output.logits[:, -2, possible_tokens]
            probs = logits.softmax(dim=-1)
            probs_list.append(probs)
    probs = torch.cat(probs_list, dim=0)

    strat = [{possible_moves[idx] : vec[idx].item() for idx in range(len(vec))} for vec in probs]
 
    return strat if not return_queries else (strat, queries)


@simple_cache
def compute_strategies_and_estimation(conversations, train_llm_num, train_llm, opponent_llm):
    print("Computing new strategies and estimations", flush=True)

    partial_conversations = [list(c.get_subconversations(train_llm_num)) for c in conversations]
    parts_per_conversation = [len(c) for c in partial_conversations]
    partial_conversations = sum(partial_conversations, [])

    Game = type(conversations[0].game)
    p1_estimation = estimate_strategy(
        train_llm, [c.get_query() for c in partial_conversations], Game=Game, player_num=3-train_llm_num, other_name="user"
    )
    p1_strategy = estimate_strategy(
        train_llm, [c.get_query() for c in partial_conversations], Game=Game, player_num=train_llm_num, other_name=None
    )
    p2_strategy = estimate_strategy(
        opponent_llm, [c.get_query(other_player=True) for c in partial_conversations], Game=Game, player_num=3-train_llm_num, other_name=None
    )
    
    part_indices = [0] + list(accumulate(parts_per_conversation))
    p1_estimation = [ p1_estimation[start:end] for start, end in zip(part_indices[:-1], part_indices[1:]) ]
    p1_strategy = [ p1_strategy[start:end] for start, end in zip(part_indices[:-1], part_indices[1:]) ]
    p2_strategy = [ p2_strategy[start:end] for start, end in zip(part_indices[:-1], part_indices[1:]) ]

    return p1_estimation, p1_strategy, p2_strategy

### ev func

def RPS_score(move1, move2):
    mapping = {"rock": 0, "paper": 1, "scissors": 2}
    score = (mapping[move1] - mapping[move2] + 3) % 3
    score = 0 if score == 2 else score+1
    return score

def get_allmoves_ev(estimation, moves, game):
    if game.NAME == RPS.NAME:
        return {
            move : sum(RPS_score(move, move2) * estimation[move2] for move2 in estimation) 
        for move in moves}
    elif game.NAME == BertrandCompetition.NAME:
        cost, max_price, demand_den = game.cost, game.max_price_with_demand, game.demand_den

        estimation = [estimation["$" + str(i)] for i in range(len(estimation))]
        demand_prob = 0.
        allmoves_ev = []
        for i in range(len(estimation)-1, -1, -1):
            allmoves_ev.append(
                estimation[i] * (i-cost) * max(0, (max_price - i) // (2*demand_den))
                + demand_prob * (i-cost) * max(0, (max_price - i) // demand_den)
            )
            demand_prob += estimation[i]
        allmoves_ev = list(reversed(allmoves_ev))
        return { "$" + str(i) : allmoves_ev[i] for i in range(len(allmoves_ev)) }


########### Internal State Evaluation ###########
# i.e. How well does my internal state adjust to the opponent strategy

# kl divergence computation specifically for the case of p, q being dictionaries
def kl_div(p, q):
    kl_div = 0.0
    for event, p_prob in p.items():
        q_prob = q[event]
        kl_div += p_prob * math.log(p_prob / q_prob)
    return kl_div

def internalStateEvaluation(conversations, train_llm_num, train_llm, opponent_llm, lastTurn=False):
    if len(conversations) == 0 or type(conversations[0].game) == SizePrizeGame: return [0.0] * len(conversations)

    p1_estimation, _, p2_strategy = compute_strategies_and_estimation(conversations, train_llm_num, train_llm, opponent_llm)

    if lastTurn: return [
        kl_div(conv_est[-1], conv_strategy[-1]) if len(conv_strategy) > 0 else 0.0
        for conv_est, conv_strategy in zip(p1_estimation, p2_strategy)
    ]
    else: return [
        float(np.mean([
            kl_div(est, strategy) for est, strategy in zip(conv_est, conv_strategy)
        ])) if len(conv_strategy) > 0 else 0.0
        for conv_est, conv_strategy in zip(p1_estimation, p2_strategy)
    ]


########### State-Relative Performance ###########
# i.e. How well do I perform conditioned to my internal state

def individual_stateRelativePerformance(estimation, strategy, game):
    moves_evs = get_allmoves_ev(estimation, strategy.keys(), game)
    best_ev, worse_ev = max(moves_evs.values()), min(moves_evs.values())

    my_ev = sum(strategy[move] * moves_evs[move] for move in strategy)
    return (my_ev - worse_ev) / (best_ev - worse_ev)

def stateRelativePerformance(conversations, train_llm_num, train_llm, opponent_llm, lastTurn=False):
    if len(conversations) == 0 or type(conversations[0].game) == SizePrizeGame: return [0.0] * len(conversations)
    
    p1_estimation, p1_strategy, _ = compute_strategies_and_estimation(conversations, train_llm_num, train_llm, opponent_llm)

    if lastTurn: return [
        individual_stateRelativePerformance(conv_est[-1], conv_strategy[-1], c.game) if len(conv_strategy) > 0 else 0.0
        for conv_est, conv_strategy, c in zip(p1_estimation, p1_strategy, conversations)
    ]
    else: return [
        float(np.mean([
            individual_stateRelativePerformance(est, strategy, c.game) for est, strategy in zip(conv_est, conv_strategy)
        ])) if len(conv_strategy) > 0 else 0.0
        for conv_est, conv_strategy, c in zip(p1_estimation, p1_strategy, conversations)
    ]


########### Leverage Opportunity ###########
# i.e. How much e.v. can I get against the opponent's strategy

def leverageOpportunity(conversations, train_llm_num, train_llm, opponent_llm, lastTurn=True):
    if len(conversations) == 0 or type(conversations[0].game) == SizePrizeGame: return [0.0] * len(conversations)
    
    _, _, p2_strategy = compute_strategies_and_estimation(conversations, train_llm_num, train_llm, opponent_llm)

    moves = conversations[0].game.get_possible_moves(train_llm_num)
    if lastTurn: return [
        max(get_allmoves_ev(conv_strategy[-1], moves, c.game).values()) if len(conv_strategy) > 0 else 0.
        for conv_strategy, c in zip(p2_strategy, conversations)
    ]
    else: return [
        float(np.mean([
            max(get_allmoves_ev(conv_strategy[-1], moves, c.game).values())
        ])) if len(conv_strategy) > 0 else 0.
        for conv_strategy, c in zip(p2_strategy, conversations)
    ]

###################### Reward Functions ######################

########### Game Reward ###########

@simple_cache
def finish_conversation_from_completion(
        completions, conversation, train_llm, opponent_llm, train_llm_num, config
    ):
    conversations = [deepcopy(c) for c in conversation]
    for idx, action in enumerate(completions): conversations[idx].turn(action)
    return finish_conversations(conversations, train_llm, opponent_llm, train_llm_num, config)

def game_reward(
        prompts, completions, conversation, train_llm_num, # from current batch
        train_llm, opponent_llm, conversation_file, config # general
    ):
    print("\nComputing rewards", flush=True)
    train_llm_num = train_llm_num[0]
    if completions is not None:
        conversation = finish_conversation_from_completion(completions, conversation, train_llm, opponent_llm, train_llm_num, config)

    rewards = [c.game.score(train_llm_num) for c in conversation]

    print('train conversations', file=conversation_file)
    for c, r in zip(conversation, rewards):
        print("CONVERSATION:\n", c, file=conversation_file)
        print("REWARD:", r, '\n'*3, flush=True, file=conversation_file)

    print("Rewards computed", flush=True)
    return rewards


########### Leverage Opportunity as a reward function ###########

@simple_cache
def compute_end_strategy(conversations, llm_num, llm):
    partial_conversations = [list(c.get_subconversations(llm_num))[-1] for c in conversations]

    Game = type(conversations[0].game)
    return estimate_strategy(
        llm, [c.get_query() for c in partial_conversations], Game=Game, player_num=llm_num, other_name=None, return_queries=True
    )
    
def leverageOpportunity_reward(
        prompts, completions, conversation, train_llm_num, # from current batch
        train_llm, opponent_llm, conversation_file, config # general
    ):
    if len(conversation) == 0 or type(conversation[0].game) == SizePrizeGame: return [0.0] * len(conversation)

    train_llm_num = train_llm_num[0]
    if completions is not None:
        conversation = finish_conversation_from_completion(completions, conversation, train_llm, opponent_llm, train_llm_num, config)
    p2_strategy, queries = compute_end_strategy(conversation, 3-train_llm_num, opponent_llm)

    moves = conversation[0].game.get_possible_moves(train_llm_num)
    rewards = [
        max(get_allmoves_ev(conv_strategy, moves, c.game).values())
        for conv_strategy, c in zip(p2_strategy, conversation)
    ]
    
    print('train conversations (leverageOpportunity_reward)', file=conversation_file)
    for x in range(len(conversation)):
        print("CONVERSATION:\n", queries[x], file=conversation_file)
        print("strategy :", p2_strategy[x], flush=True, file=conversation_file)
        print("leverageOpportunity_reward :", rewards[x], '\n'*3, flush=True, file=conversation_file)

    return rewards

########### Naturalness reward ###########

def naturalness_reward(
        prompts, completions, conversation, train_llm_num, # from current batch
        judge, naturalness_prompt, conversation_file, threshold, config # general
    ):
    examples = ["Naturalness score: Yes\n", "Naturalness score: No\n"]
    yes_no_ids = judge.tokenizer(examples, return_tensors="pt")["input_ids"][:, -2] # [7566, 2360]

    template = '\nResponse: "{text}"\nNaturalness score: {score}\n'
    score_pos = []
    if completions is not None:
        for text in completions:
            naturalness_prompt += template.format(text=text, score="Yes" if random.random() > 0.5 else "No")
            score_pos.append(len(naturalness_prompt)-2)
    else:
        actions_per_conversation = [0] * len(conversation)
        for i, conv in enumerate(conversation):
            for action in conv.get_player(train_llm_num).parsed_actions:
                if 'talk' in action:
                    naturalness_prompt += template.format(text=action['talk'], score="Yes" if random.random() > 0.5 else "No")
                    score_pos.append(len(naturalness_prompt)-2)
                    actions_per_conversation[i] += 1


    tokenized = judge.tokenizer(naturalness_prompt, return_offsets_mapping=True)
    input_ids = torch.tensor(tokenized['input_ids']).unsqueeze(0).to('cuda')
    offsets = tokenized['offset_mapping']

    score_tokens = []
    j = 0  # score_pos pointer
    for i in range(len(offsets)):
        if offsets[i][0] <= score_pos[j] < offsets[i][1]:
            score_tokens.append(i-1)
            j += 1
            if j >= len(score_pos): break
    
    with torch.no_grad():
        out = judge.model(input_ids=input_ids)
    yes_no_logits = out.logits[:, score_tokens][:,:,yes_no_ids]
    probs = yes_no_logits.softmax(dim=-1)
    naturalness_reward = probs[0,:,0].tolist()

    if threshold is not None:
        naturalness_reward = [1.0 if r > threshold else 0.0 for r in naturalness_reward]

    print('train conversations, naturalness_reward prompt', file=conversation_file)
    print(naturalness_prompt, '\n', file=conversation_file)
    print("naturalness_reward :", naturalness_reward, '\n'*3, flush=True, file=conversation_file)

    if completions is None:
        limit_idx = [0] + list(accumulate(actions_per_conversation))
        naturalness_reward = [
            sum(naturalness_reward[start:end]) / (end - start) if end != start else 0.0
            for start, end in zip(limit_idx[:-1], limit_idx[1:])
        ]

        print('actions_per_conversation:', actions_per_conversation, file=conversation_file)
        print("naturalness_reward :", naturalness_reward, '\n'*3, flush=True, file=conversation_file)

    return naturalness_reward
