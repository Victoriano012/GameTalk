import numpy as np
import torch
import math
import re

from itertools import accumulate
from functools import partial

from games import RPS, BertrandCompetition, SizePrizeGame
from utils import simple_cache

def get_eval_metrics(train_llm, opponent_llm):
    return {
        "word_based_loss" : wordBasedLoss,
        "internal_state_evaluation" : partial(internalStateEvaluation, train_llm=train_llm, opponent_llm=opponent_llm),
        "state_relative_performance" : partial(stateRelativePerformance, train_llm=train_llm, opponent_llm=opponent_llm),
        "leverage_opportunity" : partial(leverageOpportunity, train_llm=train_llm, opponent_llm=opponent_llm),
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
def estimate_strategy(llm, queries, Game, player_num, other_name=None):

    possible_moves = Game.get_possible_moves(player_num)
    possible_tokens = llm.tokenizer([" " + name for name in list(possible_moves)], return_tensors='pt')['input_ids'][:,-1]
    
    sample_move = possible_moves[0]
    sample_token = possible_tokens[0]

    if other_name == None:
        think_token = "<think>"
        for idx in range(len(queries)):
            if queries[idx].endswith(think_token):
                queries[idx] = queries[idx][:len(think_token)]
            queries[idx] += f"<play> " + sample_move
    else:
        for idx in range(len(queries)):
            queries[idx] += f" I think {other_name} will play " + sample_move
            
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

    return [{possible_moves[idx] : vec[idx].item() for idx in range(len(vec))} for vec in probs]


@simple_cache
def compute_estrategies_and_estimation(conversations, train_llm_num, train_llm, opponent_llm):
    print("Computing new strategies and estimations", flush=True)

    partial_conversations = [list(c.get_subconversations(train_llm_num)) for c in conversations]
    parts_per_conversation = [len(c) for c in partial_conversations]
    partial_conversations = sum(partial_conversations, [])

    Game = type(conversations[0].game)
    p1_estimation = estimate_strategy(
        train_llm, [c.get_query() for c in partial_conversations], Game=Game, player_num=2-train_llm_num, other_name="user"
    )
    p1_strategy = estimate_strategy(
        train_llm, [c.get_query() for c in partial_conversations], Game=Game, player_num=train_llm_num, other_name=None
    )
    p2_strategy = estimate_strategy(
        opponent_llm, [c.get_query(other_player=True) for c in partial_conversations], Game=Game, player_num=2-train_llm_num, other_name=None
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
    if type(game) == RPS:
        return {
            move : sum(RPS_score(move, move2) * estimation[move2] for move2 in estimation) 
        for move in moves}
    elif type(game) == BertrandCompetition:
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

def internalStateEvaluation(conversations, train_llm_num, train_llm, opponent_llm):
    if len(conversations) == 0 or type(conversations[0].game) == SizePrizeGame: return [0.0] * len(conversations)

    p1_estimation, _, p2_strategy = compute_estrategies_and_estimation(conversations, train_llm_num, train_llm, opponent_llm)

    return [
        float(np.mean([
            kl_div(est, strategy) for est, strategy in zip(conv_est, conv_strategy)
        ])) 
        for conv_est, conv_strategy in zip(p1_estimation, p2_strategy)
    ]


########### State-Relative Performance ###########
# i.e. How well do I perform conditioned to my internal state

def individual_stateRelativePerformance(estimation, strategy, game):
    moves_evs = get_allmoves_ev(estimation, strategy.keys(), game)
    best_ev, worse_ev = max(moves_evs.values()), min(moves_evs.values())

    my_ev = sum(strategy[move] * moves_evs[move] for move in strategy)
    return (my_ev - worse_ev) / (best_ev - worse_ev)

def stateRelativePerformance(conversations, train_llm_num, train_llm, opponent_llm):
    if len(conversations) == 0 or type(conversations[0].game) == SizePrizeGame: return [0.0] * len(conversations)
    
    p1_estimation, p1_strategy, _ = compute_estrategies_and_estimation(conversations, train_llm_num, train_llm, opponent_llm)

    return [
        float(np.mean([
            individual_stateRelativePerformance(est, strategy, c.game) for est, strategy in zip(conv_est, conv_strategy)
        ]))
        for conv_est, conv_strategy, c in zip(p1_estimation, p1_strategy, conversations)
    ]


########### Leverage Opportunity ###########
# i.e. How much e.v. can I get against the opponent's strategy

def leverageOpportunity(conversations, train_llm_num, train_llm, opponent_llm):
    if len(conversations) == 0 or type(conversations[0].game) == SizePrizeGame: return [0.0] * len(conversations)
    
    _, _, p2_strategy = compute_estrategies_and_estimation(conversations, train_llm_num, train_llm, opponent_llm)

    moves = conversations[0].game.get_possible_moves(train_llm_num)
    return [
        max(get_allmoves_ev(conv_strategy[-1], moves, c.game).values())
        for conv_strategy, c in zip(p2_strategy, conversations) if len(conv_strategy) > 0
    ]
