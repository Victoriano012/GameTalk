import numpy as np
import torch
import math
import re

from functools import partial

def get_eval_metrics(train_llm, opponent_llm):
    return {
        "word_based_loss" : wordBasedLoss,
        "internal_state_evaluation" : partial(internalStateEvaluation, train_llm=train_llm, opponent_llm=opponent_llm)
    }


########### Internal State Evaluation ###########

# kl divergence computation specifically for the case of p, q being dictionaries
def kl_div(p, q):
    kl_div = 0.0
    for event, p_prob in p.items():
        q_prob = q[event]
        kl_div += p_prob * math.log(p_prob / q_prob)
    return kl_div

# other_name=None indicates that the strategy to estimate is llm's strategy
def estimate_strategy(llm, queries, Game, other_name=None):

    possible_moves = Game.get_possible_moves()
    possible_tokens = llm.tokenizer([" " + name for name in list(possible_moves)], return_tensors='pt')['input_ids'][:,1:].squeeze()
    
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


def internalStateEvaluation(conversations, train_llm_num, llm_trained, llm_opponent):
    partial_conversations = [list(c.get_subconversations(train_llm_num)) for c in conversations]
    parts_per_conversation = [len(c) for c in partial_conversations]
    partial_conversations = sum(partial_conversations, [])

    Game = type(conversations[0].game)
    player_1_estimation = estimate_strategy(
        llm_trained, [c.get_query() for c in partial_conversations], Game=Game, other_name="user"
    )
    player_2_strategy = estimate_strategy(
        llm_opponent, [c.get_query(other_player=True) for c in partial_conversations], Game=Game, other_name=None
    )

    kl_divs = [kl_div(est, strategy) for est, strategy in zip(player_1_estimation, player_2_strategy)]
    return [float(np.mean(group)) for group in np.split(kl_divs, np.cumsum(parts_per_conversation)[:-1])]


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
