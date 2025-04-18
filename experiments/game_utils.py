import torch
from math import log


def masked_call(cls, queries, mask, unpack=True):
    filtered_inputs = [q for q, m in zip(queries, mask) if m] # Extract elements where mask is 1
    if filtered_inputs == []:
        return [""]*len(queries)
    
    filtered_outputs = cls(filtered_inputs)                   # Call cls once with all necessary elements
    if not unpack:
        return filtered_outputs
    
    output_iter = iter(filtered_outputs)                      # Iterator to retrieve processed elements
    return [next(output_iter) if m else "" for m in mask]



def get_end_tokens(tokenizer, Game):
    moves = {item.value : item for item in Game if not item.is_error()}
    ends = [" " + name for name in list(moves)]
    tokens = tokenizer(ends, return_tensors='pt')['input_ids'][:,1:].squeeze()
    assert len(tokens.shape) == 1, "Game actions are more than one token long, NotImplementedError"
    return {tokens[idx].item() : moves[ends[idx][1:]] for idx in range(len(ends))}

# other_name=None indicates that the strategy to estimate is llm's strategy
def estimate_strategy(llm, queries, Game, other_name=None):

    sample_move = [item.value for item in Game if not item.is_error()][0]
    sample_token = llm.tokenizer(" " + sample_move)['input_ids'][1]

    if other_name == None:
        think_token = "<think>"
        for idx in range(len(queries)):
            if queries[idx][len(think_token):] == think_token:
                queries[idx] = queries[idx][:len(think_token)]
        for idx in range(len(queries)):
            queries[idx] += f"<play> " + sample_move
    else:
        for idx in range(len(queries)):
            queries[idx] += f" I think {other_name} will play " + sample_move
            
    token_to_instance = get_end_tokens(llm.tokenizer, Game)
    tokens = list(token_to_instance)

    tokenized = llm.tokenizer(queries, padding=True, truncation=True, return_tensors='pt').to('cuda')

    assert (sample_token == tokenized['input_ids'][:,-1]).all()

    probs_list = []
    with torch.no_grad():
        for i in range(tokenized["input_ids"].shape[0]):  # Iterate over sentences (memory issues)
            output = llm.model(
                input_ids=tokenized["input_ids"][i].unsqueeze(0),
                attention_mask=tokenized["attention_mask"][i].unsqueeze(0),
            )
            logits = output.logits[:, -1, tokens]
            probs = logits.softmax(dim=-1)
            probs_list.append(probs)
    probs = torch.cat(probs_list, dim=0)

    return [{token_to_instance[tokens[idx]] : vec[idx].item() for idx in range(len(vec))} for vec in probs]
    
# kl divergence computation specifically for the case of p, q being dictionaries
def kl_div(p, q):
    kl_div = 0.0
    for event, p_prob in p.items():
        q_prob = q[event]
        kl_div += p_prob * log(p_prob / q_prob)
    return kl_div


def internalStateEvaluation(llm_trained, llm_opponent, Game, conversations):
    player_1_estimation = masked_call(
        lambda q : estimate_strategy(llm_trained, q, Game, other_name="user"),
        queries= [c.get_query() for c in conversations],
        mask   = [not c.finished() for c in conversations],
        unpack = False
    )
    player_2_strategy = masked_call(
        lambda q : estimate_strategy(llm_opponent, q, Game, other_name=None),
        queries= [c.get_query(other_player=True) for c in conversations],
        mask   = [not c.finished() for c in conversations],
        unpack = False
    )

    kl = 0.0
    for x in range(len(player_1_estimation)):
        kl += kl_div(player_1_estimation[x], player_2_strategy[x])
    kl /= len(player_1_estimation)
    return kl
