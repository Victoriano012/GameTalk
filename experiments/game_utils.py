from bs4 import BeautifulSoup
import torch

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


# generate response + solve format errors from early stopping 
def one_turn(llm, query, max_new_tokens=2000):
    text = llm.generate(query, max_new_tokens=max_new_tokens)

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
        text_2 = llm.generate(query_again, max_new_tokens=max_new_tokens)
        for idx, t in zip(reindex, text_2):
            text[idx] += " </think> " + t

    # finish unfinished talks
    for i, t in enumerate(text):
        talk_start = t.rfind("<talk>")
        talk_end = t.rfind("</talk>")
        if talk_end == -1 and talk_start > -1:
            text[i] += " </talk>\n"
    
    return ["<think>" + t for t in text]


def masked_call(cls, queries, mask, unpack=True):
    filtered_inputs = [q for q, m in zip(queries, mask) if m] # Extract elements where mask is 1
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
    with torch.no_grad():  # Disable gradient computation
        for i in range(tokenized["input_ids"].shape[0]):  # Iterate over sentences (memory issues)
            output = llm.model(
                input_ids=tokenized["input_ids"][i].unsqueeze(0),
                attention_mask=tokenized["attention_mask"][i].unsqueeze(0),
            )
            logits = output.logits[:, -1, tokens]
            probs = logits.softmax(dim=-1)
            probs_list.append(probs)
    probs = torch.cat(probs_list, dim=0)

    predicted_strategy = [{token_to_instance[tokens[idx]] : vec[idx].item() for idx in range(len(vec))} for vec in probs]

    return predicted_strategy



from math import log

# kl divergence computation specifically for the case of p, q being dictionaries
def kl_div(p, q):
    kl_div = 0.0
    for event, p_prob in p.items():
        q_prob = q[event]
        kl_div += p_prob * log(p_prob / q_prob)
    return kl_div