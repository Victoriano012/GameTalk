from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
from peft import get_peft_model
import torch.nn as nn
import torch

from bs4 import BeautifulSoup


class LLM(nn.Module):
    def __init__(self, llm_name, stopping_criteria=None, lora_config=None):
        super().__init__()
        self.stopping_criteria = stopping_criteria

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(llm_name)
        if lora_config is not None:
            self.model = get_peft_model(self.model, lora_config)
        

    def generate(self, prompt, max_new_tokens=2000):
        inputs = self.tokenizer(prompt,padding=True,truncation=True, return_tensors="pt").to('cuda')
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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
    def __init__(self, tokenizer, stopping_text = default_stopping_text):
        super().__init__()
        self.tokenizer = tokenizer
        self.stopping_text = stopping_text
        self.text = None

    def __call__(self, input_ids, scores):
        if self.text is None:
            self.text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            return False
        
        stop = []
        for i in range(input_ids.shape[0]):
            self.text[i] += self.tokenizer.decode(input_ids[i,-1:])
            stop.append(any(
                self.text[i].strip().endswith(st) for st in self.stopping_text
            ))

        return torch.tensor(stop).to('cuda')

one_turn_stop_criteria = lambda tokenizer: StoppingCriteriaList([one_player_generation(tokenizer = tokenizer)])



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


def masked_call(cls, queries, mask):
    filtered_inputs = [q for q, m in zip(queries, mask) if m] # Extract elements where mask is 1
    filtered_outputs = cls(filtered_inputs)                   # Call cls once with all necessary elements
    output_iter = iter(filtered_outputs)                      # Iterator to retrieve processed elements

    return [next(output_iter) if m else "" for m in mask]

