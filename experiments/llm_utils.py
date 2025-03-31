from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
from peft import get_peft_model
import torch.nn as nn
import torch

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

