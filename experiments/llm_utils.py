# from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
from peft import get_peft_model, LoraConfig
import torch.nn as nn
import torch

def get_actual_llm_name(llm_name, unsloth=False):
    if unsloth:
        llm_name = 'unsloth-' + llm_name
    supported_llms = {
        'llama-3B' : "meta-llama/Llama-3.2-3B-Instruct",
        'llama-8B' : "meta-llama/Llama-3.1-8B-Instruct",
        'unsloth-llama-3B' : "unsloth/Llama-3.2-3B-Instruct",
        'unsloth-llama-8B' : "unsloth/Llama-3.1-8B-Instruct"
    }
    if llm_name in supported_llms:
        return supported_llms[llm_name]
    if llm_name in supported_llms.values():
        return llm_name
    raise Exception('Non-supported LLM')


class LLM(nn.Module):
    def __init__(self, llm_name, stopping_criteria=None, lora_config=None, unsloth=False):
        super().__init__()
        self.stopping_criteria = stopping_criteria
        llm_name = get_actual_llm_name(llm_name, unsloth)

        if unsloth:
            raise Exception("Unsloth doesn't work, don't use it")
            pass
            # self.model, self.tokenizer = FastLanguageModel.from_pretrained(model_name = llm_name)
            # if lora_config is not None:
            #     self.model = FastLanguageModel.get_peft_model(self.model, **lora_config)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name, padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(llm_name)
            if lora_config is not None:
                lora_config = LoraConfig(task_type="CAUSAL_LM", **lora_config)
                self.model = get_peft_model(self.model, lora_config)
        

    def generate(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt,padding=True,truncation=True, return_tensors="pt").to('cuda')
        output = self.model.generate(
            **inputs,
            stopping_criteria=self.stopping_criteria(self.tokenizer) if self.stopping_criteria is not None else None,
            pad_token_id=128009, # llm.tokenizer.eos_token
            **kwargs
        )
        output = output[:, inputs['input_ids'].shape[-1]:]
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)
    
    # def get_log_probs(self, input, attention_mask=None):
    #     output = self.model(input, attention_mask=attention_mask)
    #     log_prob = output.logits.log_softmax(dim=-1)
    #     log_prob = log_prob[torch.arange(input.shape[0]).unsqueeze(1), torch.arange(input.shape[1]), input].reshape(input.shape)
    #     return log_prob
    
    # Reference: https://www.tylerromero.com/posts/2025-02-selective-log-softmax/
    def get_log_probs(self, input, attention_mask=None):
        logits = self.model(input, attention_mask=attention_mask).logits
        logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
        token_logits = torch.gather(logits, dim=-1, index=input.unsqueeze(-1)).squeeze(-1)
        token_logprobs = token_logits - logsumexp_values
        return token_logprobs


default_stopping_text = ["</talk>", "</play>"]
# default_stopping_text = ["</play>"]

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

