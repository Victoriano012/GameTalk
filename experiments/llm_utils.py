from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
import torch.nn as nn
import torch

def get_actual_llm_name(llm_name):
    supported_llms = {
        'llama-3B' : "meta-llama/Llama-3.2-3B-Instruct",
        'llama-8B' : "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen-4B" : "Qwen/Qwen3-4B",
    }

    if llm_name in supported_llms:
        return supported_llms[llm_name]
    if llm_name in supported_llms.values():
        return llm_name
    
    raise Exception('Non-supported LLM')


class LLM(nn.Module):
    def __init__(self, llm_name, lora_config=None, unsloth=False):
        super().__init__()
        llm_name = get_actual_llm_name(llm_name)

        if unsloth:
            raise Exception('Unsloth is not supported, it breaks evaluation')
            if 'FastLanguageModel' not in globals():
                from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(model_name = llm_name)
            if lora_config is not None:
                self.model = FastLanguageModel.get_peft_model(self.model, use_gradient_checkpointing = "unsloth", **lora_config)
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
            pad_token_id=128009, # llm.tokenizer.eos_token
            **kwargs
        )
        output = output[:, inputs['input_ids'].shape[-1]:]
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)
    
    # Reference: https://www.tylerromero.com/posts/2025-02-selective-log-softmax/
    def get_log_probs(self, input, attention_mask=None):
        logits = self.model(input, attention_mask=attention_mask).logits
        logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
        token_logits = torch.gather(logits, dim=-1, index=input.unsqueeze(-1)).squeeze(-1)
        token_logprobs = token_logits - logsumexp_values
        return token_logprobs

