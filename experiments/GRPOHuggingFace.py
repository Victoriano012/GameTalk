from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from trl import GRPOConfig, GRPOTrainer
from bs4 import BeautifulSoup
from torch.utils.data import Dataset

class SingleStringDataset(Dataset):
    def __init__(self, string):
        self.string = string

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"prompt": self.string}

def score_one_completion(text):
    text = '<play>'+text
    soup = BeautifulSoup(text, 'html.parser')
    matches = [(tag.name, tag.get_text()) for tag in soup.find_all()]
    if len(matches) != 1:
        return -1.
    tag, content = list(matches)[0]
    if tag != "play":
        return -1.
    move = content.strip().lower()
    if move == 'scissors':
        return 0.
    elif move == 'rock':
        return 1.
    elif move == 'paper':
        return 2.
    return -1.

def reward(prompts, completions):
    return [score_one_completion(completion) for completion in completions]

with open('/home/azureuser/main/LLMGameTheory/experiments/prompts/RPS/just_play.txt', "r") as f:
    initial_prompt = f.read()
    initial_prompt += '<play>'
dataset = SingleStringDataset(initial_prompt)

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

lora_config = {'r': 32, 'lora_alpha': 64, 'lora_dropout': 0.0}
lora_config = LoraConfig(task_type="CAUSAL_LM", **lora_config)
model = get_peft_model(model, lora_config)

run_name = "no-kl"
training_args = GRPOConfig(
    output_dir="GRPOHuggingFace/" + run_name,
    run_name=run_name,
    logging_steps=1,
    num_train_epochs=300,
    max_completion_length=5,

    learning_rate=1e-4,
    beta=0.
)
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()