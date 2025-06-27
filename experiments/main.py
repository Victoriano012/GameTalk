from transformers import EarlyStoppingCallback

from functools import partial, update_wrapper
from omegaconf import OmegaConf
from pickle import load
import inspect
import hydra
import wandb
import sys
import os

from BaseCustomEvalTrainer import BaseCustomEvalTrainer
from CustomSTaRTrainer import CustomSTaRTrainer, CustomSTaRConfig
from CustomGRPOTrainer import CustomGRPOTrainer, CustomGRPOConfig
from CustomDPOTrainer import CustomDPOTrainer, CustomDPOConfig
from game_dataset import GameDataset, OutdateDatasetCallback, MetricsLogger
from llm_utils import LLM
from metrics import get_eval_metrics, game_reward, internalStateEvaluation_reward, leverageOpportunity_reward, naturalness_reward
from utils import get_wandb_id


@hydra.main(config_path='config', config_name='config', version_base=None)
def __main__(config):
    
    ### Logging ###
    os.makedirs(f"{config.logs.folder}{config.run_name}", exist_ok=True)
    write_mode = "a" if config.train.resume_from_checkpoint else "w"
    general_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.general}", write_mode)
    metics_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.metrics}", write_mode)
    conversation_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.conversations}", write_mode)
    eval_conversation_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.eval_conversations}", write_mode)
    sys.stdout = general_file
    sys.stderr = general_file

    wandb.init(
        config = OmegaConf.to_container(config, resolve=True),
        name = config.run_name,
        resume = "allow" if config.train.resume_from_checkpoint else "never",
        id = get_wandb_id(config.run_name, config.train.resume_from_checkpoint),
        **config.wandb
    )

    ### Models ###
    train_llm = LLM(config.llms.train_llm_name, lora_config=config.lora, unsloth=config.llms.unsloth).to('cuda')
    opponent_llm = LLM(config.llms.opponent_llm_name, unsloth=config.llms.unsloth).to('cuda')

    ### Dataset ###
    keep_partial_conversations = config.train.method != 'star' # ugly
    with open(config.dataset.data, 'rb') as f:
        data = load(f)
    dataset = GameDataset(train_llm, opponent_llm, data, config, keep_partial_conversations=keep_partial_conversations)

    ### Callbacks ###
    dataset_callback = OutdateDatasetCallback(dataset)
    metrics_logger = MetricsLogger(dataset, metics_file, conversation_file, eval_conversation_file, config)
    earlyStop = EarlyStoppingCallback(config.train.early_stopping_patience)
    callbacks = [dataset_callback, metrics_logger, earlyStop]

    ### Reward function ###
    reward_funcs = [game_reward]
    reward_weights = [1.]

    if config.train.internalStateEvaluation_weight:
        reward_funcs.append(internalStateEvaluation_reward)
        reward_weights.append(config.train.internalStateEvaluation_weight)

    if config.train.leverageOpportunity_weight:
        reward_funcs.append(leverageOpportunity_reward)
        reward_weights.append(config.train.leverageOpportunity_weight)
    
    naturalness_prompt = None
    if config.train.naturalness_weight:
        with open(config.prompts.naturalness_prompt, 'r') as f:
            naturalness_prompt = f.read()
        reward_funcs.append(naturalness_reward)
        reward_weights.append(config.train.naturalness_weight)
    
    reward_args = {'train_llm': train_llm, 'opponent_llm': opponent_llm, 'judge': opponent_llm, 'conversation_file': conversation_file, 'naturalness_prompt': naturalness_prompt, 'threshold' : config.train.naturalness_threshold, 'config': config}
    for i in range(len(reward_funcs)):
        reward_mod = partial(reward_funcs[i], **{ name: value for name, value in reward_args.items() if name in inspect.signature(reward_funcs[i]).parameters})
        update_wrapper(reward_mod, reward_funcs[i])
        reward_funcs[i] = reward_mod

    ### Trainer + Config ###
    Config = {
        "grpo" : CustomGRPOConfig,
        "dpo" : CustomDPOConfig,
        "star" : CustomSTaRConfig,
    }[config.train.method]
    BaseTrainer = {
        "grpo" : CustomGRPOTrainer,
        "dpo" : CustomDPOTrainer,
        "star" : CustomSTaRTrainer,
    }[config.train.method]
    class Trainer(BaseCustomEvalTrainer, BaseTrainer): pass

    Config_params = set(inspect.signature(Config.__init__).parameters)
    training_args = {k: v for k, v in config.train.items() if k in Config_params}
    training_args = Config(
        **training_args,

        reward_weights=reward_weights,

        output_dir=config.logs.folder + config.run_name,
        run_name=config.run_name
    )

    trainer = Trainer(
        model=train_llm.model,
        processing_class=train_llm.tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        callbacks=callbacks,

        eval_funcs=get_eval_metrics(train_llm, opponent_llm)
    )

    ### train ###
    trainer.train(resume_from_checkpoint=config.train.resume_from_checkpoint)

if __name__ == "__main__":
    __main__()
