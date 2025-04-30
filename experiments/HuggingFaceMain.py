from trl import GRPOTrainer
from transformers import EarlyStoppingCallback

from functools import partial, update_wrapper
from pickle import load
import inspect
import hydra
import sys
import os

from BaseCustomEvalTrainer import BaseCustomEvalTrainer, CustomGRPOConfig
from CustomSTaRTrainer import CustomSTaRTrainer, CustomSTaRConfig
from CustomDPOTrainer import CustomDPOTrainer, CustomDPOConfig
from game_dataset import GameDataset, OutdateDatasetCallback, MetricsLogger, game_reward
from llm_utils import LLM


@hydra.main(config_path='config', config_name='config', version_base=None)
def __main__(config):
    
    os.environ["WANDB_PROJECT"] = config.wandb.project
    
    os.makedirs(f"{config.logs.folder}{config.run_name}", exist_ok=True)
    general_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.general}", "w")
    metics_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.metrics}", "w")
    conversation_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.conversations}", "w")
    eval_conversation_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.eval_conversations}", "w")
    sys.stdout = general_file
    sys.stderr = general_file

    train_llm = LLM(config.llms.train_llm_name, lora_config=config.lora, unsloth=config.llms.unsloth).to('cuda')
    opponent_llm = LLM(config.llms.opponent_llm_name, unsloth=config.llms.unsloth).to('cuda')

    keep_partial_conversations = config.train.method != 'star' # ugly
    with open(config.dataset.data, 'rb') as f:
        data = load(f)
    dataset = GameDataset(train_llm, opponent_llm, data, config, keep_partial_conversations=keep_partial_conversations)

    dataset_callback = OutdateDatasetCallback(dataset)
    metrics_logger = MetricsLogger(dataset, metics_file, conversation_file, eval_conversation_file, config)
    earlyStop = EarlyStoppingCallback(config.train.early_stopping_patience)
    callbacks = [dataset_callback, metrics_logger, earlyStop]

    reward_mod = partial(game_reward, train_llm=train_llm, opponent_llm=opponent_llm, conversation_file=conversation_file, config=config)
    update_wrapper(reward_mod, game_reward)
    
    Config = {
        "grpo" : CustomGRPOConfig,
        "dpo" : CustomDPOConfig,
        "star" : CustomSTaRConfig,
    }[config.train.method]
    BaseTrainer = {
        "grpo" : GRPOTrainer,
        "dpo" : CustomDPOTrainer,
        "star" : CustomSTaRTrainer,
    }[config.train.method]
    class Trainer(BaseCustomEvalTrainer, BaseTrainer): pass

    Config_params = set(inspect.signature(Config.__init__).parameters)
    training_args = {k: v for k, v in config.train.items() if k in Config_params}
    training_args = Config(
        **training_args,

        output_dir=config.logs.folder + config.run_name,
        run_name=config.run_name
    )

    trainer = Trainer(
        model=train_llm.model,
        processing_class=train_llm.tokenizer,
        reward_funcs=reward_mod,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        callbacks=callbacks,
    )
    trainer.train()

if __name__ == "__main__":
    __main__()
