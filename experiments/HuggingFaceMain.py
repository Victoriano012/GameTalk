from trl import GRPOConfig, GRPOTrainer
from transformers import EarlyStoppingCallback
from Trainer_mod import TrainerCustomEval

from functools import partial, update_wrapper
import inspect
import hydra
import sys
import os

from game_dataset import GameDataset, OutdateDatasetCallback, MetricsLogger, game_reward
from llm_utils import LLM


@hydra.main(config_path='config', config_name='config', version_base=None)
def __main__(config):
    
    os.makedirs(f"{config.logs.folder}{config.run_name}", exist_ok=True)
    general_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.general}", "w")
    metics_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.metrics}", "w")
    conversation_file = open(f"{config.logs.folder}{config.run_name}/{config.logs.conversations}", "w")
    sys.stdout = general_file
    sys.stderr = general_file
    os.environ["WANDB_PROJECT"] = config.wandb.project

    train_llm = LLM(config.llms.train_llm_name, lora_config=config.lora, unsloth=config.llms.unsloth).to('cuda')
    opponent_llm = LLM(config.llms.opponent_llm_name, unsloth=config.llms.unsloth).to('cuda')

    dataset = GameDataset(train_llm, opponent_llm, config)
    dataset_callback = OutdateDatasetCallback(dataset)
    metrics_logger = MetricsLogger(dataset, metics_file, conversation_file, config)
    earlyStop = EarlyStoppingCallback(config.train.early_stopping_patience)
    callbacks = [dataset_callback, metrics_logger, earlyStop]

    reward_mod = partial(game_reward, Game=dataset.Game, train_llm=train_llm, opponent_llm=opponent_llm, config=config)
    update_wrapper(reward_mod, game_reward)
    
    GRPOConfig_params = set(inspect.signature(GRPOConfig.__init__).parameters)
    training_args = {k: v for k, v in config.train.items() if k in GRPOConfig_params}
    training_args = GRPOConfig(
        **training_args,
        
        output_dir=config.logs.folder + config.run_name,
        run_name=config.run_name,

        eval_strategy="steps",
        max_prompt_length=None,
        logging_first_step = True,
        metric_for_best_model = "reward",

        temperature=config.train.trained_temperature,
    )

    trainer = TrainerCustomEval(
        model=train_llm.model,
        processing_class=train_llm.tokenizer,
        reward_funcs=reward_mod,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        callbacks=callbacks,

        eval_samples=config.train.eval_samples
    )
    trainer.train()

if __name__ == "__main__":
    __main__()
