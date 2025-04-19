from trl import GRPOConfig, GRPOTrainer
from functools import partial, update_wrapper
import hydra

from game_dataset import GameDataset, OutdateDatasetCallback, reward
from llm_utils import LLM


@hydra.main(config_path='config', config_name='config', version_base=None)
def __main__(config):

    train_llm = LLM(config.llms.train_llm_name, lora_config=config.lora, unsloth=config.llms.unsloth).to('cuda')
    opponent_llm = LLM(config.llms.opponent_llm_name, unsloth=config.llms.unsloth).to('cuda')

    dataset = GameDataset(train_llm, opponent_llm, config)
    dataset_callback = OutdateDatasetCallback(dataset)
    
    reward_mod = partial(reward, Game=dataset.Game, train_llm=train_llm, opponent_llm=opponent_llm, config=config)
    update_wrapper(reward_mod, reward)

    training_args = GRPOConfig(
        output_dir=config.logs.folder + config.run_name,
        run_name=config.run_name,
        logging_steps=1,
        num_train_epochs=config.train.epochs,
        max_completion_length=config.train.max_new_tokens,
        max_prompt_length=None,

        learning_rate=config.train.lr,
        beta=config.train.kl_coef,

        auto_find_batch_size = True
    )
    trainer = GRPOTrainer(
        model=train_llm.model,
        processing_class=train_llm.tokenizer,
        reward_funcs=reward_mod,
        args=training_args,
        train_dataset=dataset,
        callbacks=[dataset_callback]
    )
    trainer.train()

if __name__ == "__main__":
    __main__()
