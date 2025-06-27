# GameTalk

This repository has the code of my Bachelor's thesis *Learning to Converse Strategically: A Game-Based Approach for LLMs*.

The goal of the thesis is to train LLMs to strategically converse through RL, i.e. to have a conversation having into account its final goal.
To do so, two LLMs are set to have a conversation with each other and they will be playing a game in parallel to such conversation. One of them is being trained and the other isn't.
The rewards used for training are those provided by the game, and through the training process, the LLM learns to use the conversation to understand the counterpart's intentions and even to try to manipulate them, in order to obtain more reward.

If you want more details about it, please check the publication of the thesis ~~here~~.

## Repository Structure
- [notebooks/](notebooks/) : Folder with some notebooks I used for analysing and presenting the data.
- [experiments/](experiments/) : Folder with all the necessary files to run the experiments of the project.
  - [main.py](experiments/main.py) : The main file, that should be used to run any experiment through `python3 main.py {config options}`. If no configuration options are defined, those in `experiments/config/config.yaml` will be used.
  - Other `.py` files : Auxiliary files imported by `main.py`. These could be split in:
    - Classes that inherit from HuggingFace Trainers : [CustomGRPOTrainer.py](experiments/CustomGRPOTrainer.py), [CustomDPOTrainer.py](experiments/CustomDPOTrainer.py), [CustomSTaRTrainer.py](experiments/CustomSTaRTrainer.py), [BaseCustomEvalTrainer.py](experiments/BaseCustomEvalTrainer.py)
    - Standalone classes : [conversation_manager.py](experiments/conversation_manager.py), [games.py](experiments/games.py), [game_dataset.py](experiments/game_dataset.py), [llm_utils.py](experiments/llm_utils.py)
    - Utility files : [metrics.py](experiments/metrics.py), [utils.py](experiments/utils.py)
  - [config/](experiments/config/) : Folder with all config files. [config.yaml](experiments/config/config.yaml) is the one used by default, and others can be used to overwrite it when calling `main.py` with `+folder=file`, for example calling `python3 main.py +methods=dpo +games=SPBargaining`.
  - [data/](experiments/data/) : Folder with files to generate the data (`.py`) used in the different games, and data files themselves (`.pkl`).
  - [prompts/](experiments/prompts/) : Folder with text files with all the prompts used
  - [logs/](experiments/logs/) : Empty folder where all the logs of the experiments are stored (not empty in my local).
