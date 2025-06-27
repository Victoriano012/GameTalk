This repository has the code of my Bachelor's thesis "*Learning to Converse Strategically: A Game-Based Approach for LLMs*".

The goal of the thesis is to train LLMs to strategically converse through RL, i.e. to have a conversation having into account its final goal.
To do so, two LLMs are set to have a conversation with each other and they will be playing a game in parallel to such conversation. One of them is being trained and the other isn't.
The rewards used for training are those provided by the game, and through the training process, the LLM learns to use the conversation to understand the counterpart's intentions and even to try to manipulate them, in order to obtain more reward.

If you want more details about it, please check the publication of the thesis ~~here~~.

This repo has the following structure: 
- `notebooks` : Folder with some notebooks I used for analysing and presenting the data.
- `experiments` : Folder with all the necessary files to run the experiments of the project.
| - `main.py` : The main file, that should be used to run any experiment through `python3 main.py {config options}`. If no configuration options are defined, those in `experiments/config/config.yaml` will be used.
| - Other files : Auxiliary files imported by `main.py`. These could be split in: Classes that inherit from HuggingFace Trainers (`CustomGRPOTrainer.py`, `CustomDPOTrainer.py`, `CustomSTaRTrainer.py`, `BaseCustomEvalTrainer.py`), standalone classes (`conversation_manager.py`, `games.py`, `game_dataset.py`, `llm_utils.py`), and util files (`metrics.py`, `utils.py`)
| - `config` : Folder with all config files. `config.yaml` is the one used by default, and others can be used to overwrite it when calling `main.py` with `+folder=file`, for example calling `python3 main.py +methods=dpo +games=SPBargaining`.
| - `data` : Folder with files to generate the data (`.py`) used in the different games, and data files themselves (`.pkl`).
| - `prompts` : Folder with text files with all the prompts used
| - `logs` : Empty folder where all the logs of the experiments are stored (not empty in my local).
