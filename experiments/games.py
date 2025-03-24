"""
This file contains the implementation of games
Each instance of a game is just one move, initialized as Game(llm_move)
    where llm_move is a lowercase string with no space at beginning or end
    if errors happened, llm_move = "error"
Each game must have defined a score method that takes two instances and returns the score (float/double) of the first player
"""

# Returns the game instance based on the game name
def get_game(game_name: str):
    if game_name == "rock-paper-scissors":
        return RPS
    return None


from enum import Enum

# rock-paper-scissors implementation
class RPS(Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"
    ERROR = "error"

    def score(move1, move2) -> int:
        """
        move1 wins -> 1, move2 wins -> -1, tie -> 0
        """
        mapping = {"rock": 0, "paper": 1, "scissors": 2}
        ans = (mapping[move1.value] - mapping[move2.value] + 3) % 3
        ans = -1 if ans == 2 else ans
        return float(ans)