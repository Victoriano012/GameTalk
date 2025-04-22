from scipy.optimize import bisect
from functools import lru_cache
from collections import defaultdict
from types import SimpleNamespace
from utils import autoassign
from copy import copy
import re

# Returns the game instance based on the game name
def get_game(game_name: str):
    if game_name == "rock-paper-scissors":
        return RPS
    if game_name == "bertrand-competition":
        return BertrandCompetition
    if game_name == "size-prize-bargaining-game":
        return SizePrizeGame
    return None


# rock-paper-scissors
class RPS():
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"
    ERROR = "error"

    def __init__(self, id_1, id_2, **kwargs):
        self.player_1 = SimpleNamespace(id=id_1, move=None)
        self.player_2 = SimpleNamespace(id=id_2, move=None)

        self.ids = (id_1, id_2)

    def make_move(self, move, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
    
        curr_player, other_player = (self.player_1, self.player_2) if player_id == self.player_1.id else (self.player_2, self.player_1)
        
        if move is None:
            if other_player.move is not None:
                curr_player.move = RPS.ERROR
            return

        move = move.strip().lower()
        if move not in (RPS.ROCK, RPS.PAPER, RPS.SCISSORS):
            curr_player.move = RPS.ERROR
        else:
            curr_player.move = move
    
    # move1 wins -> 2., move2 wins -> 0., tie -> 1.
    def score(self, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        move1, move2 = (self.player_1.move, self.player_2.move) if player_id == self.player_1.id else (self.player_2.move, self.player_1.move)

        # error cases
        if move1 == RPS.ERROR or move1 == None: return -1.
        if move2 == RPS.ERROR or move2 == None: return 2.
        
        mapping = {RPS.ROCK: 0, RPS.PAPER: 1, RPS.SCISSORS: 2}
        score = (mapping[move1] - mapping[move2] + 3) % 3
        score = -1 if score == 2 else score
        return float(score+1)
    
    def game_metrics(games, player_id):
        rewards = [g.score(player_id) for g in games]
        
        metrics = defaultdict(float)
        metrics["win_rate"] = sum(1 for r in rewards if r == 2.)
        metrics["draw_rate"] = sum(1 for r in rewards if r == 1.)
        metrics["loss_rate"] = sum(1 for r in rewards if r <= 0.)

        metrics["lost_by_error (%)"] += sum(1 for r in rewards if r == -1.)
        metrics["won_by_error (%)"] += sum(1 for g in games if g._won_by_error(player_id))

        return metrics
    
    def is_finished(self):
        return self.player_1.move is not None and self.player_2.move is not None
        
        
    def _won_by_error(self, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        move1, move2 = (self.player_1.move, self.player_2.move) if player_id == self.player_1.id else (self.player_2.move, self.player_1.move)

        if move1 == RPS.ERROR or move1 == None: return False
        return move2 == RPS.ERROR or move2 == None


def can_cast_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

class BertrandCompetition():
    @autoassign
    def __init__(self, id_1, id_2, cost, demand_den, max_price_with_demand, **kwargs):
        self.player_1 = SimpleNamespace(id=id_1, moves=[])
        self.player_2 = SimpleNamespace(id=id_2, moves=[])

        self.ids = (id_1, id_2)
        self.__max_earnings = None

    def make_move(self, move, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        curr_player = self.player_1 if player_id == self.player_1.id else self.player_2

        if move is None:
            curr_player.moves.append("error")
        else:
            move = move.strip().lower()
            if move[0] != '$' or not can_cast_to_int(move[1:]):
                curr_player.moves.append("error")
            else: curr_player.moves.append(int(move))
        
    
    def score(self, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        moves1, moves2 = (self.player_1.moves, self.player_2.moves) if player_id == self.player_1.id else (self.player_2.moves, self.player_1.moves)
        moves1, moves2 = copy(moves1), copy(moves1)

        if len(moves1) == 0 and moves1[-1] == "error": moves1 = moves1[:-1]
        if len(moves2) == 0 and moves2[-1] == "error": moves2 = moves2[:-1]
        if len(moves1) > len(moves2): moves1 = moves1[:len(moves2)]
        if len(moves1) < len(moves2): moves2 = moves2[:len(moves1)]

        benefit = 0
        for p1, p2 in zip(moves1, moves2):
            benefit += (p1-self.cost) * self._demand_function(p1,p2)
        return benefit
    
    def game_metrics(games, player_id):
        normalized_earnings = [g.score(player_id)/g._max_earnings() for g in games]

        return {"normalized_earnings" : sum(normalized_earnings) / len(games)}
        
    def is_finished(self):
        p1_error = len(self.player_1.moves) > 0 and self.player_1.moves[-1] == 'error'
        p2_error = len(self.player_2.moves) > 0 and self.player_2.moves[-1] == 'error'
        return p1_error or p2_error

    def _len(self):
        if len(moves1) == 0 and moves1[-1] == "error": moves1 = moves1[:-1]
        if len(moves2) == 0 and moves2[-1] == "error": moves2 = moves2[:-1]
        return min(len(moves1), len(moves2))

    def _demand_function(self, price, rival_price):
        if price > rival_price: return 0
        elif price < rival_price: return (self.max_price_with_demand - price) // self.demand_den
        else: return (self.max_price_with_demand - price) // (2 * self.demand_den)
        
    def _max_earnings(self):
        if self.__max_earnings is None:
            self.__max_earnings = max((p-self.cost) * self._demand_function(p, self.max_price_with_demand+2) for p in range(self.cost, self.max_price_with_demand))
        return self.__max_earnings


@lru_cache(maxsize=None)
def harmonic(n):
    return sum(1/i for i in range(1, n+1))

class SizePrizeGame():
    ERROR = "error"
    ACCEPT = "accept"

    @autoassign
    def __init__(self, id_1, id_2, buyer_num, product_name, cost, initial_value, **kwargs):
        self.player_1 = SimpleNamespace(id=id_1, move=None)
        self.player_2 = SimpleNamespace(id=id_2, move=None)

        self.ids = (id_1, id_2)

    def make_move(self, move, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        curr_player, other_player = (self.player_1, self.player_2) if player_id == self.player_1.id else (self.player_2, self.player_1)

        move = move.strip().lower()
        if move == SizePrizeGame.ACCEPT:
            curr_player.move = SizePrizeGame.ACCEPT
        else:
            proposal = self._read_proposal(move)
            if proposal is None:
                curr_player.move = SizePrizeGame.ERROR
            else:
                curr_player.move = proposal
                other_player.move = None

    def score(self, player_id, other=False):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        move1, move2 = self.player_1.move, self.player_2.move

        if move1 == SizePrizeGame.ERROR or move2 == SizePrizeGame.ERROR: return 0.
        
        if move1 == SizePrizeGame.ACCEPT: agreement = move2
        elif move2 == SizePrizeGame.ACCEPT: agreement = move1
        else: return 0.

        is_buyer = player_id == (self.player_1.id if self.buyer_num%2 == 1 else self.player_2.id)
        is_buyer ^= other
        if is_buyer:
            return self.initial_value * harmonic(agreement[0]) - agreement[1]
        else:
            return agreement[1] - self.cost * agreement[0]
    
    def game_metrics(games, player_id):
        bargaining_power = []
        for g in games:
            u1 = g.score(player_id)
            u2 = g.score(player_id, other=True)
            f = lambda a : -a*(u1**(a-1)) + (1-a)*(u2**(-a))
            bargaining_power.append(bisect(f, 0, 1))

        return {"bargaining_power" : sum(bargaining_power) / len(games)}
        
    def is_finished(self):
        p1_done = self.player_1.move in (SizePrizeGame.ACCEPT, SizePrizeGame.ERROR)
        p2_done = self.player_2.move in (SizePrizeGame.ACCEPT, SizePrizeGame.ERROR)
        return p1_done or p2_done


    def _read_proposal(self, string):
        pattern = rf"^(\w+) {self.product} at price \$(\w+)$"
        match = re.match(pattern, string)
        try:
            x = int(match.group(1))
            y = float(match.group(2))
            assert x >= 0
            return x, y
        except (ValueError, AssertionError):
            return None
