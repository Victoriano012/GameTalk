from scipy.optimize import bisect
from functools import lru_cache
from collections import defaultdict
from types import SimpleNamespace
from utils import autoassign
from copy import copy
import math
import re

# Returns the game instance based on the game name
def get_game(game_name: str):
    if game_name == "rock-paper-scissors":
        return RPS
    if game_name == "bertrand-competition":
        return BertrandCompetition
    if game_name == "size-prize-bargaining-game":
        return SizePrizeGame
        
    if game_name == "rock-scissors":
        RPS.banned_moves_2 = [RPS.PAPER]  # TRAIN_LLM IS PLAYER 2
        return RPS
    return None


# rock-paper-scissors
class RPS():
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"
    ERROR = "error"

    banned_moves_1 = []
    banned_moves_2 = []
    
    @staticmethod
    def show_moves(): return False

    @staticmethod
    def get_possible_moves(player_num):
        if player_num%2 == 1: return [x for x in [RPS.ROCK, RPS.PAPER, RPS.SCISSORS] if x not in RPS.banned_moves_1]
        if player_num%2 == 0: return [x for x in [RPS.ROCK, RPS.PAPER, RPS.SCISSORS] if x not in RPS.banned_moves_2]

    def __init__(self, id_1, id_2, **kwargs):
        self.player_1 = SimpleNamespace(id=id_1, move=None, banned_moves=RPS.banned_moves_1)
        self.player_2 = SimpleNamespace(id=id_2, move=None, banned_moves=RPS.banned_moves_2)

        self.ids = (id_1, id_2)
        
    def is_error(self):
        return self.player_1.move == RPS.ERROR or self.player_2.move == RPS.ERROR
    
    def is_finished(self):
        return self.player_1.move == RPS.ERROR or self.player_2.move == RPS.ERROR or \
               self.player_1.move is not None and self.player_2.move is not None

    # make_move returns my_kwargs and other_kwargs, to add intermediate prompts in the conversation
    # kwargs=None -> No need for intermediate prompt
    def make_move(self, move, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
    
        curr_player, other_player = (self.player_1, self.player_2) if player_id == self.player_1.id else (self.player_2, self.player_1)
        
        if move is None:
            if other_player.move is not None:
                curr_player.move = RPS.ERROR
            return None, None

        move = move.strip().lower()
        if move not in (RPS.ROCK, RPS.PAPER, RPS.SCISSORS) or move in curr_player.banned_moves:
            curr_player.move = RPS.ERROR
            return None, None
        else:
            curr_player.move = move
            return None, {}
    
    # player_id wins -> 2., other wins -> 0., tie -> 1.
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
        
        
    def _won_by_error(self, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        move1, move2 = (self.player_1.move, self.player_2.move) if player_id == self.player_1.id else (self.player_2.move, self.player_1.move)

        if move1 == RPS.ERROR or move1 == None: return False
        return move2 == RPS.ERROR or move2 == None



def price_to_int(s):
    s = s.strip().lower()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    if s[0] != '$':
        return None
    try:
        return int(s[1:])
    except ValueError:
        return None

class BertrandCompetition():
    @autoassign
    def __init__(self, id_1, id_2, cost, demand_den, max_price_with_demand, **kwargs):
        self.player_1 = SimpleNamespace(id=id_1, moves=[])
        self.player_2 = SimpleNamespace(id=id_2, moves=[])

        self.ids = (id_1, id_2)
        self.cost = float(cost)
        self.demand_den = float(demand_den)
        self.max_price_with_demand = float(max_price_with_demand)
        self.__max_earnings = None

    def make_move(self, move, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        curr_player = self.player_1 if player_id == self.player_1.id else self.player_2

        if move is None or move == "error":
            curr_player.moves.append("error")
            return None, None
        else:
            price = price_to_int(move)
            curr_player.moves.append(price if price is not None else "error")
            if price is None or player_id == self.player_1.id:
                return None, None
            else:
                price_1 = self.player_1.moves[-1]
                price_2 = self.player_2.moves[-1]
                my_kwargs = {'my_price' : price_1, 'other_price' : price_2, 'my_benefit' : self._benefit(price_1, price_2)}
                other_kwargs = {'my_price' : price_2, 'other_price' : price_1, 'my_benefit' : self._benefit(price_2, price_1)}
                return my_kwargs, other_kwargs
    
    def score(self, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        moves1, moves2 = (self.player_1.moves, self.player_2.moves) if player_id == self.player_1.id else (self.player_2.moves, self.player_1.moves)
        moves1, moves2 = copy(moves1), copy(moves1)

        if len(moves1) != 0 and moves1[-1] == "error": moves1 = moves1[:-1]
        if len(moves2) != 0 and moves2[-1] == "error": moves2 = moves2[:-1]
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
        
    @staticmethod
    def show_moves():
        return False
    
    def is_error(self):
        moves1, moves2 = self.player_1.moves, self.player_2.moves
        return (len(moves1) != 0 and moves1[-1] == "error") or (len(moves2) != 0 and moves2[-1] == "error")


    def _len(self):
        if len(moves1) == 0 and moves1[-1] == "error": moves1 = moves1[:-1]
        if len(moves2) == 0 and moves2[-1] == "error": moves2 = moves2[:-1]
        return min(len(moves1), len(moves2))

    def _benefit(self, price_1, price_2):
        return (price_1 - self.cost) * self._demand_function(price_1, price_2)
    
    def _demand_function(self, price, rival_price):
        if price > rival_price: return 0
        elif price < rival_price: return max(0, (self.max_price_with_demand - price) // self.demand_den )
        else: return max(0, (self.max_price_with_demand - price) // (2 * self.demand_den) )
        
    def _max_earnings(self):
        if self.__max_earnings is None:
            self.__max_earnings = max((p-self.cost) * self._demand_function(p, self.max_price_with_demand+2) for p in range(math.floor(self.cost), math.ceil(self.max_price_with_demand)))
            # actually the best price is ~ (cost + max_price_with_demand)/2
        return self.__max_earnings


@lru_cache(maxsize=None)
def harmonic(n):
    return sum(1/i for i in range(1, n+1))

class SizePrizeGame():
    ERROR = "error"
    ACCEPT = "accept"

    @autoassign
    def __init__(self, id_1, id_2, player_A_num, products, cost, value, **kwargs):
        self.player_1 = SimpleNamespace(id=id_1, move=None)
        self.player_2 = SimpleNamespace(id=id_2, move=None)

        self.ids = (id_1, id_2)
        self.buyer_num = player_A_num

    def make_move(self, move, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        curr_player, other_player = (self.player_1, self.player_2) if player_id == self.player_1.id else (self.player_2, self.player_1)
        
        if move is None:
            curr_player.move = SizePrizeGame.ERROR
            return None, None

        move = move.strip().lower()
        if move[:len(SizePrizeGame.ACCEPT)] == SizePrizeGame.ACCEPT:
            curr_player.move = SizePrizeGame.ACCEPT
        else:
            proposal = self._read_proposal(move)
            if proposal is None:
                curr_player.move = SizePrizeGame.ERROR
            else:
                curr_player.move = proposal
                other_player.move = None
        return None, None

    def score(self, player_id, other=False):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        move1, move2 = self.player_1.move, self.player_2.move

        if move1 == SizePrizeGame.ERROR or move2 == SizePrizeGame.ERROR: return 0.
        if move1 is None or move2 is None: return 0.
        
        if move1 == SizePrizeGame.ACCEPT: agreement = move2
        elif move2 == SizePrizeGame.ACCEPT: agreement = move1
        else: return 0. # This cannot happen, I think

        is_buyer = player_id == (self.player_1.id if self.buyer_num%2 == 1 else self.player_2.id)
        is_buyer ^= other
        if is_buyer:
            return self.value * harmonic(agreement['units']) - agreement['price']*agreement['units']
        else:
            return agreement['units'] * (agreement['price'] - self.cost)
    
    def game_metrics(games, player_id):
        metrics = {}

        bargaining_power = []
        for g in games:
            u1 = g.score(player_id)
            u2 = g.score(player_id, other=True)
            bp = 0.5 if u1 <= 0 and u2 <= 0 else 0. if u1 <= 0 else 1. if u2 <= 0 else "normal situation"
            if bp == "normal situation":
                f = lambda a : -a*(u1**(a-1)) + (1-a)*(u2**(-a))
                bp = bisect(f, 0, 1)
            bargaining_power.append(bp)
        metrics["bargaining_power"] = sum(bargaining_power) / len(games)

        metrics["no-deal (%)"] = sum(1 for g in games if SizePrizeGame.ACCEPT in (g.player_1.move, g.player_2.move)) / len(games)
        return metrics
        
    def is_finished(self):
        p1_done = self.player_1.move in (SizePrizeGame.ACCEPT, SizePrizeGame.ERROR)
        p2_done = self.player_2.move in (SizePrizeGame.ACCEPT, SizePrizeGame.ERROR)
        return p1_done or p2_done

    @staticmethod
    def show_moves():
        return True

    def is_error(self):
        return self.player_1.move == SizePrizeGame.ERROR or self.player_2.move == SizePrizeGame.ERROR

    @staticmethod
    def _read_proposal(string):
        patterns = [
            r"(\d+) units at \$(\d+(\.\d+)?) each",
            r"(\d+) unit at \$(\d+(\.\d+)?) each",
            r"(\d+) units at (\d+(\.\d+)?) each",
            r"(\d+) unit at (\d+(\.\d+)?) each",
            r"(\d+) units at \$(\d+(\.\d+)?)",
            r"(\d+) unit at \$(\d+(\.\d+)?)",
            r"(\d+) units at (\d+(\.\d+)?)",
            r"(\d+) unit at (\d+(\.\d+)?)"
        ]
        for pattern in patterns:
            try:
                match = re.match(pattern, string)
                x = int(match.group(1))
                y = float(match.group(2))
                assert x >= 0
                return {'units' : x, 'price' : y}
            except (ValueError, AssertionError, AttributeError):
                # ValueError -> cannot cast, AssertionError -> x < 0, AttributeError -> no match
                pass
        return None
