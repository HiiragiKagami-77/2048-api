import numpy as np


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class MyOwnAgent(Agent):
    def __init__(self, game, display = None):
        super().__init__(game, display)
        import torch
        import torch.nn as nn
        import torch.nn.functional as f
        from math import log
        from game2048.my2048 import Net
        PATH = './game2048/2048-0.pth'
        self.model = Net()
        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()
        
    def step(self):
        if self.game.score==128:
            import torch
            PATH = './game2048/2048-128.pth'
            self.model.load_state_dict(torch.load(PATH))
        if self.game.score==256:
            import torch
            PATH = './game2048/2048-256.pth'
            self.model.load_state_dict(torch.load(PATH))
        if self.game.score==512:
            import torch
            PATH = './game2048/2048-512.pth'
            self.model.load_state_dict(torch.load(PATH))
        direction = self.model.find_direction(self.game.board)
        return direction
