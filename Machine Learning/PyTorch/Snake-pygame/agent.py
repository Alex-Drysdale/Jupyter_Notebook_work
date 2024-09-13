import torch
import random
import numpy as np
from collections import deque # a data structure that is used to store our memeory
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # parameter to control the randomness 
        self.gamma = 0 # discount rate 
        self.memory = deque(maxlen=MAX_MEMORY) # if we excedd this memory
        self.model = None 
        self.trainer = None
        # TODO: model, trainer
    
    def get_state(self, game):
        # this is the states array so that the agent has more information
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y) # created a named tuple and then - 20 has been hardcoded for the size of the snake
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20) # creates points around the head
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT # game direction so only one of these is 1 at any time
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or # so here if we are going right and the point right gives a collision, the value is 1 for danger ahead
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Moving direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down (I think though it should be the other way round)
        ]

        return np.array(state, dtype=int) # convert the list to a numpy array and turn the booleans to integers
    
    def remember(self, state, action, reward, next_state, done):
        # we want to remember all of this in our memory
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached (Find out what popleft is)

    def train_long_memory(self):
        # here we are taking the variables from our memory (Specifically the no. of variables in a batch)
        if len(self.memory) > BATCH_SIZE: 
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples 
        else: # if we don't have enough elements in memory for a single batch
            mini_sample = self.memory # then we jsut take the whole memory.

        states, actions, rewards, next_states, dones = zip(*mini_sample) # compiles into a list of tuples. not sure why there is an "*"
        self.trainer.train_step(states, actions, rewards, next_states, dones) 

    def train_short_memory(self, state, action, reward, next_state, done): 
        self.trainer.train_step(state, action, reward, next_state, done) # all we need to train it for one game step but can take an array or tensor for batch size
                                                                         

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation  (Random moves for exploration at the beginning)
        self.epsilon = 80 - self.n_games # can play around with this. More games we have the smaller our epsilon and therefore number of random moves
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon: # random move generator
            move = random.randint(0, 2)
            final_move[move] = 1
        else: # otherwise we are getting a prediction from our model.
            pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform the move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train the long memory (replay memory), (experience replay) 
            # so it trains again on all the previous moves and all the games it played (so super helpful)
            # Also want to plot the result.
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score
                # TODO agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)
                # TODO: Plot


if __name__ == '__main__':
    train()

    
