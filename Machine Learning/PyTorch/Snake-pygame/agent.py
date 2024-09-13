import torch
import random
import numpy as np
from collections import deque # a data structure that is used to store our memeory
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000