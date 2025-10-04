import numpy as np
import random

class SnakeVecEnv:
    def __init__(self, n_envs=8, w=10, h=10, num_apples=3):
        self.n_envs = n_envs
        self.w = w
        self.h = h
        self.num_apples = num_apples
        self.max_steps = w * h * 4

        # env state
        self.snakes = []
        self.directions = []
        self.foods = []
        self.steps = []
        self.steps_since_food = []
        self.history = [[] for _ in range(n_envs)]

    def reset(self):
        self.snakes = [[(self.w // 2, self.h // 2)] for _ in range(self.n_envs)]
        self.directions = [random.choice([(1,0), (-1,0), (0,1), (0,-1)]) for _ in range(self.n_envs)]
        self.foods = [[self._new_food(s)] for s in self.snakes]
        for i in range(self.n_envs):
            while len(self.foods[i]) < self.num_apples:
                self.foods[i].append(self._new_food(self.snakes[i]))
        self.steps = [0 for _ in range(self.n_envs)]
        self.steps_since_food = [0 for _ in range(self.n_envs)]
        self.history = [[] for _ in range(self.n_envs)]
        return np.array([self._get_state(i) for i in range(self.n_envs)])

    def _new_food(self, snake):
        while True:
            f = (random.randint(0, self.w-1), random.randint(0, self.h-1))
            if f not in snake:
                return f

    def _get_state(self, idx):
        grid = np.zeros((self.h, self.w), dtype=np.float32)
        for (x,y) in self.snakes[idx]:
            grid[y][x] = -1.0
        for (fx,fy) in self.foods[idx]:
            grid[fy][fx] = 1.0
        return grid.flatten()

    def step(self, actions):
        rewards, dones, lengths = [], [], []
        new_states = []

        for i in range(self.n_envs):
            if actions[i] == 0: d = (0, -1)  # up
            elif actions[i] == 1: d = (0, 1) # down
            elif actions[i] == 2: d = (-1, 0) # left
            else: d = (1, 0)      # right
            self.directions[i] = d

            head_x, head_y = self.snakes[i][0]
            new_head = (head_x + d[0], head_y + d[1])

            reward, done = -0.01, False  # small step cost

            # collision
            if (new_head[0] < 0 or new_head[0] >= self.w or
                new_head[1] < 0 or new_head[1] >= self.h or
                new_head in self.snakes[i]):
                done, reward = True, -1
            else:
                self.snakes[i].insert(0, new_head)
                if new_head in self.foods[i]:
                    reward = 1
                    self.foods[i].remove(new_head)
                    self.foods[i].append(self._new_food(self.snakes[i]))
                    self.steps_since_food[i] = 0
                else:
                    self.snakes[i].pop()
                    self.steps_since_food[i] += 1

            self.steps[i] += 1
            self.history[i].append(new_head)

            # --- Anti-stalling ---
            if self.steps[i] > self.max_steps:
                done, reward = True, -0.5
            if self.steps_since_food[i] > self.w * self.h:
                done, reward = True, -0.5
            if len(self.history[i]) > 20:
                recent = self.history[i][-20:]
                if len(set(recent)) < 5:
                    done, reward = True, -0.5

            rewards.append(reward)
            dones.append(done)
            lengths.append(len(self.snakes[i]))
            new_states.append(self._get_state(i))

            if done:
                # reset snake + apples
                self.snakes[i] = [(self.w // 2, self.h // 2)]
                self.directions[i] = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
                self.foods[i] = [self._new_food(self.snakes[i])]
                while len(self.foods[i]) < self.num_apples:
                    self.foods[i].append(self._new_food(self.snakes[i]))
                self.steps[i] = 0
                self.steps_since_food[i] = 0
                self.history[i] = []

        return np.array(new_states), np.array(rewards), np.array(dones), np.array(lengths)
