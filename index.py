import numpy as np
import random
import pygame
import sys

class TlouIa:
    def __init__(self, size=10, num_holes=8, num_supplies=5, num_obstacles=3):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start_state = (0, 0)
        self.goal_state = (size-1, size-1)
        self.hole_states = self._place_random_items(num_holes)
        self.supply_states = self._place_random_items(num_supplies, exclude=self.hole_states)
        self.obstacle_states = self._place_random_items(num_obstacles, exclude=self.hole_states + self.supply_states)
        self.supplies_collected = set()
        
        for i, j in self.hole_states:
            self.grid[i][j] = 1
        for i, j in self.supply_states:
            self.grid[i][j] = 2
        for i, j in self.obstacle_states:
            self.grid[i][j] = 3
    
    def _place_random_items(self, num_items, exclude=[]):
        items = []
        while len(items) < num_items:
            i, j = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (i, j) not in items and (i, j) not in exclude and (i, j) != self.start_state and (i, j) != self.goal_state:
                items.append((i, j))
        return items
    
    def reset(self):
        self.current_state = self.start_state
        self.supplies_collected = set()
        return self.current_state, tuple(self.supplies_collected)
    
    def step(self, action):
        i, j = self.current_state
        if action == 0:
            i = max(i-1, 0)
        elif action == 1:
            i = min(i+1, self.size-1)
        elif action == 2:
            j = max(j-1, 0)
        elif action == 3:
            j = min(j+1, self.size-1)
        
        if (i, j) in self.obstacle_states:
            i, j = self.current_state
        
        self.current_state = (i, j)
        
        if self.current_state == self.goal_state:
            if len(self.supplies_collected) == len(self.supply_states):
                reward = 10
                done = True
            else:
                reward = -1
                done = False
        elif self.current_state in self.hole_states:
            reward = -5 
            done = True
        elif self.current_state in self.supply_states and self.current_state not in self.supplies_collected:
            self.supplies_collected.add(self.current_state)
            reward = 2 
            done = False
        else:
            reward = -0.1
            done = False
        
        return self.current_state, tuple(self.supplies_collected), reward, done

    def render(self, screen, images, cell_size=60):
        screen.fill((0, 0, 0))

        for i in range(self.size):
            for j in range(self.size):
                x = j * cell_size
                y = i * cell_size
                screen.blit(images["background"], (x, y))
                
                if (i, j) == self.current_state:
                    screen.blit(images["agent"], (x, y))
                elif (i, j) == self.goal_state:
                    screen.blit(images["goal"], (x, y))
                elif self.grid[i][j] == 1:
                    screen.blit(images["zombie"], (x, y))
                elif self.grid[i][j] == 2:
                    if (i, j) not in self.supplies_collected:
                        screen.blit(images["supply"], (x, y))
                elif self.grid[i][j] == 3:
                    screen.blit(images["obstacle"], (x, y))

                pygame.draw.rect(screen, (0, 0, 0), (x, y, cell_size, cell_size), 1)

        pygame.display.flip()

pygame.init()
cell_size = 60
screen = pygame.display.set_mode((10 * cell_size, 10 * cell_size))
pygame.display.set_caption('The Last Of Us - IA')

images = {
    "agent": pygame.transform.scale(pygame.image.load("agent.png"), (cell_size, cell_size)),
    "zombie": pygame.transform.scale(pygame.image.load("zombie.png"), (cell_size, cell_size)),
    "goal": pygame.transform.scale(pygame.image.load("door.png"), (cell_size, cell_size)),
    "supply": pygame.transform.scale(pygame.image.load("supply.png"), (cell_size, cell_size)),
    "obstacle": pygame.transform.scale(pygame.image.load("stone.png"), (cell_size, cell_size)),
    "background": pygame.transform.scale(pygame.image.load("grass.png"), (cell_size, cell_size))
}

env = TlouIa(size=10, num_holes=8, num_supplies=5, num_obstacles=3)

q_table = np.zeros((env.size, env.size, 2 ** len(env.supply_states), 4))  

num_episodes = 10000
max_steps_per_episode = 100
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay_rate = 0.001

def epsilon_greedy_policy(state, collected_supplies):
    supply_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in env.supply_states]), 2)
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(q_table[state[0]][state[1]][supply_index])

for episode in range(num_episodes):
    state, collected_supplies = env.reset()
    done = False
    t = 0
    while not done and t < max_steps_per_episode:
        action = epsilon_greedy_policy(state, collected_supplies)
        next_state, next_collected_supplies, reward, done = env.step(action)
        supply_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in env.supply_states]), 2)
        next_supply_index = int(''.join(['1' if (i, j) in next_collected_supplies else '0' for (i, j) in env.supply_states]), 2)
        q_table[state[0]][state[1]][supply_index][action] += learning_rate * \
            (reward + discount_factor * np.max(q_table[next_state[0]][next_state[1]][next_supply_index]) - q_table[state[0]][state[1]][supply_index][action])
        state, collected_supplies = next_state, next_collected_supplies
        t += 1

    epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay_rate))

    if episode % 1000 == 0:
        print(f'Episode: {episode}')
        env.render(screen, images, cell_size)
        pygame.time.wait(500)

state, collected_supplies = env.reset()
done = False
while not done:
    supply_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in env.supply_states]), 2)
    action = np.argmax(q_table[state[0]][state[1]][supply_index])
    next_state, next_collected_supplies, reward, done = env.step(action)
    env.render(screen, images, cell_size)
    pygame.time.wait(500)
    state, collected_supplies = next_state, next_collected_supplies

pygame.quit()
sys.exit()
