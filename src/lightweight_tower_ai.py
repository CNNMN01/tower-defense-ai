
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from collections import deque
import pickle
import os
import time

class LightweightTowerAI:
    def __init__(self, save_file="lightweight_tower_ai.pkl"):
        self.save_file = save_file
        
        # Much smaller network - memory friendly
        self.input_size = 20  # Reduced from 36
        self.hidden_sizes = [64, 32]  # Only 2 layers, smaller
        self.output_size = 12  # Reduced actions
        
        if os.path.exists(save_file):
            self.load_brain()
            print(f"[LIGHTWEIGHT] Loaded AI with {self.experience_count} experiences")
        else:
            self.initialize_brain()
            print("[LIGHTWEIGHT] Created memory-efficient AI")
        
        self.learning_rate = 0.01  # Higher for faster learning
    
    def initialize_brain(self):
        # Simple 2-layer network
        self.W1 = np.random.randn(self.input_size, self.hidden_sizes[0]) * 0.1
        self.b1 = np.zeros((1, self.hidden_sizes[0]))
        self.W2 = np.random.randn(self.hidden_sizes[0], self.hidden_sizes[1]) * 0.1
        self.b2 = np.zeros((1, self.hidden_sizes[1]))
        self.W3 = np.random.randn(self.hidden_sizes[1], self.output_size) * 0.1
        self.b3 = np.zeros((1, self.output_size))
        
        self.experience_count = 0
        self.victories = 0
        self.battle_history = []
    
    def forward(self, x):
        # Simple forward pass
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)  # ReLU
        
        z3 = np.dot(a2, self.W3) + self.b3
        return z3, a2, a1  # Return intermediates for backprop
    
    def train_step(self, state, target):
        # Single sample training to save memory
        output, a2, a1 = self.forward(state.reshape(1, -1))
        
        # Simple backprop
        error = output - target.reshape(1, -1)
        
        # Output layer
        dW3 = np.dot(a2.T, error) * self.learning_rate
        db3 = error * self.learning_rate
        
        # Hidden layer 2
        d_a2 = np.dot(error, self.W3.T)
        d_z2 = d_a2 * (a2 > 0)  # ReLU derivative
        dW2 = np.dot(a1.T, d_z2) * self.learning_rate
        db2 = d_z2 * self.learning_rate
        
        # Hidden layer 1
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * (a1 > 0)  # ReLU derivative
        dW1 = np.dot(state.reshape(1, -1).T, d_z1) * self.learning_rate
        db1 = d_z1 * self.learning_rate
        
        # Update weights
        self.W3 -= dW3
        self.b3 -= db3
        self.W2 -= dW2
        self.b2 -= db2
        self.W1 -= dW1
        self.b1 -= db1
        
        self.experience_count += 1
    
    def save_brain(self):
        brain_data = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'experience_count': self.experience_count,
            'victories': self.victories,
            'battle_history': self.battle_history[-50:]  # Keep only recent
        }
        
        try:
            with open(self.save_file, 'wb') as f:
                pickle.dump(brain_data, f)
            print(f"[SAVE] Lightweight AI saved")
        except Exception as e:
            print(f"[ERROR] Save failed: {e}")
    
    def load_brain(self):
        try:
            with open(self.save_file, 'rb') as f:
                data = pickle.load(f)
            
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.W3 = data['W3']
            self.b3 = data['b3']
            self.experience_count = data.get('experience_count', 0)
            self.victories = data.get('victories', 0)
            self.battle_history = data.get('battle_history', [])
        except Exception as e:
            print(f"[ERROR] Load failed: {e}")
            self.initialize_brain()

class MemoryEfficientGame:
    def __init__(self, scenario_seed=None):
        if scenario_seed is None:
            scenario_seed = int(time.time() * 1000) % 10000
        
        random.seed(scenario_seed)
        self.scenario_id = scenario_seed
        
        self.width = 6  # Smaller grid
        self.height = 6
        self.grid = np.zeros((self.height, self.width))
        
        # Simple straight path
        self.path = [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]
        
        self.towers = []
        self.enemies = []
        self.money = 200
        self.lives = 15
        self.wave = 1
        self.time = 0
        
        # Mark path
        for x, y in self.path:
            self.grid[y][x] = -1
    
    def get_simple_state(self):
        # Lightweight state representation
        state = []
        
        # Grid state (compressed)
        for y in range(self.height):
            for x in range(self.width):
                state.append(self.grid[y][x])
        
        # Take only first 16 cells
        state = state[:16]
        
        # Game stats
        state.extend([
            self.money / 300.0,
            self.lives / 15.0,
            len(self.enemies) / 5.0,
            len(self.towers) / 10.0
        ])
        
        # Pad to 20 features
        while len(state) < 20:
            state.append(0)
        
        return np.array(state[:20], dtype=np.float32)
    
    def spawn_enemies(self):
        for i in range(5):  # Fewer enemies
            enemy = Enemy(0, 3, health=80 + i*10)
            self.enemies.append(enemy)
    
    def can_build_tower(self, x, y):
        return (0 <= x < self.width and 0 <= y < self.height and 
                self.grid[y][x] == 0 and self.money >= 80)
    
    def build_tower(self, x, y):
        if self.can_build_tower(x, y):
            tower = Tower(x, y)
            self.towers.append(tower)
            self.grid[y][x] = len(self.towers)
            self.money -= 80
            return True
        return False
    
    def update_game(self):
        self.time += 0.1
        
        # Spawn enemies
        if not self.enemies and self.wave <= 5:
            self.spawn_enemies()
            self.wave += 1
        
        # Move enemies
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy.alive:
                enemy.x += 0.3
                if enemy.x >= self.width:
                    self.lives -= 1
                    enemies_to_remove.append(enemy)
            else:
                self.money += 20
                enemies_to_remove.append(enemy)
        
        for enemy in enemies_to_remove:
            self.enemies.remove(enemy)
        
        # Towers shoot
        for tower in self.towers:
            tower.shoot(self.enemies)
        
        # Check end conditions
        if self.lives <= 0:
            return "lose"
        elif self.wave > 5 and not self.enemies:
            return "win"
        return "continue"
    
    def visualize(self, title_suffix=""):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Simple visualization
        for i in range(self.height + 1):
            ax.axhline(y=i, color='gray', alpha=0.3)
        for j in range(self.width + 1):
            ax.axvline(x=j, color='gray', alpha=0.3)
        
        # Path
        path_x = [p[0] + 0.5 for p in self.path]
        path_y = [p[1] + 0.5 for p in self.path]
        ax.plot(path_x, path_y, 'brown', linewidth=3)
        
        # Towers
        for tower in self.towers:
            rect = Rectangle((tower.x, tower.y), 1, 1, facecolor='blue', alpha=0.7)
            ax.add_patch(rect)
        
        # Enemies
        for enemy in self.enemies:
            if enemy.alive:
                circle = Circle((enemy.x, enemy.y), 0.2, facecolor='red')
                ax.add_patch(circle)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title(f'Lightweight TD - Wave {self.wave} | Lives: {self.lives} | ${self.money} {title_suffix}')
        plt.show()

# Use same Enemy and Tower classes but simplified
class Enemy:
    def __init__(self, x, y, health=100):
        self.x = x
        self.y = y
        self.health = health
        self.alive = True
    
    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            self.alive = False
            return True
        return False

class Tower:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.damage = 40
        self.range = 2.0
        self.kills = 0
    
    def shoot(self, enemies):
        for enemy in enemies:
            if enemy.alive:
                dist = ((enemy.x - self.x)**2 + (enemy.y - self.y)**2)**0.5
                if dist <= self.range:
                    killed = enemy.take_damage(self.damage)
                    if killed:
                        self.kills += 1
                    return True
        return False

class LightweightAI:
    def __init__(self):
        self.brain = LightweightTowerAI()
        self.memory = deque(maxlen=1000)  # Much smaller memory
    
    def choose_action(self, game, epsilon=0.1):
        valid_positions = [(x, y) for x in range(game.width) 
                          for y in range(game.height) 
                          if game.can_build_tower(x, y)]
        
        if not valid_positions or random.random() < epsilon:
            if valid_positions:
                return random.choice(valid_positions)
            return None
        
        # AI decision
        state = game.get_simple_state()
        q_values, _, _ = self.brain.forward(state)
        
        # Map to valid positions
        best_pos = None
        best_value = float('-inf')
        
        for i, pos in enumerate(valid_positions[:12]):  # Limit choices
            if q_values[0][i % 12] > best_value:
                best_value = q_values[0][i % 12]
                best_pos = pos
        
        return best_pos
    
    def train_episode(self, game, epsilon=0.3):
        total_reward = 0
        
        for step in range(20):  # Fewer steps
            state = game.get_simple_state()
            action = self.choose_action(game, epsilon)
            
            if action:
                success = game.build_tower(action[0], action[1])
                reward = 10 if success else -5
            else:
                reward = 0
            
            # Simulate game
            for _ in range(10):
                result = game.update_game()
                if result != "continue":
                    break
            
            total_reward += reward
            
            # Simple training
            target = np.zeros(12)
            if action:
                action_index = hash(str(action)) % 12
                target[action_index] = reward
                self.brain.train_step(state, target)
            
            if result != "continue":
                if result == "win":
                    self.brain.victories += 1
                    total_reward += 100
                break
            
            if game.money < 80:
                break
        
        return result == "win", total_reward
    
    def save_progress(self):
        self.brain.save_brain()

def run_lightweight_tower_defense():
    print("=== LIGHTWEIGHT TOWER DEFENSE AI ===")
    print("=== Memory-optimized for iOS ===")
    print("=" * 45)
    
    ai = LightweightAI()
    
    print(f"[STATS] AI Knowledge: {ai.brain.experience_count} experiences, {ai.brain.victories} victories")
    
    # Create game
    game = MemoryEfficientGame()
    print(f"[GAME] Scenario #{game.scenario_id} - {game.width}x{game.height} grid")
    
    game.visualize("(Starting Battle)")
    
    # Quick training
    print("[TRAIN] Quick training session...")
    wins = 0
    for episode in range(20):  # Fewer episodes
        training_game = MemoryEfficientGame(game.scenario_id)
        victory, reward = ai.train_episode(training_game, epsilon=max(0.1, 0.8 * 0.9**episode))
        if victory:
            wins += 1
    
    print(f"[TRAIN] Training complete: {wins}/20 wins ({wins*5}%)")
    
    # Test
    test_game = MemoryEfficientGame(game.scenario_id)
    print("[TEST] Testing AI...")
    
    for step in range(15):
        action = ai.choose_action(test_game, epsilon=0)
        if action:
            test_game.build_tower(action[0], action[1])
        
        for _ in range(10):
            result = test_game.update_game()
            if result != "continue":
                break
        
        if result != "continue" or test_game.money < 80:
            break
    
    if result == "win":
        print(f"[WIN] Victory! {len(test_game.towers)} towers, {sum(t.kills for t in test_game.towers)} kills")
    else:
        print(f"[LOSE] Defeated. {len(test_game.towers)} towers built")
    
    test_game.visualize("(Final Result)")
    
    ai.save_progress()
    print(f"[SAVE] Progress saved. Total experiences: {ai.brain.experience_count}")

# Clear memory and run lightweight version
import gc
gc.collect()  # Force garbage collection

if __name__ == "__main__":
    run_lightweight_tower_defense()
