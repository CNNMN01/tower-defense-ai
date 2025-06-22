import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from collections import deque
import pickle
import os
import time

class DeepPersistentTowerAI:
    def __init__(self, input_size=36, save_file="deep_tower_ai_brain.pkl"):
        self.save_file = save_file
        self.input_size = input_size
        
        # Deep network architecture - 6 hidden layers
        self.layer_sizes = [input_size, 256, 512, 256, 128, 64, 32]
        self.output_size = 24  # More action possibilities
        
        if os.path.exists(save_file):
            self.load_brain()
            print(f"[DEEP-BRAIN] Loaded sophisticated AI with {self.experience_count} battle experiences!")
        else:
            self.initialize_fresh_brain()
            print("[DEEP-NEW] Created advanced deep learning AI - ready for sophisticated strategy!")
        
        self.learning_rate = 0.001  # Lower learning rate for stability
        self.dropout_rate = 0.2
        self.momentum_factor = 0.9
    
    def initialize_fresh_brain(self):
        # Initialize deep network with Xavier/He initialization
        self.weights = []
        self.biases = []
        
        # He initialization for better gradient flow with ReLU
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            
            # He initialization
            limit = np.sqrt(2.0 / fan_in)
            weight = np.random.normal(0, limit, (fan_in, fan_out))
            bias = np.zeros((1, fan_out))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        # Output layer with smaller initialization
        fan_in = self.layer_sizes[-1]
        limit = np.sqrt(1.0 / fan_in)
        output_weight = np.random.normal(0, limit, (fan_in, self.output_size))
        output_bias = np.zeros((1, self.output_size))
        
        self.weights.append(output_weight)
        self.biases.append(output_bias)
        
        # Advanced learning components
        self.momentum_weights = [np.zeros_like(w) for w in self.weights]
        self.momentum_biases = [np.zeros_like(b) for b in self.biases]
        
        # Adaptive learning rates (RMSprop-style)
        self.grad_sq_weights = [np.ones_like(w) for w in self.weights]
        self.grad_sq_biases = [np.ones_like(b) for b in self.biases]
        
        # Performance tracking
        self.experience_count = 0
        self.total_battles_won = 0
        self.battle_history = []
        self.strategy_evolution = []
        self.training_losses = []
    
    def save_brain(self):
        brain_data = {
            'weights': self.weights,
            'biases': self.biases,
            'momentum_weights': self.momentum_weights,
            'momentum_biases': self.momentum_biases,
            'grad_sq_weights': self.grad_sq_weights,
            'grad_sq_biases': self.grad_sq_biases,
            'experience_count': self.experience_count,
            'total_battles_won': self.total_battles_won,
            'battle_history': self.battle_history[-200:],  # Keep recent history
            'strategy_evolution': self.strategy_evolution,
            'training_losses': self.training_losses[-1000:],  # Keep recent losses
            'layer_sizes': self.layer_sizes,
            'output_size': self.output_size
        }
        
        try:
            with open(self.save_file, 'wb') as f:
                pickle.dump(brain_data, f)
            print(f"[DEEP-SAVE] Saved sophisticated AI with {self.experience_count} experiences")
        except Exception as e:
            print(f"[DEEP-ERROR] Could not save AI: {e}")
    
    def load_brain(self):
        try:
            with open(self.save_file, 'rb') as f:
                brain_data = pickle.load(f)
            
            self.weights = brain_data['weights']
            self.biases = brain_data['biases']
            self.momentum_weights = brain_data.get('momentum_weights', [np.zeros_like(w) for w in self.weights])
            self.momentum_biases = brain_data.get('momentum_biases', [np.zeros_like(b) for b in self.biases])
            self.grad_sq_weights = brain_data.get('grad_sq_weights', [np.ones_like(w) for w in self.weights])
            self.grad_sq_biases = brain_data.get('grad_sq_biases', [np.ones_like(b) for b in self.biases])
            
            self.experience_count = brain_data.get('experience_count', 0)
            self.total_battles_won = brain_data.get('total_battles_won', 0)
            self.battle_history = brain_data.get('battle_history', [])
            self.strategy_evolution = brain_data.get('strategy_evolution', [])
            self.training_losses = brain_data.get('training_losses', [])
            self.layer_sizes = brain_data.get('layer_sizes', self.layer_sizes)
            self.output_size = brain_data.get('output_size', self.output_size)
        except Exception as e:
            print(f"[DEEP-ERROR] Could not load AI: {e}")
            self.initialize_fresh_brain()
    
    def forward(self, x, training=False):
        """Advanced forward pass with dropout and leaky ReLU"""
        current_input = x
        self.activations = [current_input]
        self.dropout_masks = []
        
        # Forward through all hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            
            # Leaky ReLU activation
            activation = np.maximum(0.01 * z, z)
            
            # Dropout during training
            if training and self.dropout_rate > 0:
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, activation.shape)
                activation = activation * dropout_mask / (1 - self.dropout_rate)
                self.dropout_masks.append(dropout_mask)
            else:
                self.dropout_masks.append(np.ones_like(activation))
            
            self.activations.append(activation)
            current_input = activation
        
        # Output layer (no activation for Q-values)
        output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.activations.append(output)
        
        return output
    
    def train_step(self, states, targets):
        """Advanced training with momentum and adaptive learning rates"""
        batch_size = states.shape[0]
        
        # Forward pass
        outputs = self.forward(states, training=True)
        
        # Compute loss (Huber loss for stability)
        diff = outputs - targets
        huber_delta = 1.0
        loss = np.where(np.abs(diff) < huber_delta,
                       0.5 * diff * diff,
                       huber_delta * (np.abs(diff) - 0.5 * huber_delta))
        total_loss = np.mean(loss)
        
        # Backward pass with advanced optimization
        delta = np.where(np.abs(diff) < huber_delta, diff, huber_delta * np.sign(diff))
        delta = delta / batch_size
        delta = np.clip(delta, -1, 1)  # Gradient clipping
        
        # Backpropagate through network
        gradients_w = []
        gradients_b = []
        
        for i in reversed(range(len(self.weights))):
            # Compute gradients
            dW = np.dot(self.activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)
            
            if i > 0:  # Not input layer
                # Propagate error backwards
                delta = np.dot(delta, self.weights[i].T)
                
                # Apply derivative of leaky ReLU
                delta = delta * np.where(self.activations[i] > 0, 1, 0.01)
                
                # Apply dropout mask from forward pass
                if i - 1 < len(self.dropout_masks):
                    delta = delta * self.dropout_masks[i - 1] / (1 - self.dropout_rate)
        
        # Advanced optimization: Momentum + RMSprop-style adaptive learning
        decay_rate = 0.9
        epsilon = 1e-8
        
        for i in range(len(self.weights)):
            # Update gradient squared moving average
            self.grad_sq_weights[i] = (decay_rate * self.grad_sq_weights[i] + 
                                     (1 - decay_rate) * gradients_w[i] ** 2)
            self.grad_sq_biases[i] = (decay_rate * self.grad_sq_biases[i] + 
                                    (1 - decay_rate) * gradients_b[i] ** 2)
            
            # Adaptive learning rate
            adaptive_lr_w = self.learning_rate / (np.sqrt(self.grad_sq_weights[i]) + epsilon)
            adaptive_lr_b = self.learning_rate / (np.sqrt(self.grad_sq_biases[i]) + epsilon)
            
            # Momentum update
            self.momentum_weights[i] = (self.momentum_factor * self.momentum_weights[i] + 
                                      adaptive_lr_w * gradients_w[i])
            self.momentum_biases[i] = (self.momentum_factor * self.momentum_biases[i] + 
                                     adaptive_lr_b * gradients_b[i])
            
            # Apply updates
            self.weights[i] -= self.momentum_weights[i]
            self.biases[i] -= self.momentum_biases[i]
        
        self.experience_count += batch_size
        self.training_losses.append(total_loss)
        return total_loss
    
    def record_battle(self, scenario_id, victory, enemies_killed, towers_built, efficiency, wave_reached):
        if victory:
            self.total_battles_won += 1
        
        battle_record = {
            'scenario': scenario_id,
            'victory': victory,
            'enemies_killed': enemies_killed,
            'towers_built': towers_built,
            'efficiency': efficiency,
            'wave_reached': wave_reached,
            'timestamp': time.time()
        }
        
        self.battle_history.append(battle_record)
        
        # Track strategy evolution
        if len(self.battle_history) % 10 == 0:
            recent_wins = sum(1 for b in self.battle_history[-10:] if b['victory'])
            avg_efficiency = np.mean([b['efficiency'] for b in self.battle_history[-10:] if b['victory']])
            
            self.strategy_evolution.append({
                'battles_completed': len(self.battle_history),
                'recent_win_rate': recent_wins / 10.0,
                'avg_efficiency': avg_efficiency if not np.isnan(avg_efficiency) else 0,
                'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0
            })
        
        # Keep memory manageable
        if len(self.battle_history) > 300:
            self.battle_history = self.battle_history[-300:]
    
    def get_performance_stats(self):
        if not self.battle_history:
            return "No battle history yet"
        
        recent_battles = self.battle_history[-50:] if len(self.battle_history) >= 50 else self.battle_history
        recent_win_rate = sum(1 for b in recent_battles if b['victory']) / len(recent_battles) * 100
        
        all_time_win_rate = self.total_battles_won / len(self.battle_history) * 100
        
        victories = [b for b in recent_battles if b['victory']]
        avg_efficiency = sum(b['efficiency'] for b in victories) / len(victories) if victories else 0
        avg_waves = sum(b['wave_reached'] for b in recent_battles) / len(recent_battles)
        
        return {
            'total_experiences': self.experience_count,
            'total_battles': len(self.battle_history),
            'total_victories': self.total_battles_won,
            'all_time_win_rate': all_time_win_rate,
            'recent_win_rate': recent_win_rate,
            'average_efficiency': avg_efficiency,
            'average_waves_reached': avg_waves,
            'network_depth': len(self.layer_sizes) - 1,
            'total_parameters': sum(w.size for w in self.weights) + sum(b.size for b in self.biases)
        }

class AdvancedTowerDefenseGame:
    def __init__(self, width=8, height=8, scenario_seed=None, difficulty="normal"):
        self.width = width
        self.height = height
        self.difficulty = difficulty
        
        if scenario_seed is None:
            scenario_seed = int(time.time() * 1000) % 100000
        
        random.seed(scenario_seed)
        np.random.seed(scenario_seed)
        self.scenario_id = scenario_seed
        
        self.grid = np.zeros((height, width))
        self.path = self.generate_complex_path()
        self.towers = []
        self.enemies = []
        
        # Difficulty-based starting conditions
        difficulty_settings = {
            "easy": {"money": 400, "lives": 30, "max_waves": 8},
            "normal": {"money": 300, "lives": 25, "max_waves": 12},
            "hard": {"money": 200, "lives": 20, "max_waves": 15}
        }
        
        settings = difficulty_settings.get(difficulty, difficulty_settings["normal"])
        self.money = settings["money"]
        self.lives = settings["lives"]
        self.max_waves = settings["max_waves"]
        
        self.wave_number = 1
        self.time = 0
        self.total_enemies_killed = 0
        self.total_money_earned = 0
        
        # Mark path as unbuildable
        for x, y in self.path:
            if 0 <= int(x) < width and 0 <= int(y) < height:
                self.grid[int(y)][int(x)] = -1
    
    def generate_complex_path(self):
        """Generate sophisticated winding paths"""
        path_types = ["straight", "zigzag", "spiral", "maze"]
        path_type = random.choice(path_types)
        
        if path_type == "straight":
            y = random.randint(1, self.height-2)
            return [(x, y) for x in range(self.width)]
        
        elif path_type == "zigzag":
            path = [(0, self.height//2)]
            x, y = 0, self.height//2
            direction = random.choice([-1, 1])
            
            while x < self.width - 1:
                # Move right
                x += 1
                path.append((x, y))
                
                # Occasionally change direction
                if random.random() < 0.4 and x < self.width - 2:
                    new_y = y + direction
                    if 0 <= new_y < self.height:
                        y = new_y
                        path.append((x, y))
                        direction *= -1  # Reverse direction
            
            return path
        
        elif path_type == "spiral":
            # Create a spiral path
            center_x, center_y = self.width//2, self.height//2
            path = [(0, center_y)]
            
            current_x, current_y = 0, center_y
            while current_x < self.width - 1:
                # Move toward center in a spiral
                if current_x < center_x:
                    current_x += 1
                if current_y < center_y and random.random() < 0.3:
                    current_y += 1
                elif current_y > center_y and random.random() < 0.3:
                    current_y -= 1
                
                path.append((current_x, current_y))
            
            return path
        
        else:  # maze-like
            return self.generate_maze_path()
    
    def generate_maze_path(self):
        """Generate a maze-like path with turns"""
        path = [(0, random.randint(1, self.height-2))]
        current_x, current_y = path[0]
        
        while current_x < self.width - 1:
            # 70% chance to move right, 30% chance to move vertically
            if random.random() < 0.7:
                current_x += 1
            else:
                # Move vertically
                directions = []
                if current_y > 0:
                    directions.append(-1)
                if current_y < self.height - 1:
                    directions.append(1)
                
                if directions:
                    current_y += random.choice(directions)
            
            path.append((current_x, current_y))
            
            # Add some complexity with occasional detours
            if random.random() < 0.2 and current_x < self.width - 2:
                # Create a small detour
                detour_length = random.randint(1, 3)
                for _ in range(detour_length):
                    if random.random() < 0.5 and current_y > 0:
                        current_y -= 1
                    elif current_y < self.height - 1:
                        current_y += 1
                    path.append((current_x, current_y))
        
        return path
    
    def get_sophisticated_state(self):
        """Create detailed state representation for deep AI"""
        state = []
        
        # Enhanced grid representation
        grid_features = []
        for y in range(self.height):
            for x in range(self.width):
                cell_value = 0.0
                
                if self.grid[y][x] == -1:  # Path
                    cell_value = -1.0
                elif self.grid[y][x] > 0:  # Tower
                    tower_idx = int(self.grid[y][x]) - 1
                    if tower_idx < len(self.towers):
                        tower = self.towers[tower_idx]
                        if tower.type == "basic":
                            cell_value = 0.3
                        elif tower.type == "sniper":
                            cell_value = 0.6
                        elif tower.type == "rapid":
                            cell_value = 0.9
                        elif tower.type == "cannon":
                            cell_value = 1.2
                
                grid_features.append(cell_value)
        
        # Take representative sample of grid (16 key positions)
        sample_positions = [
            (1,1), (3,1), (5,1), (7,1),
            (1,3), (3,3), (5,3), (7,3),
            (1,5), (3,5), (5,5), (7,5),
            (1,7), (3,7), (5,7), (7,7)
        ]
        
        for x, y in sample_positions:
            if x < self.width and y < self.height:
                state.append(grid_features[y * self.width + x])
            else:
                state.append(0.0)
        
        # Advanced game statistics
        state.extend([
            self.money / 1000.0,                    # Money (normalized)
            self.lives / 30.0,                      # Lives remaining
            len(self.enemies) / 20.0,               # Current enemies
            self.wave_number / 15.0,                # Wave progress
            len(self.towers) / 25.0,                # Tower count
            self.total_enemies_killed / 200.0,      # Kill efficiency
            self.total_money_earned / 1000.0,       # Economic efficiency
            len(self.path) / 20.0,                  # Path complexity
        ])
        
        # Sophisticated enemy analysis
        enemy_threat_levels = np.zeros(4)  # Four quadrants
        enemy_health_levels = np.zeros(4)
        enemy_positions = np.zeros(4)
        
        for enemy in self.enemies:
            if enemy.alive:
                # Determine quadrant
                quad_x = 0 if enemy.x < self.width/2 else 1
                quad_y = 0 if enemy.y < self.height/2 else 1
                quadrant = quad_y * 2 + quad_x
                
                # Calculate threat (closer to end = higher threat)
                threat = 1.0 - (enemy.path_index / max(1, len(self.path) - 1))
                enemy_threat_levels[quadrant] += threat
                
                # Health ratio
                health_ratio = enemy.health / enemy.max_health
                enemy_health_levels[quadrant] += health_ratio
                
                # Position density
                enemy_positions[quadrant] += 1
        
        # Normalize enemy data
        state.extend(enemy_threat_levels / 5.0)
        state.extend(enemy_health_levels / 5.0)
        state.extend(enemy_positions / 10.0)
        
        # Ensure exactly 36 features
        while len(state) < 36:
            state.append(0.0)
        
        return np.array(state[:36], dtype=np.float32)
    
    def get_valid_build_positions(self):
        """Get all valid positions where towers can be built"""
        valid_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 0:  # Empty space
                    valid_positions.append((x, y))
        return valid_positions
    
    def can_build_tower(self, x, y, tower_type="basic"):
        tower_costs = {
            "basic": 80, 
            "sniper": 180, 
            "rapid": 120,
            "cannon": 250
        }
        return (0 <= x < self.width and 0 <= y < self.height and 
                self.grid[y][x] == 0 and self.money >= tower_costs[tower_type])
    
    def build_tower(self, x, y, tower_type="basic"):
        if self.can_build_tower(x, y, tower_type):
            tower = AdvancedTower(x, y, tower_type)
            self.towers.append(tower)
            self.grid[y][x] = len(self.towers)  # Mark as occupied
            self.money -= tower.cost
            return True
        return False
    
    def spawn_enemy_wave(self):
        """Spawn sophisticated enemy waves"""
        base_enemies = 6 + self.wave_number
        max_enemies = min(base_enemies, 20)
        
        enemy_types = [
            {"name": "scout", "health": 80, "speed": 1.5, "reward": 25, "armor": 0},
            {"name": "soldier", "health": 150, "speed": 1.0, "reward": 40, "armor": 1},
            {"name": "tank", "health": 300, "speed": 0.6, "reward": 75, "armor": 3},
            {"name": "speedster", "health": 60, "speed": 2.2, "reward": 30, "armor": 0},
        ]
        
        # Boss enemies every 4th wave
        if self.wave_number % 4 == 0:
            boss_health = 500 + self.wave_number * 50
            boss = AdvancedEnemy(self.path[0][0], self.path[0][1], 
                               health=boss_health, speed=0.4, reward=150, 
                               armor=5, enemy_type="boss")
            self.enemies.append(boss)
        
        # Regular enemies
        for i in range(max_enemies):
            # Choose enemy type based on wave
            if self.wave_number <= 3:
                enemy_type = enemy_types[0]  # Scouts only
            elif self.wave_number <= 6:
                enemy_type = random.choice(enemy_types[:2])  # Scouts and soldiers
            elif self.wave_number <= 10:
                enemy_type = random.choice(enemy_types[:3])  # All except speedsters
            else:
                enemy_type = random.choice(enemy_types)  # All types
            
            # Scale health with wave number
            scaled_health = enemy_type["health"] + (self.wave_number - 1) * 10
            
            enemy = AdvancedEnemy(
                self.path[0][0], self.path[0][1],
                health=scaled_health,
                speed=enemy_type["speed"],
                reward=enemy_type["reward"],
                armor=enemy_type["armor"],
                enemy_type=enemy_type["name"]
            )
            self.enemies.append(enemy)
    
    def update_game(self, dt=0.1):
        """Advanced game update with sophisticated mechanics"""
        self.time += dt
        
        # Spawn new wave if needed
        if not self.enemies and self.wave_number <= self.max_waves:
            self.spawn_enemy_wave()
            self.wave_number += 1
        
        # Move enemies with sophisticated pathfinding
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy.alive:
                reached_end = enemy.move_along_path(self.path)
                if reached_end:
                    # Different enemies cause different damage
                    damage = 1
                    if enemy.enemy_type == "tank":
                        damage = 2
                    elif enemy.enemy_type == "boss":
                        damage = 3
                    
                    self.lives -= damage
                    enemies_to_remove.append(enemy)
            else:
                self.money += enemy.reward
                self.total_money_earned += enemy.reward
                self.total_enemies_killed += 1
                enemies_to_remove.append(enemy)
        
        # Remove dead/escaped enemies
        for enemy in enemies_to_remove:
            self.enemies.remove(enemy)
        
        # Advanced tower combat
        for tower in self.towers:
            tower.shoot([e for e in self.enemies if e.alive], self.time)
        
        # Check win/lose conditions
        if self.lives <= 0:
            return "lose"
        elif self.wave_number > self.max_waves and not self.enemies:
            return "win"
        
        return "continue"
    
    def visualize(self, title_suffix=""):
        """Enhanced visualization with more details"""
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Draw grid
        for i in range(self.height + 1):
            ax.axhline(y=i, color='lightgray', linestyle='-', alpha=0.3)
        for j in range(self.width + 1):
            ax.axvline(x=j, color='lightgray', linestyle='-', alpha=0.3)
        
        # Draw path with complexity indication
        if len(self.path) > 10:
            path_color = 'darkred'  # Complex path
        elif len(self.path) > 8:
            path_color = 'orange'   # Medium path
        else:
            path_color = 'brown'    # Simple path
        
        path_x = [p[0] + 0.5 for p in self.path]
        path_y = [p[1] + 0.5 for p in self.path]
        ax.plot(path_x, path_y, path_color, linewidth=4, alpha=0.8, linestyle='--')
        
        # Draw towers with type indication
        tower_colors = {
            "basic": "blue", 
            "sniper": "red", 
            "rapid": "green", 
            "cannon": "purple"
        }
        
        for tower in self.towers:
            color = tower_colors.get(tower.type, "blue")
            tower_rect = Rectangle((tower.x, tower.y), 1, 1, facecolor=color, alpha=0.8)
            ax.add_patch(tower_rect)
            
            # Draw range circle
            range_circle = Circle((tower.x + 0.5, tower.y + 0.5), tower.range, 
                                fill=False, color=color, alpha=0.3, linestyle=':')
            ax.add_patch(range_circle)
            
            # Add kill count
            if tower.kills > 0:
                ax.text(tower.x + 0.5, tower.y + 0.2, str(tower.kills), 
                       ha='center', va='center', fontsize=8, color='white', weight='bold')
        
        # Draw enemies with type indication
        for enemy in self.enemies:
            if enemy.alive:
                health_ratio = enemy.health / enemy.max_health
                
                # Color based on enemy type
                if enemy.enemy_type == "boss":
                    color = 'darkred'
                    size = 0.4
                elif enemy.enemy_type == "tank":
                    color = 'maroon'
                    size = 0.35
                elif enemy.enemy_type == "speedster":
                    color = 'yellow'
                    size = 0.25
                else:
                    color = 'red' if health_ratio < 0.3 else 'orange' if health_ratio < 0.7 else 'darkred'
                    size = 0.3
                
                enemy_circle = Circle((enemy.x, enemy.y), size, facecolor=color, alpha=0.8)
                ax.add_patch(enemy_circle)
                
                # Health bar for bosses and tanks
                if enemy.enemy_type in ["boss", "tank"]:
                    bar_width = 0.6
                    bar_height = 0.1
                    bar_x = enemy.x + 0.5 - bar_width/2
                    bar_y = enemy.y + 0.6
                    
                    # Background
                    bg_rect = Rectangle((bar_x, bar_y), bar_width, bar_height, 
                                      facecolor='black', alpha=0.5)
                    ax.add_patch(bg_rect)
                    
                    # Health
                    health_rect = Rectangle((bar_x, bar_y), bar_width * health_ratio, bar_height, 
                                          facecolor='green' if health_ratio > 0.5 else 'red', alpha=0.8)
                    ax.add_patch(health_rect)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        
        # Enhanced title with more information
        efficiency = (self.total_enemies_killed / max(1, len(self.towers))) if self.towers else 0
        title = (f'Advanced Tower Defense - Wave {self.wave_number}/{self.max_waves} | '
                f'Lives: {self.lives} | Money: ${self.money} | '
                f'Towers: {len(self.towers)} | Kills: {self.total_enemies_killed} | '
                f'Efficiency: {efficiency:.1f} {title_suffix}')
        ax.set_title(title, fontsize=10)
        
        # Enhanced legend
        legend_elements = [
            plt.Line2D([0], [0], color=path_color, lw=3, label='Enemy Path'),
            plt.Rectangle((0,0),1,1, facecolor='blue', alpha=0.8, label='Basic Tower ($80)'),
            plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.8, label='Sniper Tower ($180)'),
            plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.8, label='Rapid Tower ($120)'),
            plt.Rectangle((0,0),1,1, facecolor='purple', alpha=0.8, label='Cannon Tower ($250)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        
        plt.tight_layout()
        plt.show()

class AdvancedEnemy:
    def __init__(self, x, y, health=100, speed=1, reward=10, armor=0, enemy_type="basic"):
        self.x = x
        self.y = y
        self.health = health
        self.max_health = health
        self.speed = speed
        self.reward = reward
        self.armor = armor  # Damage reduction
        self.enemy_type = enemy_type
        self.alive = True
        self.path_index = 0
        self.status_effects = {}  # For future features like slow, poison, etc.
    
    def take_damage(self, damage):
        # Apply armor reduction
        actual_damage = max(1, damage - self.armor)
        self.health -= actual_damage
        
        if self.health <= 0:
            self.alive = False
            return True
        return False
    
    def move_along_path(self, path):
        if self.path_index < len(path) - 1:
            target = path[self.path_index + 1]
            dx = target[0] - self.x
            dy = target[1] - self.y
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Apply speed modifications
            current_speed = self.speed
            if 'slow' in self.status_effects:
                current_speed *= 0.5
            
            if dist <= current_speed:
                self.path_index += 1
                if self.path_index < len(path):
                    self.x, self.y = path[self.path_index]
            else:
                self.x += (dx / dist) * current_speed
                self.y += (dy / dist) * current_speed
        
        return self.path_index >= len(path) - 1

class AdvancedTower:
    def __init__(self, x, y, tower_type="basic"):
        self.x = x
        self.y = y
        self.type = tower_type
        
        # Enhanced tower types with special abilities
        tower_specs = {
            "basic": {
                "damage": 35, "range": 2.2, "fire_rate": 1.2, "cost": 80,
                "special": None
            },
            "sniper": {
                "damage": 120, "range": 5.0, "fire_rate": 0.4, "cost": 180,
                "special": "piercing"  # Can hit multiple enemies
            },
            "rapid": {
                "damage": 20, "range": 1.8, "fire_rate": 4.0, "cost": 120,
                "special": "rapid_burst"  # Multiple shots per attack
            },
            "cannon": {
                "damage": 80, "range": 2.5, "fire_rate": 0.8, "cost": 250,
                "special": "splash"  # Area damage
            }
        }
        
        specs = tower_specs.get(tower_type, tower_specs["basic"])
        self.damage = specs["damage"]
        self.range = specs["range"]
        self.fire_rate = specs["fire_rate"]
        self.cost = specs["cost"]
        self.special = specs["special"]
        
        self.last_shot_time = 0
        self.kills = 0
        self.total_damage_dealt = 0
        self.shots_fired = 0
    
    def can_shoot(self, current_time):
        return current_time - self.last_shot_time >= (1.0 / self.fire_rate)
    
    def shoot(self, enemies, current_time):
        if not self.can_shoot(current_time):
            return False
        
        targets = []
        
        # Find targets within range
        for enemy in enemies:
            if enemy.alive:
                distance = np.sqrt((enemy.x - self.x)**2 + (enemy.y - self.y)**2)
                if distance <= self.range:
                    targets.append((enemy, distance))
        
        if not targets:
            return False
        
        # Sort by distance (closest first) or priority
        targets.sort(key=lambda x: x[1])
        
        shots_taken = False
        
        if self.special == "piercing" and len(targets) >= 2:
            # Sniper can hit multiple enemies in line
            for enemy, _ in targets[:2]:
                killed = enemy.take_damage(self.damage)
                self.total_damage_dealt += min(self.damage, enemy.health + self.damage)
                if killed:
                    self.kills += 1
            shots_taken = True
            
        elif self.special == "rapid_burst":
            # Rapid tower shoots multiple times
            burst_count = min(3, len(targets))
            for i in range(burst_count):
                if i < len(targets):
                    enemy = targets[i][0]
                    killed = enemy.take_damage(self.damage)
                    self.total_damage_dealt += min(self.damage, enemy.health + self.damage)
                    if killed:
                        self.kills += 1
            shots_taken = True
            
        elif self.special == "splash":
            # Cannon does splash damage
            primary_target = targets[0][0]
            killed = primary_target.take_damage(self.damage)
            self.total_damage_dealt += min(self.damage, primary_target.health + self.damage)
            if killed:
                self.kills += 1
            
            # Splash damage to nearby enemies
            splash_range = 1.5
            for enemy, _ in targets[1:]:
                splash_distance = np.sqrt((enemy.x - primary_target.x)**2 + (enemy.y - primary_target.y)**2)
                if splash_distance <= splash_range:
                    splash_damage = self.damage // 3
                    killed = enemy.take_damage(splash_damage)
                    self.total_damage_dealt += min(splash_damage, enemy.health + splash_damage)
                    if killed:
                        self.kills += 1
            shots_taken = True
            
        else:
            # Basic shooting
            target = targets[0][0]
            killed = target.take_damage(self.damage)
            self.total_damage_dealt += min(self.damage, target.health + self.damage)
            if killed:
                self.kills += 1
            shots_taken = True
        
        if shots_taken:
            self.last_shot_time = current_time
            self.shots_fired += 1
            return True
        
        return False

class SophisticatedTowerDefenseAI:
    def __init__(self, persistent_brain=None):
        if persistent_brain is None:
            self.brain = DeepPersistentTowerAI()
        else:
            self.brain = persistent_brain
        
        self.memory = deque(maxlen=50000)  # Large memory for complex learning
        self.current_game = None
        self.strategy_memory = deque(maxlen=1000)  # Remember successful strategies
    
    def set_game(self, game):
        self.current_game = game
    
    def choose_action(self, state, epsilon=0.1):
        """Sophisticated action selection with strategic analysis"""
        valid_positions = self.current_game.get_valid_build_positions()
        
        if not valid_positions or random.random() < epsilon:
            # Smart random action - prefer positions near path
            if valid_positions:
                # Score positions by distance to path
                scored_positions = []
                for pos in valid_positions:
                    min_path_dist = min(abs(pos[0] - px) + abs(pos[1] - py) 
                                      for px, py in self.current_game.path)
                    score = 1.0 / (min_path_dist + 1)  # Closer = higher score
                    scored_positions.append((pos, score))
                
                # Weighted random selection
                total_score = sum(score for _, score in scored_positions)
                if total_score > 0:
                    weights = [score/total_score for _, score in scored_positions]
                    chosen_idx = np.random.choice(len(scored_positions), p=weights)
                    pos = scored_positions[chosen_idx][0]
                    
                    # Smart tower type selection
                    tower_type = self.select_tower_type(pos, self.current_game)
                    return (*pos, tower_type)
            return None
        
        # AI decision using deep neural network
        q_values = self.brain.forward(state.reshape(1, -1), training=False)[0]
        
        best_action = None
        best_value = float('-inf')
        
        # Enhanced action mapping with strategic considerations
        for i, pos in enumerate(valid_positions[:8]):  # Top 8 positions
            for j, tower_type in enumerate(["basic", "sniper", "rapid", "cannon"]):
                action_index = min(i * 3 + j, 23)  # Map to network output
                
                tower_costs = {"basic": 80, "sniper": 180, "rapid": 120, "cannon": 250}
                if self.current_game.money >= tower_costs[tower_type]:
                    # Base Q-value
                    base_value = q_values[action_index]
                    
                    # Strategic bonuses
                    strategic_value = self.calculate_strategic_value(pos, tower_type, self.current_game)
                    
                    total_value = base_value + strategic_value
                    
                    if total_value > best_value:
                        best_value = total_value
                        best_action = (*pos, tower_type)
        
        return best_action
    
    def select_tower_type(self, position, game):
        """Intelligent tower type selection based on game state"""
        # Analyze current situation
        enemy_count = len([e for e in game.enemies if e.alive])
        wave_progress = game.wave_number / game.max_waves
        money_ratio = game.money / 500.0
        
        # Early game - prefer basic towers
        if wave_progress < 0.3 and money_ratio < 0.5:
            return "basic"
        
        # High enemy count - prefer rapid fire
        if enemy_count > 8:
            return "rapid"
        
        # Late game with money - prefer snipers or cannons
        if wave_progress > 0.6 and game.money > 200:
            return random.choice(["sniper", "cannon"])
        
        # Default balanced choice
        return random.choice(["basic", "rapid"])
    
    def calculate_strategic_value(self, position, tower_type, game):
        """Calculate strategic value of a tower placement"""
        value = 0.0
        
        # Distance to path bonus
        min_path_dist = min(abs(position[0] - px) + abs(position[1] - py) 
                          for px, py in game.path)
        if min_path_dist <= 1:
            value += 0.3
        elif min_path_dist <= 2:
            value += 0.1
        
        # Coverage bonus - prefer positions that cover multiple path segments
        coverage = 0
        tower_range = {"basic": 2.2, "sniper": 5.0, "rapid": 1.8, "cannon": 2.5}[tower_type]
        
        for px, py in game.path:
            dist = np.sqrt((position[0] - px)**2 + (position[1] - py)**2)
            if dist <= tower_range:
                coverage += 1
        
        value += coverage * 0.05
        
        # Synergy with existing towers
        for tower in game.towers:
            tower_dist = np.sqrt((position[0] - tower.x)**2 + (position[1] - tower.y)**2)
            if 1.5 <= tower_dist <= 3.0:  # Good supporting distance
                if (tower_type == "rapid" and tower.type == "sniper") or \
                   (tower_type == "sniper" and tower.type == "rapid"):
                    value += 0.15  # Good combination
        
        # Economic efficiency
        tower_costs = {"basic": 80, "sniper": 180, "rapid": 120, "cannon": 250}
        cost_efficiency = 1.0 - (tower_costs[tower_type] / 300.0)
        value += cost_efficiency * 0.1
        
        return value
    
    def calculate_advanced_reward(self, old_state, new_state, action_taken):
        """Sophisticated reward calculation"""
        reward = 0
        
        # Survival bonus (scaled by difficulty)
        if self.current_game.lives > 0:
            survival_bonus = 10 * (self.current_game.lives / 25.0)
            reward += survival_bonus
        
        # Economic efficiency
        if action_taken:
            tower_costs = {"basic": 80, "sniper": 180, "rapid": 120, "cannon": 250}
            cost = tower_costs.get(action_taken[2], 80)
            
            # Reward efficient spending
            if self.current_game.money > cost * 2:
                reward -= cost * 0.02  # Small penalty for overspending
            else:
                reward += 5  # Bonus for economic efficiency
        
        # Combat effectiveness
        total_kills = sum(tower.kills for tower in self.current_game.towers)
        total_damage = sum(tower.total_damage_dealt for tower in self.current_game.towers)
        reward += total_kills * 20 + total_damage * 0.1
        
        # Wave progression bonus
        wave_bonus = self.current_game.wave_number * 15
        reward += wave_bonus
        
        # Strategic positioning bonus
        if action_taken and len(self.current_game.towers) > 0:
            newest_tower = self.current_game.towers[-1]
            strategic_value = self.calculate_strategic_value(
                (newest_tower.x, newest_tower.y), newest_tower.type, self.current_game
            )
            reward += strategic_value * 50
        
        # Money management
        money_ratio = self.current_game.money / 500.0
        if 0.2 <= money_ratio <= 0.8:
            reward += 10  # Good money management
        elif money_ratio > 0.9:
            reward -= 15  # Hoarding penalty
        
        # End game rewards
        game_result = self.current_game.update_game()
        if game_result == "win":
            # Scale victory bonus by efficiency
            efficiency_bonus = (self.current_game.lives / 25.0) * 500
            wave_completion_bonus = (self.current_game.wave_number / self.current_game.max_waves) * 300
            reward += 1000 + efficiency_bonus + wave_completion_bonus
        elif game_result == "lose":
            # Penalty scaled by how early the loss occurred
            progress_penalty = (1.0 - self.current_game.wave_number / self.current_game.max_waves) * 200
            reward -= (400 + progress_penalty)
        
        return reward
    
    def train_episode(self, epsilon=0.25):
        """Advanced training episode with strategic learning"""
        episode_memory = []
        total_reward = 0
        strategy_actions = []
        
        # Extended building phase for complex strategies
        for step in range(80):  # More building opportunities
            state = self.current_game.get_sophisticated_state()
            action = self.choose_action(state, epsilon)
            
            old_lives = self.current_game.lives
            old_money = self.current_game.money
            old_wave = self.current_game.wave_number
            
            # Take action
            if action:
                success = self.current_game.build_tower(action[0], action[1], action[2])
                if success:
                    strategy_actions.append(action)
                else:
                    action = None
            
            # Extended simulation for complex interactions
            for _ in range(20):
                result = self.current_game.update_game()
                if result != "continue":
                    break
            
            new_state = self.current_game.get_sophisticated_state()
            reward = self.calculate_advanced_reward(state, new_state, action)
            total_reward += reward
            
            episode_memory.append((state, action, reward, new_state))
            
            # Check if game ended
            result = self.current_game.update_game()
            if result != "continue":
                # Enhanced end-game bonus distribution
                end_bonus = 300 if result == "win" else -150
                decay_factor = 0.95
                
                for i in range(len(episode_memory)):
                    bonus = end_bonus * (decay_factor ** (len(episode_memory) - i - 1))
                    old_state, old_action, old_reward, old_new_state = episode_memory[i]
                    episode_memory[i] = (old_state, old_action, old_reward + bonus, old_new_state)
                
                # Remember successful strategies
                if result == "win" and strategy_actions:
                    self.strategy_memory.append({
                        'actions': strategy_actions.copy(),
                        'scenario': self.current_game.scenario_id,
                        'efficiency': len(strategy_actions),
                        'wave_completed': self.current_game.wave_number
                    })
                
                break
            
            # Stop if insufficient money
            if self.current_game.money < 80:
                break
        
        # Add to memory
        self.memory.extend(episode_memory)
        
        # Advanced batch training
        if len(self.memory) > 1000:
            loss = self.train_batch(batch_size=512)
        
        final_result = self.current_game.update_game()
        return final_result == "win", total_reward, len(self.current_game.towers)
    
    def train_batch(self, batch_size=512):
        """Advanced batch training with prioritized experience replay"""
        if len(self.memory) < batch_size:
            return 0
        
        # Sample batch with some prioritization
        recent_samples = int(batch_size * 0.7)  # 70% recent experiences
        random_samples = batch_size - recent_samples
        
        batch = []
        batch.extend(random.sample(list(self.memory)[-recent_samples*2:], recent_samples))
        batch.extend(random.sample(list(self.memory), random_samples))
        
        states = np.array([exp[0] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        
        current_q_values = self.brain.forward(states, training=True)
        next_q_values = self.brain.forward(next_states, training=False)
        
        target_q_values = current_q_values.copy()
        
        # Advanced Q-learning with double DQN-style updates
        for i, (state, action, reward, next_state) in enumerate(batch):
            if action is not None:
                action_index = hash(str(action)) % 24
                
                # Target value with some smoothing and future reward consideration
                next_max_q = np.max(next_q_values[i])
                target_value = reward + 0.9 * next_max_q  # Discount factor
                
                # Smooth update
                target_q_values[i][action_index] = (0.1 * target_q_values[i][action_index] + 
                                                   0.9 * target_value)
        
        loss = self.brain.train_step(states, target_q_values)
        return loss
    
    def play_game(self, game):
        """Play a complete game with sophisticated strategy"""
        self.set_game(game)
        
        # Strategic building phase
        for step in range(50):
            state = game.get_sophisticated_state()
            action = self.choose_action(state, epsilon=0)  # No exploration
            
            if action:
                game.build_tower(action[0], action[1], action[2])
            
            # Simulate with more steps for complex interactions
            for _ in range(20):
                result = game.update_game()
                if result != "continue":
                    return result == "win", len(game.towers)
            
            if game.money < 80:
                break
        
        # Let game finish
        while True:
            result = game.update_game()
            if result != "continue":
                return result == "win", len(game.towers)
    
    def save_progress(self):
        self.brain.save_brain()

def run_deep_tower_defense_lab():
    print("*** SOPHISTICATED DEEP LEARNING TOWER DEFENSE AI ***")
    print("*** 6-Layer Neural Network with Advanced Strategic Learning ***")
    print("=" * 70)
    
    # Create sophisticated AI
    agent = SophisticatedTowerDefenseAI()
    
    # Show AI's accumulated knowledge
    stats = agent.brain.get_performance_stats()
    if isinstance(stats, dict):
        print(f"[DEEP-BRAIN] Sophisticated AI Knowledge Base:")
        print(f"   Neural experiences: {stats['total_experiences']:,}")
        print(f"   Battles fought: {stats['total_battles']}")
        print(f"   Strategic victories: {stats['total_victories']}")
        print(f"   All-time win rate: {stats['all_time_win_rate']:.1f}%")
        print(f"   Recent performance: {stats['recent_win_rate']:.1f}%")
        print(f"   Network architecture: {stats['network_depth']} layers, {stats['total_parameters']:,} parameters")
    else:
        print("[DEEP-BRAIN] Fresh sophisticated AI - ready for advanced strategic learning!")
    
    print("\n" + "="*70)
    
    # Create advanced scenario
    difficulty = random.choice(["normal", "hard"])
    scenario_seed = int(time.time() * 1000) % 100000
    game = AdvancedTowerDefenseGame(scenario_seed=scenario_seed, difficulty=difficulty)
    
    print(f"[ADVANCED-SCENARIO] Generated {difficulty} battle scenario #{game.scenario_id}")
    print(f"   Battlefield: {game.width}x{game.height} grid")
    print(f"   Path complexity: {len(game.path)} segments")
    print(f"   Starting resources: ${game.money}, {game.lives} lives")
    print(f"   Mission: Survive {game.max_waves} waves")
    
    game.visualize(title_suffix="(Deep AI Strategic Analysis)")
    
    # Intensive deep learning training
    print(f"[DEEP-TRAINING] Commencing advanced neural network training...")
    print("[INFO] 6-layer deep network learning sophisticated strategies...")
    
    training_victories = 0
    training_losses = []
    
    for episode in range(200):  # Extended training for deep learning
        # Create training scenario
        training_game = AdvancedTowerDefenseGame(scenario_seed=game.scenario_id, difficulty=difficulty)
        agent.set_game(training_game)
        
        # Advanced epsilon decay with exploration phases
        if episode < 50:
            epsilon = 0.8  # High exploration early
        elif episode < 100:
            epsilon = 0.4  # Medium exploration
        elif episode < 150:
            epsilon = 0.2  # Low exploration
        else:
            epsilon = 0.1  # Mostly exploitation
        
        victory, reward, towers_built = agent.train_episode(epsilon)
        
        if victory:
            training_victories += 1
        
        # Track training progress
        if len(agent.brain.training_losses) > 0:
            training_losses.append(agent.brain.training_losses[-1])
        
        # Progress reports
        if episode % 40 == 39:
            win_rate = training_victories / (episode + 1) * 100
            avg_loss = np.mean(training_losses[-40:]) if training_losses else 0
            print(f"  Episode {episode+1}: {training_victories}/{episode+1} victories ({win_rate:.1f}%), Loss: {avg_loss:.4f}")
    
    # Test the sophisticated AI
    print(f"\n[DEEP-TEST] Testing sophisticated AI on scenario #{game.scenario_id}...")
    test_game = AdvancedTowerDefenseGame(scenario_seed=game.scenario_id, difficulty=difficulty)
    victory, towers_used = agent.play_game(test_game)
    
    if victory:
        efficiency = max(0, 150 - towers_used * 4)  # Higher efficiency scale for advanced game
        print(f"[DEEP-VICTORY] TACTICAL MASTERY! Sophisticated AI conquered the scenario!")
        print(f"   Strategic deployment: {towers_used} towers")
        print(f"   Enemy casualties: {test_game.total_enemies_killed}")
        print(f"   Tactical efficiency: {efficiency:.1f}%")
        print(f"   Lives preserved: {test_game.lives}/{test_game.max_waves + 10}")
        print(f"   Economic performance: ${test_game.total_money_earned} earned")
        test_game.visualize(title_suffix="(DEEP AI TACTICAL VICTORY!)")
    else:
        efficiency = 0
        print(f"[DEEP-ANALYSIS] Strategic defeat - analyzing tactical weaknesses...")
        print(f"   Deployment attempted: {towers_used} towers")
        print(f"   Enemy casualties: {test_game.total_enemies_killed}")
        print(f"   Final wave reached: {test_game.wave_number}/{test_game.max_waves}")
        print(f"   Tactical lessons: {25 - test_game.lives} critical failures")
        test_game.visualize(title_suffix="(Strategic Analysis - Learning Mode)")
    
    # Record battle with comprehensive metrics
    agent.brain.record_battle(game.scenario_id, victory, 
                             test_game.total_enemies_killed,
                             towers_used, efficiency, test_game.wave_number)
    
    # Advanced generalization testing across multiple scenarios
    print(f"\n[STRATEGIC-GENERALIZATION] Testing across 7 diverse battle scenarios...")
    generalization_wins = 0
    total_efficiency = 0
    scenario_types = ["easy_straight", "normal_zigzag", "hard_maze", "boss_rush", "economic_challenge", "speed_trial", "ultimate_test"]
    
    for i, scenario_type in enumerate(scenario_types):
        test_seed = int(time.time() * 1000 + i * 5000) % 100000
        test_difficulty = "easy" if "easy" in scenario_type else "hard" if "hard" in scenario_type else "normal"
        test_scenario = AdvancedTowerDefenseGame(scenario_seed=test_seed, difficulty=test_difficulty)
        
        test_victory, test_towers = agent.play_game(test_scenario)
        
        if test_victory:
            generalization_wins += 1
            eff = max(0, 150 - test_towers * 4)
            total_efficiency += eff
            print(f"   {scenario_type.replace('_', ' ').title()} #{test_scenario.scenario_id}: [VICTORY] Strategic success! ({test_towers} towers, {eff:.1f}% efficient)")
        else:
            print(f"   {scenario_type.replace('_', ' ').title()} #{test_scenario.scenario_id}: [DEFEAT] Tactical challenge ({test_towers} towers, wave {test_scenario.wave_number})")
        
        # Record each strategic test
        test_eff = max(0, 150 - test_towers * 4) if test_victory else 0
        agent.brain.record_battle(test_scenario.scenario_id, test_victory,
                                 test_scenario.total_enemies_killed,
                                 test_towers, test_eff, test_scenario.wave_number)
    
    # Save sophisticated AI progress
    agent.save_progress()
    
    # Comprehensive performance analysis
    avg_efficiency = total_efficiency / generalization_wins if generalization_wins > 0 else 0
    total_victories = (1 if victory else 0) + generalization_wins
    
    print(f"\n[DEEP-RESULTS] SOPHISTICATED BATTLE SESSION ANALYSIS:")
    print(f"   Primary scenario: {'[VICTORY] Strategic mastery' if victory else '[DEFEAT] Tactical challenge'}")
    print(f"   Generalization tests: {generalization_wins}/7 scenarios conquered")
    print(f"   Total session victories: {total_victories}/8 ({total_victories/8*100:.1f}%)")
    print(f"   Strategic efficiency: {avg_efficiency:.1f}%")
    print(f"   Training iterations: 200 episodes")
    print(f"   Neural network depth: 6 layers")
    
    # Show advanced AI evolution
    final_stats = agent.brain.get_performance_stats()
    if isinstance(final_stats, dict):
        print(f"\n[EVOLVED-INTELLIGENCE] DEEP AI STRATEGIC EVOLUTION:")
        print(f"   Neural experiences: {final_stats['total_experiences']:,}")
        print(f"   Strategic battles: {final_stats['total_battles']}")
        print(f"   Tactical victories: {final_stats['total_victories']}")
        print(f"   All-time win rate: {final_stats['all_time_win_rate']:.1f}%")
        print(f"   Current performance: {final_stats['recent_win_rate']:.1f}%")
        print(f"   Average tactical efficiency: {final_stats['average_efficiency']:.1f}%")
        print(f"   Strategic depth: {final_stats['network_depth']} hidden layers")
        print(f"   Neural parameters: {final_stats['total_parameters']:,} connections")
    
    # Calculate strategic improvement
    if isinstance(stats, dict):
        improvement = final_stats['recent_win_rate'] - stats.get('recent_win_rate', 0)
    else:
        improvement = final_stats['recent_win_rate']
    
    # Advanced progress assessment
    if improvement > 30:
        print(f"[BREAKTHROUGH] EXCEPTIONAL STRATEGIC BREAKTHROUGH! Deep AI achieved +{improvement:.1f}% tactical improvement!")
    elif improvement > 20:
        print(f"[MASTERY] OUTSTANDING STRATEGIC EVOLUTION! Deep learning success: +{improvement:.1f}% improvement!")
    elif improvement > 10:
        print(f"[PROGRESS] EXCELLENT! Advanced neural evolution: +{improvement:.1f}% tactical improvement!")
    elif improvement > -5:
        print(f"[STABLE] Deep AI maintaining sophisticated strategic expertise!")
    else:
        print(f"[ADAPTATION] Deep neural network refining advanced strategies...")
    
    # Training convergence analysis
    if len(agent.brain.training_losses) > 0:
        final_loss = np.mean(agent.brain.training_losses[-20:])
        loss_trend = "converging" if final_loss < 1.0 else "stabilizing"
        print(f"\n[NEURAL-ANALYSIS] Training convergence: {final_loss:.4f} ({loss_trend})")
    
    # Strategic insights
    if len(agent.strategy_memory) > 0:
        successful_strategies = len(agent.strategy_memory)
        avg_strategy_efficiency = np.mean([s['efficiency'] for s in agent.strategy_memory])
        print(f"[STRATEGIC-MEMORY] Learned {successful_strategies} winning strategies, avg efficiency: {avg_strategy_efficiency:.1f}")
    
    print(f"\n[CONTINUE] Run again to further evolve the sophisticated neural network!")
    print(f"[DEEP-SAVE] Advanced AI progress saved to: {agent.brain.save_file}")
    print("=" * 70)

def run_ai_comparison():
    """Compare AI performance across different architectures"""
    print("\n[BONUS] NEURAL ARCHITECTURE PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Test scenarios for comparison
    test_scenarios = [
        {"name": "Basic Challenge", "seed": 12345, "difficulty": "easy"},
        {"name": "Tactical Test", "seed": 54321, "difficulty": "normal"},
        {"name": "Strategic Mastery", "seed": 98765, "difficulty": "hard"}
    ]
    
    print("[COMPARISON] Deep AI vs Previous Architectures...")
    
    deep_ai = SophisticatedTowerDefenseAI()
    
    comparison_results = []
    
    for scenario in test_scenarios:
        print(f"\n[TEST] Scenario: {scenario['name']} (Difficulty: {scenario['difficulty']})")
        
        # Test Deep AI
        game = AdvancedTowerDefenseGame(scenario_seed=scenario["seed"], difficulty=scenario["difficulty"])
        victory, towers = deep_ai.play_game(game)
        
        efficiency = max(0, 150 - towers * 4) if victory else 0
        result = {
            'scenario': scenario['name'],
            'victory': victory,
            'towers': towers,
            'efficiency': efficiency,
            'kills': game.total_enemies_killed,
            'waves': game.wave_number
        }
        comparison_results.append(result)
        
        if victory:
            print(f"   Deep AI: [VICTORY] Strategic success with {towers} towers ({efficiency:.1f}% efficient)")
        else:
            print(f"   Deep AI: [DEFEAT] Tactical challenge, {towers} towers built (wave {game.wave_number})")
    
    # Summary
    victories = sum(1 for r in comparison_results if r['victory'])
    avg_efficiency = np.mean([r['efficiency'] for r in comparison_results if r['victory']])
    
    print(f"\n[SUMMARY] Deep AI Performance:")
    print(f"   Overall success rate: {victories}/{len(test_scenarios)} ({victories/len(test_scenarios)*100:.1f}%)")
    print(f"   Average efficiency: {avg_efficiency:.1f}%")
    print(f"   Strategic adaptability: {'Excellent' if victories >= 2 else 'Developing'}")

def run_strategy_analysis():
    """Analyze and visualize AI strategy evolution"""
    print("\n[BONUS] AI STRATEGY EVOLUTION ANALYSIS")
    print("=" * 50)
    
    try:
        ai = DeepPersistentTowerAI()
        
        if len(ai.strategy_evolution) > 0:
            print("[EVOLUTION] Strategy development over time:")
            
            for i, evolution in enumerate(ai.strategy_evolution[-10:]):  # Last 10 data points
                battles = evolution['battles_completed']
                win_rate = evolution['recent_win_rate'] * 100
                efficiency = evolution['avg_efficiency']
                loss = evolution['avg_loss']
                
                print(f"   Checkpoint {i+1}: {battles} battles, {win_rate:.1f}% wins, {efficiency:.1f}% efficiency, {loss:.4f} loss")
            
            # Trend analysis
            if len(ai.strategy_evolution) >= 2:
                recent_wr = ai.strategy_evolution[-1]['recent_win_rate']
                early_wr = ai.strategy_evolution[0]['recent_win_rate']
                improvement = (recent_wr - early_wr) * 100
                
                print(f"\n[TREND] Overall strategic improvement: {improvement:+.1f}% win rate")
                print(f"[TREND] Learning trajectory: {'Ascending' if improvement > 0 else 'Stabilizing'}")
        else:
            print("[INFO] No strategy evolution data available - AI needs more training")
    
    except Exception as e:
        print(f"[ERROR] Could not analyze strategy evolution: {e}")

def create_training_scenario(difficulty="normal", scenario_type="balanced"):
    """Create custom training scenarios for specific learning objectives"""
    
    scenario_configs = {
        "speed_training": {
            "enemy_speed_multiplier": 1.5,
            "enemy_health_multiplier": 0.8,
            "description": "Fast enemies, lower health - teaches rapid response"
        },
        "tank_training": {
            "enemy_speed_multiplier": 0.6,
            "enemy_health_multiplier": 2.0,
            "description": "Slow, high-health enemies - teaches sustained DPS"
        },
        "economic_training": {
            "starting_money_multiplier": 0.5,
            "enemy_reward_multiplier": 0.7,
            "description": "Limited resources - teaches economic efficiency"
        },
        "balanced": {
            "enemy_speed_multiplier": 1.0,
            "enemy_health_multiplier": 1.0,
            "description": "Standard balanced gameplay"
        }
    }
    
    config = scenario_configs.get(scenario_type, scenario_configs["balanced"])
    print(f"[SCENARIO] Created {scenario_type} training scenario: {config['description']}")
    
    return config

# Clear old memory files to start fresh with sophisticated AI (optional)
def reset_ai_memory():
    """Reset AI memory to start fresh learning"""
    brain_files = [
        "deep_tower_ai_brain.pkl",
        "tower_ai_brain.pkl",
        "lightweight_tower_ai.pkl"
    ]
    
    deleted_files = []
    for file in brain_files:
        if os.path.exists(file):
            os.remove(file)
            deleted_files.append(file)
    
    if deleted_files:
        print(f"[RESET] Deleted AI memory files: {', '.join(deleted_files)}")
        print("[RESET] AI will start learning from scratch")
    else:
        print("[RESET] No existing AI memory files found")

# Uncomment the line below to reset AI memory (start fresh)
# reset_ai_memory()

# Run the Sophisticated Deep Tower Defense AI Lab
if __name__ == "__main__":
    try:
        run_deep_tower_defense_lab()
        
        # Optional: Run additional analyses
        print("\n" + "="*70)
        response = input("Run AI comparison analysis? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            run_ai_comparison()
        
        print("\n" + "="*70)
        response = input("Analyze strategy evolution? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            run_strategy_analysis()
            
    except MemoryError:
        print("\n[MEMORY-ERROR] Insufficient memory for deep AI on this device.")
        print("[RECOMMENDATION] Use the lightweight version instead:")
        print("exec(open('lightweight_tower_ai.py').read())")
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        print("[RECOVERY] Try the lightweight version if issues persist")