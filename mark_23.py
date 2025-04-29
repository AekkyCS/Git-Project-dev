import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import pyvista as pv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import imageio

class Truck:
    # Truck class representing a truck with defined dimensions and maximum weight
    def __init__(self, width, length, height, max_weight):
        self.width = width  # Truck width
        self.length = length  # Truck length
        self.height = height  # Truck height
        self.max_weight = max_weight  # Maximum weight capacity
        self.container = np.zeros((width, length, height), dtype=bool)  # 3D array representing truck space
        self.current_weight = 0  # Current weight
        self.placed_boxes = []  # List of placed boxes

    # Reset truck state to empty
    def reset(self):
        self.container = np.zeros((self.width, self.length, self.height), dtype=bool)
        self.current_weight = 0
        self.placed_boxes = []

    # Check if a box can be placed at the specified position
    def can_place(self, box_size, pos):
        x, y, z = pos  # Position x, y, z
        w, l, h = box_size  # Box dimensions
        # Check if the box exceeds truck boundaries
        if (x + w > self.width or y + l > self.length or z + h > self.height):
            return False
        # Check if the space is free
        if np.any(self.container[x:x+w, y:y+l, z:z+h]):
            return False
        # Check bottom support (if not on the floor)
        if z > 0:
            support_area = self.container[x:x+w, y:y+l, z-1]
            if np.sum(support_area) < (w * l * 0.95):  # Requires at least 95% support
                return False
        return True

    # Place a box in the truck and record details
    def place_box(self, box, pos, rotation_idx):
        x, y, z = pos
        w, l, h = self.get_rotated_size(box['size'], rotation_idx)  # Box dimensions after rotation
        self.container[x:x+w, y:y+l, z:z+h] = True  # Mark the space as occupied
        self.current_weight += box['weight']  # Add weight
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))  # Random color for visualization
        self.placed_boxes.append({
            'id': box['id'], 'position': pos, 'size': (w, l, h),
            'weight': box['weight'], 'color': color, 'rotation_idx': rotation_idx
        })

    # Calculate box dimensions after rotation based on rotation_idx
    def get_rotated_size(self, size, rotation_idx):
        w, l, h = size
        rotations = [(w, l, h), (l, w, h), (h, w, l), (w, h, l), (l, h, w), (h, l, w)]  # All 6 rotation patterns
        return rotations[rotation_idx % 6]

class DQNAgent:
    # DQNAgent class for learning with Deep Q-Network (DQN)
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Size of the state
        self.action_size = action_size  # Number of possible actions
        self.memory = deque(maxlen=20000)  # Memory for storing experiences
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial epsilon value (exploration)
        self.epsilon_min = 0.01  # Minimum epsilon value
        self.epsilon_decay = 0.99  # Epsilon decay rate
        self.learning_rate = 0.001  # Learning rate
        self.model = self._build_model()  # Main model
        self.target_model = self._build_model()  # Target model
        self.update_target_model()  # Update target model
        self.tau = 0.01  # Soft update coefficient
        # Training history log
        self.training_history = {
            'episode': [], 'total_reward': [], 'epsilon': [],
            'volume_usage': [], 'boxes_placed': []
        }

    # Build the DQN model structure
    def _build_model(self):
        model = Sequential()
        model.add(tf.keras.Input(shape=(self.state_size,)))
        model.add(Dense(1024, activation='relu'))  # First layer with 1024 nodes
        model.add(LayerNormalization())  # Normalize data
        model.add(Dropout(0.3))  # Reduce overfitting
        model.add(Dense(512, activation='relu'))  # Second layer with 512 nodes
        model.add(LayerNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))  # Third layer with 256 nodes
        model.add(Dense(128, activation='relu'))  # Fourth layer with 128 nodes
        model.add(Dense(self.action_size, activation='linear'))  # Output layer
        model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))  # Use Huber loss
        return model

    # Update target model weights to match main model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Soft update of target model
    def soft_update_target_model(self):
        target_weights = self.target_model.get_weights()
        model_weights = self.model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    # Save models to files
    def save_model(self, model_path="dqn_model.keras", target_model_path="dqn_target_model.keras"):
        self.model.save(model_path)
        self.target_model.save(target_model_path)
        print(f"Saved models to {model_path} and {target_model_path}")

    # Load models from files
    def load_model(self, model_path="dqn_model.keras", target_model_path="dqn_target_model.keras"):
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.target_model = tf.keras.models.load_model(target_model_path)
            print(f"Loaded models from {model_path} and {target_model_path}")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()

    # Store experience in memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Select an action based on the current state
    def act(self, state, truck, remaining_parcels):
        valid_actions = []  # List of valid actions
        current_height = max([b['position'][2] + b['size'][2] for b in truck.placed_boxes] + [0])  # Current maximum height

        # Find all possible actions
        for i, parcel in enumerate(remaining_parcels):
            parcel_volume = parcel['size'][0] * parcel['size'][1] * parcel['size'][2]
            for rot in range(6):  # Try all rotations
                w, l, h = truck.get_rotated_size(parcel['size'], rot)
                for z in range(min(current_height + 1, truck.height - h + 1)):
                    for y in range(truck.length - l + 1):
                        for x in range(truck.width - w + 1):
                            if truck.can_place((w, l, h), (x, y, z)):
                                action = (i * 6 * truck.width * truck.length * truck.height +
                                          rot * truck.width * truck.length * truck.height +
                                          x * truck.length * truck.height + y * truck.height + z)
                                # Calculate score for this action
                                score = (parcel_volume * 700 + (truck.length - y) * 100 - z * 50 +
                                         (600 if x == 0 or y == 0 else 0))
                                valid_actions.append((action, score))
                                break
                        if valid_actions and (valid_actions[-1][0] % (truck.length * truck.height)) // truck.height == y:
                            break
                    if valid_actions and (valid_actions[-1][0] % (truck.width * truck.length * truck.height)) // (truck.length * truck.height) == x:
                        break

        if not valid_actions:  # If no valid actions are available
            return -1
        if np.random.rand() <= self.epsilon:  # Random exploration
            return random.choice(valid_actions)[0]

        # Use the model to select the best action
        act_values = self.model.predict(state, verbose=0)[0]
        best_action = max(valid_actions, key=lambda a: act_values[a[0]] + a[1])[0]
        return best_action

    # Train the model using experiences from memory
    def replay(self, batch_size=128):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)  # Sample from memory
        states = np.array([m[0][0] for m in minibatch])
        next_states = np.array([m[3][0] for m in minibatch])
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        next_q_values_main = self.model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:  # Calculate Q-value using Bellman equation
                next_action = np.argmax(next_q_values_main[i])
                target = reward + self.gamma * next_q_values[i][next_action]
            targets[i][action] = target

        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=32)  # Train the model
        self.soft_update_target_model()  # Update target model
        if self.epsilon > self.epsilon_min:  # Decay epsilon
            self.epsilon *= self.epsilon_decay

    # Log training data for each episode
    def log_training(self, episode, total_reward, volume_usage, boxes_placed):
        self.training_history['episode'].append(episode)
        self.training_history['total_reward'].append(total_reward)
        self.training_history['epsilon'].append(self.epsilon)
        self.training_history['volume_usage'].append(volume_usage)
        self.training_history['boxes_placed'].append(boxes_placed)

    # Plot training history graphs
    def plot_training_history(self):
        df = pd.DataFrame(self.training_history)

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(df['episode'], df['total_reward'], label='Total Reward')
        plt.title('Total Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(df['episode'], df['epsilon'], label='Epsilon', color='orange')
        plt.title('Epsilon Decay per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(df['episode'], df['volume_usage'], label='Volume Usage', color='green')
        plt.title('Volume Usage per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Volume Used')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(df['episode'], df['boxes_placed'], label='Boxes Placed', color='purple')
        plt.title('Number of Boxes Placed per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Number of Boxes')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        plt.close()
        print("Saved training history graph to 'training_history.png'")

# Calculate reward for placing a box
def calculate_reward(truck, parcel, pos, rotated_size):
    x, y, z = pos
    w, l, h = rotated_size
    total_volume = truck.width * truck.length * truck.height
    volume_used = np.sum(truck.container)
    parcel_volume = w * l * h

    reward = 100  # Base reward
    reward += parcel_volume * 200  # Reward based on box volume
    reward += (volume_used / total_volume) * 6000  # Reward based on total volume usage
    if y == 0 or not np.any(truck.container[x:x+w, :y, z:z+h]):
        reward += 10000  # Bonus for front-accessible position
    if parcel_volume < 20:  # Bonus for small boxes
        reward += 5000

    # Bonus for placing near edges
    if x == 0 or y == 0 or (x + w == truck.width) or (y + l == truck.length):
        reward += 600
    if (x == 0 and y == 0) or (x + w == truck.width and y + l == truck.length):
        reward += 400

    # Reward based on weight and support
    if z == 0 and parcel['weight'] > 50:
        reward += 1000  # Bonus for heavy boxes on the floor
    elif z > 0:
        support_area = truck.container[x:x+w, y:y+l, z-1]
        support_ratio = np.sum(support_area) / (w * l)
        if support_ratio >= 0.95:
            reward += 1400  # Bonus for good support
        elif support_ratio < 0.9:
            reward -= 700  # Penalty for poor support
        reward -= z * 50  # Penalty based on height

    # Penalty for empty space above
    if z + h < truck.height and not np.any(truck.container[x:x+w, y:y+l, z+h:]):
        reward -= (truck.height - (z + h)) * 150

    return reward

# Convert truck and parcel state into a state vector
def get_state(truck, remaining_parcels, max_parcels):
    container_flat = truck.container.flatten().astype(np.float32)  # Flatten truck array
    parcels_info = []
    for p in remaining_parcels:
        parcels_info.extend([*p['size'], p['weight']])  # Info of remaining parcels
    max_parcels_size = max_parcels * 4
    parcels_flat = np.zeros(max_parcels_size, dtype=np.float32)
    if parcels_info:
        parcels_flat[:len(parcels_info)] = parcels_info
    state = np.concatenate([container_flat, parcels_flat])  # Combine state
    return np.reshape(state, [1, len(state)])

# Load parcel data from a CSV file
def load_parcels_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        required_columns = {'id', 'width', 'length', 'height', 'weight'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        parcels = [{'id': str(row['id']), 'size': (int(row['width']), int(row['length']), int(row['height'])),
                    'weight': float(row['weight'])} for _, row in df.iterrows()]
        print(f"Loaded parcel data from {csv_path}, count: {len(parcels)} items")
        return parcels
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# Create 3D video showing box placement
def plot_3d_pyvista(truck):
    views = [
        {'filename': 'front_view.mp4', 'camera_pos': (5, -25, 10), 'focal_point': (5, 7.5, 4), 'description': 'Front view'},
        {'filename': 'back_view.mp4', 'camera_pos': (5, 40, 10), 'focal_point': (5, 7.5, 4), 'description': 'Back view'},
        {'filename': 'side_left_view.mp4', 'camera_pos': (-20, 7.5, 10), 'focal_point': (5, 7.5, 4), 'description': 'Left side view'},
        {'filename': 'side_right_view.mp4', 'camera_pos': (30, 7.5, 10), 'focal_point': (5, 7.5, 4), 'description': 'Right side view'},
        {'filename': 'top_view.mp4', 'camera_pos': (5, 7.5, 25), 'focal_point': (5, 7.5, 4), 'description': 'Top view'},
        {'filename': 'isometric_view.mp4', 'camera_pos': (20, 20, 20), 'focal_point': (5, 7.5, 4), 'description': 'Isometric view'},
        {'filename': 'full_rotation.mp4', 'camera_pos': None, 'focal_point': (5, 7.5, 4), 'description': 'Full rotation view'},
    ]

    for view in views:
        plotter = pv.Plotter(off_screen=True)  # Create off-screen plotter
        floor = pv.Plane(center=(truck.width/2, truck.length/2, 0), direction=(0, 0, 1),
                         i_size=truck.width, j_size=truck.length)  # Truck floor
        plotter.add_mesh(floor, color='gray', opacity=0.5)
        truck_mesh = pv.Cube(center=(truck.width/2, truck.length/2, truck.height/2),
                             x_length=truck.width+1, y_length=truck.length+1, z_length=truck.height)  # Truck frame
        plotter.add_mesh(truck_mesh, color='gray', opacity=0.2, show_edges=True)

        # Add each box
        for box in truck.placed_boxes:
            x, y, z = box['position']
            w, l, h = box['size']
            box_mesh = pv.Cube(center=(x + w/2, y + l/2, z + h/2), x_length=w, y_length=l, z_length=h)
            plotter.add_mesh(box_mesh, color=box['color'], show_edges=True, line_width=2)
            plotter.add_point_labels([(x + w/2, y + l/2, z + h)], [box['id']], font_size=15, point_size=1)

        plotter.add_light(pv.Light(position=(5, 5, 15), focal_point=(5, 5, 0), intensity=1.0))  # Add light
        plotter.open_movie(view['filename'], framerate=10)  # Start video creation

        # Create animation based on view
        if view['filename'] == 'full_rotation.mp4':
            num_frames = 120
            radius = 25
            for i in range(num_frames):
                angle = 2 * np.pi * i / num_frames
                cam_x = 5 + radius * np.cos(angle)
                cam_y = 7.5 + radius * np.sin(angle)
                cam_z = 10 + 5 * np.sin(angle * 2)
                plotter.camera_position = ((cam_x, cam_y, cam_z), view['focal_point'], (0, 0, 1))
                plotter.write_frame()
        else:
            plotter.camera_position = (view['camera_pos'], view['focal_point'], (0, 0, 1))
            for i in range(len(truck.placed_boxes) + 5):
                plotter.clear_actors()
                plotter.add_mesh(floor, color='gray', opacity=0.5)
                plotter.add_mesh(truck_mesh, color='gray', opacity=0.2, show_edges=True)
                for j in range(min(i, len(truck.placed_boxes))):
                    box = truck.placed_boxes[j]
                    x, y, z = box['position']
                    w, l, h = box['size']
                    box_mesh = pv.Cube(center=(x + w/2, y + l/2, z + h/2), x_length=w, y_length=l, z_length=h)
                    plotter.add_mesh(box_mesh, color=box['color'], show_edges=True, line_width=2)
                    plotter.add_point_labels([(x + w/2, y + l/2, z + h)], [box['id']], font_size=15, point_size=1)
                plotter.write_frame()

        plotter.close()
        print(f"Saved video '{view['filename']}' ({view['description']})")

# Create a heatmap of weight distribution
def plot_weight_distribution(truck):
    weight_distribution = np.zeros((truck.width, truck.length))
    for box in truck.placed_boxes:
        x, y, z = box['position']
        w, l, h = box['size']
        if z == 0:  # Only boxes on the floor
            weight_per_unit = box['weight'] / (w * l)
            weight_distribution[x:x+w, y:y+l] += weight_per_unit
    plt.figure(figsize=(10, 6))
    sns.heatmap(weight_distribution.T, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Weight (units)'}, xticklabels=range(truck.width), yticklabels=range(truck.length))
    plt.title('Heatmap of Weight Distribution on Truck Floor')
    plt.xlabel('X Axis (Width)')
    plt.ylabel('Y Axis (Length)')
    plt.gca().invert_yaxis()
    plt.savefig('weight_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved heatmap 'weight_heatmap.png'")

# Main function
def main():
    truck = Truck(width=10, length=15, height=8, max_weight=5000)  # Create truck
    csv_path = "parcels_test_100.csv"  # CSV file path
    parcels = load_parcels_from_csv(csv_path)  # Load parcel data
    if parcels is None or len(parcels) == 0:
        print("Failed to load parcel data. Program will terminate.")
        return

    parcels.sort(key=lambda p: (p['size'][0] * p['size'][1] * p['size'][2], p['weight']), reverse=True)  # Sort parcels by volume and weight
    max_parcels = len(parcels)
    state_size = truck.width * truck.length * truck.height + max_parcels * 4  # State size
    action_size = max_parcels * 6 * truck.width * truck.length * truck.height  # Action space size
    agent = DQNAgent(state_size, action_size)  # Create agent

    # Ask user whether to load an existing model or train a new one
    load_existing_model = input("Do you want to load a pre-trained model? (y/n): ").lower() == 'y'
    if load_existing_model:
        agent.load_model()
    else:
        episodes = 2000  # Number of training episodes
        batch_size = 128
        target_update_freq = 20  # Frequency of target model updates

        # Training loop
        for e in range(episodes):
            truck.reset()
            remaining_parcels = parcels.copy()
            state = get_state(truck, remaining_parcels, max_parcels)
            total_reward = 0
            done = False

            if e % 100 == 0:
                print(f"\nEpisode {e+1}/{episodes}, Epsilon: {agent.epsilon:.3f}")

            step = 0
            while not done and step < 200:  # Limit maximum steps
                step += 1
                action = agent.act(state, truck, remaining_parcels)
                if action == -1:
                    reward = -500  # Negative reward when no valid action is available
                    done = True
                else:
                    # Extract components from action
                    parcel_idx = action // (6 * truck.width * truck.length * truck.height)
                    rot_idx = (action // (truck.width * truck.length * truck.height)) % 6
                    pos_idx = action % (truck.width * truck.length * truck.height)
                    x = pos_idx // (truck.length * truck.height)
                    y = (pos_idx // truck.height) % truck.length
                    z = pos_idx % truck.height
                    pos = (x, y, z)
                    parcel = remaining_parcels[parcel_idx]
                    rotated_size = truck.get_rotated_size(parcel['size'], rot_idx)

                    # Place box if possible
                    if (truck.can_place(rotated_size, pos) and
                        truck.current_weight + parcel['weight'] <= truck.max_weight):
                        truck.place_box(parcel, pos, rot_idx)
                        reward = calculate_reward(truck, parcel, pos, rotated_size)
                        remaining_parcels.pop(parcel_idx)
                    else:
                        reward = -100  # Negative reward if placement fails

                next_state = get_state(truck, remaining_parcels, max_parcels)
                done = len(remaining_parcels) == 0 or action == -1
                total_reward += reward
                agent.remember(state, action, reward, next_state, done)
                state = next_state

            # Log training data
            volume_usage = np.sum(truck.container)
            boxes_placed = len(truck.placed_boxes)
            agent.log_training(e + 1, total_reward, volume_usage, boxes_placed)

            if e % 100 == 0:
                print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, Volume: {volume_usage}, Boxes: {boxes_placed}")

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if e % target_update_freq == 0:
                agent.update_target_model()

        agent.save_model()
        agent.plot_training_history()  # Generate graphs after training

    # Final packing simulation
    truck.reset()
    remaining_parcels = parcels.copy()
    state = get_state(truck, remaining_parcels, max_parcels)
    print("\nFinal packing simulation:")
    step = 0
    while remaining_parcels and step < 150:
        step += 1
        action = agent.act(state, truck, remaining_parcels)
        if action == -1:
            print(f"Step {step}: No more valid actions")
            break
        parcel_idx = action // (6 * truck.width * truck.length * truck.height)
        rot_idx = (action // (truck.width * truck.length * truck.height)) % 6
        pos_idx = action % (truck.width * truck.length * truck.height)
        x = pos_idx // (truck.length * truck.height)
        y = (pos_idx // truck.height) % truck.length
        z = pos_idx % truck.height
        pos = (x, y, z)
        parcel = remaining_parcels[parcel_idx]
        rotated_size = truck.get_rotated_size(parcel['size'], rot_idx)

        if (truck.can_place(rotated_size, pos) and
            truck.current_weight + parcel['weight'] <= truck.max_weight):
            truck.place_box(parcel, pos, rot_idx)
            remaining_parcels.pop(parcel_idx)
            print(f"Step {step}: Placed {parcel['id']} at {pos} with rotation {rot_idx}")
        state = get_state(truck, remaining_parcels, max_parcels)

    # Display results
    print("\n=== Parcels Loaded in Truck ===")
    for box in truck.placed_boxes:
        print(f"ID: {box['id']}, Size: {box['size']}, Weight: {box['weight']}, Position: {box['position']}")
    print(f"Total Weight Loaded: {truck.current_weight}")
    print(f"Volume Used: {np.sum(truck.container)}/{truck.width * truck.length * truck.height} ({np.sum(truck.container)/(truck.width * truck.length * truck.height)*100:.2f}%)")
    print("\n=== Remaining Parcels ===")
    for p in remaining_parcels:
        print(f"ID: {p['id']}, Size: {p['size']}, Weight: {p['weight']}")

    plot_3d_pyvista(truck)  # Create 3D video
    plot_2d_layers(truck)  # Create 2D layer images
    plot_weight_distribution(truck)  # Create weight heatmap

# Create 2D images separated by layers
def plot_2d_layers(truck):
    for z in range(truck.height):
        plt.figure(figsize=(8, 6))
        sns.heatmap(truck.container[:, :, z].T, cmap='Blues', cbar=False)
        plt.title(f'Layer z={z}')
        plt.savefig(f'layer_z{z}.png')
        plt.close()

if __name__ == "__main__":
    main()