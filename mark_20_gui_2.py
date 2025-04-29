import numpy as np
import tensorflow as tf
import pyvista as pv
import pandas as pd
import random

class Truck:
    def __init__(self, width, length, height, max_weight):
        self.width = width
        self.length = length
        self.height = height
        self.max_weight = max_weight
        self.container = np.zeros((width, length, height), dtype=bool)
        self.current_weight = 0
        self.placed_boxes = []

    def reset(self):
        self.container = np.zeros((self.width, self.length, self.height), dtype=bool)
        self.current_weight = 0
        self.placed_boxes = []

    def can_place(self, box_size, pos):
        x, y, z = pos
        w, l, h = box_size
        if (x + w > self.width or y + l > self.length or z + h > self.height):
            return False
        if np.any(self.container[x:x+w, y:y+l, z:z+h]):
            return False
        if z > 0:
            support_area = self.container[x:x+w, y:y+l, z-1]
            if np.sum(support_area) < (w * l * 0.95):
                return False
        return True

    def place_box(self, box, pos, rotation_idx):
        x, y, z = pos
        w, l, h = self.get_rotated_size(box['size'], rotation_idx)
        self.container[x:x+w, y:y+l, z:z+h] = True
        self.current_weight += box['weight']
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        self.placed_boxes.append({
            'id': box['id'], 'position': pos, 'size': (w, l, h),
            'weight': box['weight'], 'color': color, 'rotation_idx': rotation_idx
        })

    def get_rotated_size(self, size, rotation_idx):
        w, l, h = size
        rotations = [(w, l, h), (l, w, h), (h, w, l), (w, h, l), (l, h, w), (h, l, w)]
        return rotations[rotation_idx % 6]

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = None
        self.epsilon = 0.01  # Add epsilon for exploration

    def load_model(self, model_path="dqn_model.keras"):
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_valid_positions(self, truck, size):
        w, l, h = size
        valid_positions = []
        for z in range(truck.height - h + 1):
            for y in range(truck.length - l + 1):
                for x in range(truck.width - w + 1):
                    if truck.can_place((w, l, h), (x, y, z)):
                        valid_positions.append((x, y, z))
        return valid_positions

    def act(self, state, truck, remaining_parcels):
        valid_actions = []
        current_height = max([b['position'][2] + b['size'][2] for b in truck.placed_boxes] + [0])

        for i, parcel in enumerate(remaining_parcels):
            parcel_volume = parcel['size'][0] * parcel['size'][1] * parcel['size'][2]
            for rot in range(6):
                w, l, h = truck.get_rotated_size(parcel['size'], rot)
                for z in range(min(current_height + 1, truck.height - h + 1)):
                    for y in range(truck.length - l + 1):
                        for x in range(truck.width - w + 1):
                            if truck.can_place((w, l, h), (x, y, z)):
                                action = (i * 6 * truck.width * truck.length * truck.height +
                                        rot * truck.width * truck.length * truck.height +
                                        x * truck.length * truck.height + y * truck.height + z)
                                score = (parcel_volume * 700 + (truck.length - y) * 100 - z * 50 +
                                        (600 if x == 0 or y == 0 else 0))
                                valid_actions.append((action, score))
                                break
                        if valid_actions and (valid_actions[-1][0] % (truck.length * truck.height)) // truck.height == y:
                            break
                    if valid_actions and (valid_actions[-1][0] % (truck.width * truck.length * truck.height)) // (truck.length * truck.height) == x:
                        break

        if not valid_actions:
            return -1
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)[0]

        act_values = self.model.predict(state, verbose=0)[0]
        best_action = max(valid_actions, key=lambda a: act_values[a[0]] + a[1])[0]
        return best_action

def get_state(truck, remaining_parcels, max_parcels, container_flat=None):
    if container_flat is None:
        container_flat = truck.container.flatten().astype(np.float32)
    parcels_info = []
    for p in remaining_parcels:
        parcels_info.extend([*p['size'], p['weight']])
    max_parcels_size = max_parcels * 4
    parcels_flat = np.zeros(max_parcels_size, dtype=np.float32)
    if parcels_info:
        parcels_flat[:len(parcels_info)] = parcels_info
    state = np.concatenate([container_flat, parcels_flat])
    return np.reshape(state, [1, len(state)])

def load_parcels_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        required_columns = {'id', 'width', 'length', 'height', 'weight'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        parcels = [{'id': str(row['id']), 'size': (int(row['width']), int(row['length']), int(row['height'])), 
                   'weight': float(row['weight'])} for _, row in df.iterrows()]
        print(f"Loaded {len(parcels)} parcels from {csv_path}")
        return parcels
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def arrange_all_parcels(truck, agent, parcels, state_size):
    truck.reset()
    remaining_parcels = parcels.copy()
    container_flat = truck.container.flatten().astype(np.float32)
    state = get_state(truck, remaining_parcels, len(parcels), container_flat)
    state = np.reshape(state, [1, state_size])
    arranged_boxes = []
    step = 0

    while remaining_parcels and step < 150:
        step += 1
        action = agent.act(state, truck, remaining_parcels)
        if action == -1:
            print(f"Step {step}: No more placement positions available")
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
            arranged_boxes.append({
                'id': parcel['id'],
                'position': pos,
                'size': rotated_size,
                'weight': parcel['weight'],
                'rotation_idx': rot_idx
            })
            remaining_parcels.pop(parcel_idx)
            print(f"Step {step}: Placed {parcel['id']} at {pos} with rotation {rot_idx}")
            w, l, h = rotated_size
            container_flat[x * truck.length * truck.height + y * truck.height + z:
                          x * truck.length * truck.height + y * truck.height + z + w * l * h] = 1
        else:
            print(f"Step {step}: Unable to place {parcel['id']}")
            break

        state = get_state(truck, remaining_parcels, len(parcels), container_flat)
        state = np.reshape(state, [1, state_size])

    return arranged_boxes, remaining_parcels

def plot_3d_interactive(truck, arranged_boxes, remaining_parcels):
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.background_color = 'white'

    floor = pv.Plane(center=(truck.width/2, truck.length/2, 0), 
                    direction=(0, 0, 1), 
                    i_size=truck.width, j_size=truck.length)
    plotter.add_mesh(floor, color='gray', opacity=0.5)
    truck_mesh = pv.Cube(center=(truck.width/2, truck.length/2, truck.height/2), 
                        x_length=truck.width+1, y_length=truck.length+1, z_length=truck.height)
    plotter.add_mesh(truck_mesh, color='gray', opacity=0.2, show_edges=False)

    box_actors = {}
    for box in arranged_boxes:
        x, y, z = box['position']
        w, l, h = box['size']
        box_mesh = pv.Cube(center=(x + w/2, y + l/2, z + h/2), x_length=w, y_length=l, z_length=h)
        actor = plotter.add_mesh(box_mesh, color=[0.5, 0.5, 0.5], opacity=0.3, show_edges=False, pickable=True)
        actor.VisibilityOff()
        label_actor = plotter.add_point_labels([(x + w/2, y + l/2, z + h)], [box['id']], font_size=15, point_size=1)
        label_actor.VisibilityOff()
        box_actors[actor] = {'id': box['id'], 'label_actor': label_actor, 'box': box}

    displayed_boxes = []
    EXISTING_COLOR = [0.5, 0.5, 0.5]
    NEW_COLOR = [0.0, 0.0, 0.55]

    total_volume = truck.width * truck.length * truck.height
    info_text = plotter.add_text(
        f"Number of parcels: 0/{len(arranged_boxes)}\nTotal weight: 0.00\nVolume: 0/{total_volume} (0%)\nRemaining parcels: {len(remaining_parcels)}",
        position='upper_left', font_size=12, color='black'
    )

    current_index = [0]
    current_weight = [0.0]
    current_volume = [0]

    def update_info():
        volume_percent = (current_volume[0] / total_volume) * 100
        info_text.SetText(0, f"Number of parcels: {len(displayed_boxes)}/{len(arranged_boxes)}\n"
                            f"Total weight: {current_weight[0]:.2f}\n"
                            f"Volume: {current_volume[0]}/{total_volume} ({volume_percent:.2f}%)\n"
                            f"Remaining parcels: {len(remaining_parcels)}")

    def add_parcel():
        if current_index[0] < len(arranged_boxes):
            for actor in box_actors:
                actor.GetProperty().SetColor(EXISTING_COLOR)
                actor.GetProperty().SetOpacity(0.3)
            actor = list(box_actors.keys())[current_index[0]]
            actor.VisibilityOn()
            box_actors[actor]['label_actor'].VisibilityOn()
            actor.GetProperty().SetColor(NEW_COLOR)
            actor.GetProperty().SetOpacity(1.0)
            box = box_actors[actor]['box']
            displayed_boxes.append(box)
            current_weight[0] += box['weight']
            current_volume[0] += box['size'][0] * box['size'][1] * box['size'][2]
            current_index[0] += 1
            update_info()
            plotter.render()

    def remove_parcel():
        if current_index[0] > 0:
            current_index[0] -= 1
            box = displayed_boxes.pop()
            for actor, info in box_actors.items():
                if info['id'] == box['id']:
                    actor.VisibilityOff()
                    info['label_actor'].VisibilityOff()
                    break
            current_weight[0] -= box['weight']
            current_volume[0] -= box['size'][0] * box['size'][1] * box['size'][2]
            if displayed_boxes:
                last_actor = list(box_actors.keys())[current_index[0] - 1]
                last_actor.GetProperty().SetColor(NEW_COLOR)
                last_actor.GetProperty().SetOpacity(1.0)
            update_info()
            plotter.render()

    plotter.add_key_event('a', add_parcel)
    plotter.add_key_event('b', remove_parcel)

    def on_left_click(*args):
        picked = plotter.pick_mouse_position()
        if picked:
            for actor, info in box_actors.items():
                if actor.GetVisibility():
                    mesh = actor.mapper.GetInput()
                    bounds = mesh.bounds
                    x, y, z = picked
                    if (bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3] and bounds[4] <= z <= bounds[5]):
                        plotter.add_text(f"Parcel ID: {info['id']}", position='lower_right', font_size=12, color='black', name='id_text')
                        plotter.render()
                        return
            plotter.remove_actor('id_text')

    plotter.iren.add_observer('LeftButtonPressEvent', on_left_click)
    plotter.add_light(pv.Light(position=(5, 5, 15), focal_point=(5, 5, 0), intensity=1.0))
    plotter.show()

def main():
    truck = Truck(width=10, length=15, height=8, max_weight=5000)
    
    csv_path = "parcels_test2_100.csv"
    parcels = load_parcels_from_csv(csv_path)
    if parcels is None or len(parcels) == 0:
        print("Failed to load parcel data. Program will exit.")
        return

    max_parcels = len(parcels)
    state_size = truck.width * truck.length * truck.height + max_parcels * 4
    action_size = max_parcels * 6 * truck.width * truck.length * truck.height
    agent = DQNAgent(state_size, action_size)

    print("Loading pre-trained model...")
    agent.load_model()

    print("\nArranging all parcels before visualization...")
    arranged_boxes, remaining_parcels = arrange_all_parcels(truck, agent, parcels, state_size)

    print(f"\nSuccessfully arranged: {len(arranged_boxes)} parcels, Remaining: {len(remaining_parcels)} parcels")
    print("Starting interactive simulation:\n- Press 'a' to show the next parcel\n- Press 'b' to go back\n- Left-click a box to see its ID")
    plot_3d_interactive(truck, arranged_boxes, remaining_parcels)

if __name__ == "__main__":
    main()