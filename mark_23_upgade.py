# นำเข้าโมดูลที่จำเป็นสำหรับการคำนวณ, การเรียนรู้ของเครื่อง, และการแสดงผล
import numpy as np  # สำหรับการจัดการอาร์เรย์และการคำนวณเชิงตัวเลข
import tensorflow as tf  # เฟรมเวิร์กสำหรับสร้างและฝึกโมเดล Deep Learning
from tensorflow.keras.models import Sequential  # โมเดล Sequential สำหรับสร้างโครงข่ายประสาทเทียมแบบชั้นต่อชั้น
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization  # ชั้น Dense, Dropout, และ LayerNormalization สำหรับโครงข่าย
from tensorflow.keras.optimizers import Adam  # ออปติไมเซอร์ Adam สำหรับปรับพารามิเตอร์โมเดล
from collections import deque  # โครงสร้างข้อมูลแบบ deque สำหรับเก็บประสบการณ์ (replay memory)
import random  # สำหรับการสุ่มตัวเลือก เช่น การเลือกแอคชันแบบสุ่มใน DQN
import pyvista as pv  # ไลบรารีสำหรับการสร้างภาพ 3 มิติของรถบรรทุกและกล่อง
import matplotlib.pyplot as plt  # สำหรับสร้างกราฟและพล็อตข้อมูล
import seaborn as sns  # สำหรับสร้าง heatmap เพื่อแสดงการกระจายน้ำหนัก
import pandas as pd  # สำหรับจัดการข้อมูลในรูปแบบตาราง เช่น การบันทึกประวัติการฝึก
import os  # สำหรับการจัดการไฟล์และไดเรกทอรี เช่น การบันทึกโมเดล

# คลาส Truck จำลองรถบรรทุกสำหรับจัดวางกล่อง
class Truck:
    def __init__(self, width, length, height, max_weight):
        # กำหนดขนาดและน้ำหนักสูงสุดของรถ
        self.width = width  # ความกว้างของรถ (หน่วยตาราง)
        self.length = length  # ความยาวของรถ (หน่วยตาราง)
        self.height = height  # ความสูงของรถ (หน่วยตาราง)
        self.max_weight = max_weight  # น้ำหนักสูงสุดที่รถรับได้ (หน่วยน้ำหนัก)
        # สร้างอาร์เรย์ 3 มิติเพื่อจำลองพื้นที่ในรถ (True = มีกล่อง, False = ว่าง)
        self.container = np.zeros((width, length, height), dtype=bool)
        self.current_weight = 0  # น้ำหนักรวมของกล่องที่วางแล้ว
        self.placed_boxes = []  # ลิสต์เก็บข้อมูลกล่องที่วางในรถ (ID, ตำแหน่ง, ขนาด, น้ำหนัก)

    def reset(self):
        # รีเซ็ตสถานะของรถสำหรับการเริ่ม episode ใหม่
        self.container = np.zeros((self.width, self.length, self.height), dtype=bool)  # ล้างพื้นที่ในรถ
        self.current_weight = 0  # รีเซ็ตน้ำหนักรวม
        self.placed_boxes = []  # ล้างลิสต์กล่องที่วาง

    def can_place(self, box_size, pos, parcel_weight):
        # ตรวจสอบว่าสามารถวางกล่องขนาด box_size ที่ตำแหน่ง pos ได้หรือไม่
        x, y, z = pos  # แยกพิกัดตำแหน่ง (x, y, z)
        w, l, h = box_size  # แยกขนาดกล่อง (กว้าง, ยาว, สูง)
        # ตรวจสอบว่าเกินขอบเขตของรถหรือไม่
        if (x + w > self.width or y + l > self.length or z + h > self.height):
            return False
        # ตรวจสอบว่ามีกล่องอื่นทับซ้อนในพื้นที่หรือไม่
        if np.any(self.container[x:x+w, y:y+l, z:z+h]):
            return False
        # ถ้าตำแหน่ง z > 0 ตรวจสอบพื้นผิวรองรับและน้ำหนัก
        if z > 0:
            # คำนวณพื้นที่รองรับด้านล่างกล่อง
            support_area = self.container[x:x+w, y:y+l, z-1]
            # ต้องการพื้นผิวรองรับอย่างน้อย 95% ของฐานกล่อง
            if np.sum(support_area) < (w * l * 0.95):
                return False
            # ตรวจสอบน้ำหนักของกล่องด้านล่าง
            max_weight_below = 0
            for box in self.placed_boxes:
                bx, by, bz = box['position']
                bw, bl, bh = box['size']
                # ตรวจสอบว่ากล่องด้านล่างอยู่ในตำแหน่งที่รองรับกล่องใหม่
                if (bx < x + w and bx + bw > x and by < y + l and by + bl > y and bz + bh == z):
                    max_weight_below = max(max_weight_below, box['weight'])
            # กล่องใหม่ต้องไม่หนักเกิน 1.5 เท่าของกล่องด้านล่าง เพื่อความปลอดภัย
            if parcel_weight > max_weight_below * 1.5:
                return False
        return True  # ผ่านทุกเงื่อนไข สามารถวางได้

    def place_box(self, box, pos, rotation_idx):
        # วางกล่องในรถที่ตำแหน่งและการหมุนที่ระบุ
        x, y, z = pos  # ตำแหน่งวางกล่อง
        # คำนวณขนาดกล่องหลังการหมุน
        w, l, h = self.get_rotated_size(box['size'], rotation_idx)
        # อัปเดตพื้นที่ในรถเป็น True ในตำแหน่งที่วางกล่อง
        self.container[x:x+w, y:y+l, z:z+h] = True
        # เพิ่มน้ำหนักกล่องลงในน้ำหนักรวม
        self.current_weight += box['weight']
        # สร้างสีสุ่มสำหรับแสดงผลในภาพ 3 มิติ
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        # เก็บข้อมูลกล่องในลิสต์
        self.placed_boxes.append({
            'id': box['id'],  # รหัสกล่อง
            'position': pos,  # ตำแหน่ง (x, y, z)
            'size': (w, l, h),  # ขนาดหลังหมุน
            'weight': box['weight'],  # น้ำหนัก
            'color': color,  # สีสำหรับแสดงผล
            'rotation_idx': rotation_idx  # ดัชนีการหมุน
        })

    def get_rotated_size(self, size, rotation_idx):
        # คืนขนาดกล่องหลังหมุนตามดัชนีที่ระบุ
        w, l, h = size  # ขนาดเดิมของกล่อง
        # กำหนดการหมุนทั้ง 6 แบบ (ทุกแนวที่เป็นไปได้)
        rotations = [(w, l, h), (l, w, h), (h, w, l), (w, h, l), (l, h, w), (h, l, w)]
        return rotations[rotation_idx % 6]  # คืนขนาดตามการหมุนที่เลือก

# คลาส PlacementGrid ช่วยค้นหาตำแหน่งที่วางกล่องได้อย่างมีประสิทธิภาพ
class PlacementGrid:
    def __init__(self, truck):
        # รับออบเจกต์ Truck เพื่อเข้าถึงข้อมูลรถ
        self.truck = truck
        # สร้างกริดขนาดเดียวกับรถ (True = ว่าง, False = มีกล่อง)
        self.grid = np.zeros((truck.width, truck.length, truck.height), dtype=bool)

    def update(self):
        # อัปเดตกริดโดยกำหนดให้พื้นที่ว่างเป็น True และพื้นที่ที่มีกล่องเป็น False
        self.grid = ~self.truck.container

    def find_positions(self, box_size, parcel_weight):
        # ค้นหาตำแหน่งทั้งหมดที่สามารถวางกล่องขนาด box_size ได้
        w, l, h = box_size  # ขนาดกล่อง
        positions = []  # ลิสต์เก็บตำแหน่งที่เป็นไปได้
        # วนลูปผ่านทุกตำแหน่งที่เป็นไปได้ในรถ
        for x in range(self.truck.width - w + 1):
            for y in range(self.truck.length - l + 1):
                for z in range(self.truck.height - h + 1):
                    # ตรวจสอบว่าพื้นที่ว่างทั้งหมดในบริเวณนั้นหรือไม่
                    if np.all(self.grid[x:x+w, y:y+l, z:z+h]):
                        # ตรวจสอบเงื่อนไขเพิ่มเติม เช่น พื้นผิวรองรับและน้ำหนัก
                        if self.truck.can_place((w, l, h), (x, y, z), parcel_weight):
                            positions.append((x, y, z))
        return positions  # คืนลิสต์ตำแหน่งที่วางได้

# คลาส DQNAgent จัดการการเรียนรู้และการตัดสินใจด้วย Deep Q-Network
class DQNAgent:
    def __init__(self, state_size, action_size):
        # กำหนดขนาดของ state และ action space
        self.state_size = state_size  # ขนาดของเวกเตอร์สถานะ
        self.action_size = action_size  # จำนวนแอคชันที่เป็นไปได้
        # สร้าง replay memory ขนาด 20000 สำหรับเก็บประสบการณ์
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99  # ค่า discount factor สำหรับรางวัลในอนาคต
        self.epsilon = 1.0  # ค่าเริ่มต้นของ epsilon สำหรับ epsilon-greedy
        self.epsilon_min = 0.01  # ค่า epsilon ต่ำสุด
        self.epsilon_decay = 0.99  # อัตราการลด epsilon ในแต่ละ episode
        self.learning_rate = 0.001  # อัตราการเรียนรู้สำหรับโมเดล
        # สร้างโมเดลหลักและโมเดลเป้าหมาย
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # อัปเดตโมเดลเป้าหมายให้เหมือนโมเดลหลัก
        self.tau = 0.01  # ค่าสำหรับ soft update ของโมเดลเป้าหมาย
        # เก็บประวัติการฝึก เช่น รางวัล, epsilon, การใช้ปริมาตร
        self.training_history = {
            'episode': [], 'total_reward': [], 'epsilon': [],
            'volume_usage': [], 'boxes_placed': []
        }
        self.best_reward = float('-inf')  # รางวัลที่ดีที่สุดเริ่มต้นที่ลบอนันต์
        self.best_model_path = "best_dqn_model.keras"  # เส้นทางบันทึกโมเดลที่ดีที่สุด

    def _build_model(self):
        # สร้างโครงข่ายประสาทเทียมสำหรับ DQN
        model = Sequential()
        # กำหนดชั้น input ตามขนาด state
        model.add(tf.keras.Input(shape=(self.state_size,)))
        # ชั้น Dense 1024 โหนด, ใช้ ReLU เพื่อเพิ่มความไม่เชิงเส้น
        model.add(Dense(1024, activation='relu'))
        # LayerNormalization เพื่อทำให้การฝึกมีเสถียรภาพมากขึ้น
        model.add(LayerNormalization())
        # Dropout 30% เพื่อป้องกัน overfitting
        model.add(Dropout(0.3))
        # ชั้น Dense 512 โหนด
        model.add(Dense(512, activation='relu'))
        model.add(LayerNormalization())
        model.add(Dropout(0.3))
        # ชั้น Dense 256 และ 128 โหนด สำหรับการประมวลผลเพิ่มเติม
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        # ชั้น output ให้ Q-value สำหรับแต่ละแอคชัน
        model.add(Dense(self.action_size, activation='linear'))
        # คอมไพล์โมเดลโดยใช้ Huber loss (ลดความไวต่อ outlier) และ Adam optimizer
        model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # อัปเดตโมเดลเป้าหมายให้มีน้ำหนักเท่ากับโมเดลหลัก
        self.target_model.set_weights(self.model.get_weights())

    def soft_update_target_model(self):
        # อัปเดตโมเดลเป้าหมายแบบค่อยเป็นค่อยไป (soft update)
        target_weights = self.target_model.get_weights()
        model_weights = self.model.get_weights()
        # ปรับน้ำหนักโดยใช้ tau (0.01) เพื่อผสมน้ำหนักโมเดลหลักและเป้าหมาย
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def save_model(self, model_path="dqn_model.keras", target_model_path="dqn_target_model.keras"):
        # บันทึกโมเดลหลักและโมเดลเป้าหมายลงไฟล์
        self.model.save(model_path)
        self.target_model.save(target_model_path)
        print(f"Saved models to {model_path} and {target_model_path}")

    def save_best_model(self):
        # บันทึกโมเดลหลักเมื่อได้รางวัลที่ดีที่สุด
        self.model.save(self.best_model_path)
        print(f"Saved best model to {self.best_model_path}")

    def load_model(self, model_path="dqn_model.keras", target_model_path="dqn_target_model.keras"):
        # โหลดโมเดลหลักและโมเดลเป้าหมายจากไฟล์
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.target_model = tf.keras.models.load_model(target_model_path)
            print(f"Loaded models from {model_path} and {target_model_path}")
        except Exception as e:
            # ถ้าโหลดไม่ได้ ให้สร้างโมเดลใหม่
            print(f"Error loading models: {e}")
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()

    def remember(self, state, action, reward, next_state, done):
        # เก็บประสบการณ์ลงใน replay memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, truck, remaining_parcels):
        # เลือกแอคชัน (กล่องและการหมุน) และตำแหน่งสำหรับวาง
        valid_actions = []  # ลิสต์เก็บแอคชันที่เป็นไปได้
        # สร้าง PlacementGrid เพื่อค้นหาตำแหน่งว่าง
        placement_grid = PlacementGrid(truck)
        placement_grid.update()

        # จัดเรียงกล่องตามปริมาตรและน้ำหนัก (ใหญ่และหนักก่อน) เพื่อเลียนแบบมนุษย์
        sorted_parcels = sorted(
            [(i, p) for i, p in enumerate(remaining_parcels)],
            key=lambda x: (x[1]['size'][0] * x[1]['size'][1] * x[1]['size'][2], x[1]['weight']),
            reverse=True
        )

        # วนลูปหาแอคชันที่ถูกต้องสำหรับกล่องแต่ละใบ
        for parcel_idx, parcel in sorted_parcels:
            for rot in range(6):  # ลองการหมุนทั้ง 6 แบบ
                rotated_size = truck.get_rotated_size(parcel['size'], rot)
                # ค้นหาตำแหน่งที่วางได้
                positions = placement_grid.find_positions(rotated_size, parcel['weight'])
                if positions:
                    # เลือกตำแหน่งที่มี z ต่ำสุด (ใกล้พื้น) และ y ต่ำสุด (ใกล้ด้านหน้า)
                    positions.sort(key=lambda p: (p[2], p[1]))
                    pos = positions[0]
                    action = parcel_idx * 6 + rot  # แปลงเป็นดัชนีแอคชัน
                    valid_actions.append((action, pos))
                    break  # ใช้การหมุนแรกที่วางได้

        # ถ้าไม่มีแอคชันที่ถูกต้อง คืนค่า -1
        if not valid_actions:
            return -1, None

        # ใช้ epsilon-greedy ในการเลือกแอคชัน
        if np.random.rand() <= self.epsilon:
            # สุ่มเลือกแอคชัน
            action, pos = random.choice(valid_actions)
        else:
            # เลือกแอคชันที่ดีที่สุดจากโมเดล
            act_values = self.model.predict(state, verbose=0)[0]
            best_action_idx = max(range(len(valid_actions)), key=lambda i: act_values[valid_actions[i][0]])
            action, pos = valid_actions[best_action_idx]

        return action, pos  # คืนแอคชันและตำแหน่ง

    def replay(self, batch_size=128):
        # ฝึกโมเดลโดยใช้ประสบการณ์สุ่มจาก replay memory
        if len(self.memory) < batch_size:
            return  # ถ้าประสบการณ์ไม่พอ ข้ามการฝึก
        # สุ่มเลือก batch จาก memory
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0][0] for m in minibatch])  # รวบรวม states
        next_states = np.array([m[3][0] for m in minibatch])  # รวบรวม next_states
        # คำนวณ Q-values สำหรับ states ปัจจุบัน
        targets = self.model.predict(states, verbose=0)
        # คำนวณ Q-values สำหรับ next_states จากโมเดลเป้าหมาย
        next_q_values = self.target_model.predict(next_states, verbose=0)
        # คำนวณ Q-values สำหรับ next_states จากโมเดลหลัก (สำหรับ Double DQN)
        next_q_values_main = self.model.predict(next_states, verbose=0)

        # อัปเดต Q-values ตาม Bellman equation
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                # ใช้ Double DQN: เลือกแอคชันจากโมเดลหลัก ประเมิน Q-value จากโมเดลเป้าหมาย
                next_action = np.argmax(next_q_values_main[i])
                target = reward + self.gamma * next_q_values[i][next_action]
            targets[i][action] = target  # อัปเดต Q-value สำหรับแอคชันที่เลือก

        # ฝึกโมเดลหลักด้วย states และ targets
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=32)
        # อัปเดตโมเดลเป้าหมายแบบ soft update
        self.soft_update_target_model()
        # ลดค่า epsilon เพื่อลดการสุ่ม
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def log_training(self, episode, total_reward, volume_usage, boxes_placed):
        # บันทึกข้อมูลการฝึกสำหรับแต่ละ episode
        self.training_history['episode'].append(episode)
        self.training_history['total_reward'].append(total_reward)
        self.training_history['epsilon'].append(self.epsilon)
        self.training_history['volume_usage'].append(volume_usage)
        self.training_history['boxes_placed'].append(boxes_placed)
        # ถ้ารางวัลสูงสุดใหม่ บันทึกโมเดล
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.save_best_model()

    def plot_training_history(self):
        # สร้างกราฟแสดงผลประวัติการฝึก
        df = pd.DataFrame(self.training_history)
        plt.figure(figsize=(15, 10))  # กำหนดขนาดกราฟ

        # กราฟรางวัลรวมต่อ episode
        plt.subplot(2, 2, 1)
        plt.plot(df['episode'], df['total_reward'], label='Total Reward')
        plt.title('Total Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.legend()

        # กราฟค่า epsilon ต่อ episode
        plt.subplot(2, 2, 2)
        plt.plot(df['episode'], df['epsilon'], label='Epsilon', color='orange')
        plt.title('Epsilon Decay per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
        plt.legend()

        # กราฟการใช้ปริมาตรต่อ episode
        plt.subplot(2, 2, 3)
        plt.plot(df['episode'], df['volume_usage'], label='Volume Usage', color='green')
        plt.title('Volume Usage per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Volume Used')
        plt.grid(True)
        plt.legend()

        # กราฟจำนวนกล่องที่วางได้ต่อ episode
        plt.subplot(2, 2, 4)
        plt.plot(df['episode'], df['boxes_placed'], label='Boxes Placed', color='purple')
        plt.title('Number of Boxes Placed per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Number of Boxes')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()  # ปรับระยะห่างกราฟ
        plt.savefig('training_history.png', dpi=300)  # บันทึกกราฟ
        plt.close()
        print("Saved training history graph to 'training_history.png'")

# ฟังก์ชันคำนวณรางวัลสำหรับการวางกล่อง
def calculate_reward(truck, parcel, pos, rotated_size):
    x, y, z = pos  # ตำแหน่งวางกล่อง
    w, l, h = rotated_size  # ขนาดกล่องหลังหมุน
    truck_volume = truck.width * truck.length * truck.height  # ปริมาตรรวมของรถ
    parcel_volume = w * l * h  # ปริมาตรของกล่อง
    volume_used = np.sum(truck.container)  # ปริมาตรที่ใช้ไปแล้ว
    
    # รางวัลพื้นฐาน
    reward = 100
    # รางวัลตามสัดส่วนปริมาตรกล่องต่อรถ
    reward += (parcel_volume / truck_volume) * 10000
    # รางวัลตามการใช้ปริมาตรรวม
    reward += (volume_used / truck_volume) * 5000
    # รางวัลสำหรับการวางใกล้ด้านหน้า (y ต่ำ)
    reward += 5000 * (1 - y / truck.length)
    
    # ถ้าวางสูงกว่า z=0 ตรวจสอบพื้นผิวรองรับ
    if z > 0:
        support_area = truck.container[x:x+w, y:y+l, z-1]
        support_ratio = np.sum(support_area) / (w * l)
        # รางวัลตามสัดส่วนพื้นที่รองรับ
        reward += 1000 * support_ratio
        # ลงโทษถ้าพื้นที่รองรับน้อยกว่า 80%
        if support_ratio < 0.8:
            reward -= 500
    
    # ลงโทษถ้ามีช่องว่างด้านบน
    if z + h < truck.height and not np.any(truck.container[x:x+w, y:y+l, z+h:]):
        reward -= 50 * (truck.height - (z + h))
    
    # รางวัลถ้าวางชิดขอบ (x=0, y=0 หรือชิดด้านข้าง/ด้านหลัง)
    if x == 0 or y == 0 or x + w == truck.width or y + l == truck.length:
        reward += 300
    
    # รางวัลสำหรับการวางติดกล่องอื่น (neighbors)
    neighbors = 0
    if x > 0 and np.any(truck.container[x-1, y:y+l, z:z+h]):
        neighbors += 1
    if x + w < truck.width and np.any(truck.container[x+w, y:y+l, z:z+h]):
        neighbors += 1
    if y > 0 and np.any(truck.container[x:x+w, y-1, z:z+h]):
        neighbors += 1
    if y + l < truck.length and np.any(truck.container[x:x+w, y+l, z:z+h]):
        neighbors += 1
    reward += 200 * neighbors
    
    return reward

# ฟังก์ชันสร้างสถานะ (state) สำหรับ DQN
def get_state(truck, remaining_parcels, max_parcels, max_width, max_length, max_height, max_weight):
    # Padding อาร์เรย์ container ให้มีขนาดเท่ากับรถที่ใหญ่ที่สุด
    padded_container = np.zeros((max_width, max_length, max_height), dtype=np.float32)
    padded_container[:truck.width, :truck.length, :truck.height] = truck.container
    container_flat = padded_container.flatten()  # แปลงเป็นเวกเตอร์ 1 มิติ

    # Normalize ข้อมูลกล่องที่เหลือ
    parcels_info = []
    for p in remaining_parcels:
        w, l, h = p['size']
        weight = p['weight']
        # Normalize ขนาดและน้ำหนักให้อยู่ในช่วง [0, 1]
        parcels_info.extend([
            w / max_width, l / max_length, h / max_height, weight / max_weight
        ])
    max_parcels_size = max_parcels * 4  # ขนาดสูงสุดของข้อมูลกล่อง
    parcels_flat = np.zeros(max_parcels_size, dtype=np.float32)
    if parcels_info:
        parcels_flat[:len(parcels_info)] = parcels_info  # เติมข้อมูลกล่อง

    # เพิ่มข้อมูลขนาดรถที่ normalize แล้ว
    truck_info = np.array([
        truck.width / max_width,
        truck.length / max_length,
        truck.height / max_height,
        truck.max_weight / max_weight
    ], dtype=np.float32)

    # รวม container, parcels, และ truck_info เป็น state เดียว
    state = np.concatenate([container_flat, parcels_flat, truck_info])
    return np.reshape(state, [1, len(state)])  # คืน state ในรูปแบบ (1, state_size)

# ฟังก์ชันโหลดข้อมูลกล่องจากไฟล์ CSV
def load_parcels_from_csv(csv_path):
    try:
        # อ่านไฟล์ CSV
        df = pd.read_csv(csv_path)
        # ตรวจสอบว่ามีคอลัมน์ที่ต้องการครบหรือไม่
        required_columns = {'id', 'width', 'length', 'height', 'weight'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        parcels = []
        # วนลูปผ่านแถวใน CSV
        for _, row in df.iterrows():
            # ข้ามแถวที่มีขนาดหรือน้ำหนักไม่ถูกต้อง
            if row['width'] <= 0 or row['length'] <= 0 or row['height'] <= 0 or row['weight'] <= 0:
                continue
            # เพิ่มข้อมูลกล่องลงในลิสต์
            parcels.append({
                'id': str(row['id']),
                'size': (int(row['width']), int(row['length']), int(row['height'])),
                'weight': float(row['weight'])
            })
        print(f"Loaded parcel data from {csv_path}, count: {len(parcels)} items")
        return parcels
    except Exception as e:
        # ถ้ามีข้อผิดพลาด คืน None
        print(f"Error loading CSV: {e}")
        return None

# ฟังก์ชันสร้างวิดีโอ 3 มิติของการจัดวางกล่อง
def plot_3d_pyvista(truck):
    # กำหนดมุมมองต่างๆ สำหรับวิดีโอ
    views = [
        {'filename': 'front_view.mp4', 'camera_pos': (truck.width/2, -truck.length, truck.height), 'focal_point': (truck.width/2, truck.length/2, truck.height/2), 'description': 'Front view'},
        {'filename': 'back_view.mp4', 'camera_pos': (truck.width/2, 2*truck.length, truck.height), 'focal_point': (truck.width/2, truck.length/2, truck.height/2), 'description': 'Back view'},
        {'filename': 'side_left_view.mp4', 'camera_pos': (-truck.width, truck.length/2, truck.height), 'focal_point': (truck.width/2, truck.length/2, truck.height/2), 'description': 'Left side view'},
        {'filename': 'side_right_view.mp4', 'camera_pos': (2*truck.width, truck.length/2, truck.height), 'focal_point': (truck.width/2, truck.length/2, truck.height/2), 'description': 'Right side view'},
        {'filename': 'top_view.mp4', 'camera_pos': (truck.width/2, truck.length/2, 3*truck.height), 'focal_point': (truck.width/2, truck.length/2, truck.height/2), 'description': 'Top view'},
        {'filename': 'isometric_view.mp4', 'camera_pos': (2*truck.width, 2*truck.length, 2*truck.height), 'focal_point': (truck.width/2, truck.length/2, truck.height/2), 'description': 'Isometric view'},
        {'filename': 'full_rotation.mp4', 'camera_pos': None, 'focal_point': (truck.width/2, truck.length/2, truck.height/2), 'description': 'Full rotation view'},
    ]

    for view in views:
        # สร้าง plotter สำหรับการแสดงผลแบบ off-screen
        plotter = pv.Plotter(off_screen=True)
        # สร้างพื้นของรถ
        floor = pv.Plane(center=(truck.width/2, truck.length/2, 0), direction=(0, 0, 1),
                         i_size=truck.width, j_size=truck.length)
        plotter.add_mesh(floor, color='gray', opacity=0.5)
        # สร้างโครงร่างรถแบบโปร่งใส
        truck_mesh = pv.Cube(center=(truck.width/2, truck.length/2, truck.height/2),
                             x_length=truck.width+1, y_length=truck.length+1, z_length=truck.height)
        plotter.add_mesh(truck_mesh, color='gray', opacity=0.2, show_edges=True)

        # เพิ่มกล่องแต่ละใบลงใน plotter
        for box in truck.placed_boxes:
            x, y, z = box['position']
            w, l, h = box['size']
            box_mesh = pv.Cube(center=(x + w/2, y + l/2, z + h/2), x_length=w, y_length=l, z_length=h)
            plotter.add_mesh(box_mesh, color=box['color'], show_edges=True, line_width=2)
            # เพิ่มป้ายชื่อกล่อง
            plotter.add_point_labels([(x + w/2, y + l/2, z + h)], [box['id']], font_size=15, point_size=1)

        # เพิ่มแสงเพื่อให้ภาพชัดเจน
        plotter.add_light(pv.Light(position=(truck.width/2, truck.length/2, 2*truck.height), focal_point=(truck.width/2, truck.length/2, 0), intensity=1.0))
        # เริ่มบันทึกวิดีโอ
        plotter.open_movie(view['filename'], framerate=10)

        if view['filename'] == 'full_rotation.mp4':
            # สร้างวิดีโอแบบหมุนรอบรถ
            num_frames = 120
            radius = max(truck.width, truck.length) * 2
            for i in range(num_frames):
                angle = 2 * np.pi * i / num_frames
                cam_x = truck.width/2 + radius * np.cos(angle)
                cam_y = truck.length/2 + radius * np.sin(angle)
                cam_z = truck.height + truck.height/2 * np.sin(angle * 2)
                plotter.camera_position = ((cam_x, cam_y, cam_z), view['focal_point'], (0, 0, 1))
                plotter.write_frame()
        else:
            # สร้างวิดีโอแบบมุมคงที่
            plotter.camera_position = (view['camera_pos'], view['focal_point'], (0, 0, 1))
            for i in range(len(truck.placed_boxes) + 5):
                plotter.clear_actors()
                plotter.add_mesh(floor, color='gray', opacity=0.5)
                plotter.add_mesh(truck_mesh, color='gray', opacity=0.2, show_edges=True)
                # แสดงกล่องทีละใบเพื่อสร้างแอนิเมชัน
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

# ฟังก์ชันสร้าง heatmap การกระจายน้ำหนักที่พื้นรถ
def plot_weight_distribution(truck):
    # สร้างอาร์เรย์ 2 มิติสำหรับน้ำหนักที่พื้น
    weight_distribution = np.zeros((truck.width, truck.length))
    for box in truck.placed_boxes:
        x, y, z = box['position']
        w, l, h = box['size']
        # คำนวณน้ำหนักเฉลี่ยต่อหน่วยพื้นที่สำหรับกล่องที่พื้น (z=0)
        if z == 0:
            weight_per_unit = box['weight'] / (w * l)
            weight_distribution[x:x+w, y:y+l] += weight_per_unit
    # สร้าง heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(weight_distribution.T, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Weight (units)'}, xticklabels=range(truck.width), yticklabels=range(truck.length))
    plt.title('Heatmap of Weight Distribution on Truck Floor')
    plt.xlabel('X Axis (Width)')
    plt.ylabel('Y Axis (Length)')
    plt.gca().invert_yaxis()  # พลิกแกน y เพื่อให้สอดคล้องกับทิศทางรถ
    plt.savefig('weight_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved heatmap 'weight_heatmap.png'")

# ฟังก์ชันหลักของโปรแกรม
def main():
    # กำหนดขนาดรถที่ใช้ในการฝึก
    truck_configs = [
        {'width': 8, 'length': 12, 'height': 6, 'max_weight': 4000},  # รถเล็ก
        {'width': 10, 'length': 15, 'height': 8, 'max_weight': 5000},  # รถกลาง
        {'width': 12, 'length': 20, 'height': 10, 'max_weight': 7000}, # รถใหญ่
    ]

    # หาขนาดสูงสุดสำหรับ state space
    max_width = max(config['width'] for config in truck_configs)
    max_length = max(config['length'] for config in truck_configs)
    max_height = max(config['height'] for config in truck_configs)
    max_weight = max(config['max_weight'] for config in truck_configs)

    # โหลดข้อมูลกล่องจาก CSV
    csv_path = "parcels_train_100.csv"
    parcels = load_parcels_from_csv(csv_path)
    if parcels is None or len(parcels) == 0:
        print("Failed to load parcel data. Program will terminate.")
        return

    # กรองกล่องให้สามารถวางในรถที่เล็กที่สุดได้
    min_width = min(config['width'] for config in truck_configs)
    min_length = min(config['length'] for config in truck_configs)
    min_height = min(config['height'] for config in truck_configs)
    filtered_parcels = [
        p for p in parcels
        if (p['size'][0] <= min_width and p['size'][1] <= min_length and p['size'][2] <= min_height) or
           (p['size'][1] <= min_width and p['size'][0] <= min_length and p['size'][2] <= min_height) or
           (p['size'][2] <= min_width and p['size'][1] <= min_length and p['size'][0] <= min_height) or
           (p['size'][0] <= min_width and p['size'][2] <= min_length and p['size'][1] <= min_height) or
           (p['size'][2] <= min_width and p['size'][0] <= min_length and p['size'][1] <= min_height) or
           (p['size'][1] <= min_width and p['size'][2] <= min_length and p['size'][0] <= min_height)
    ]
    if not filtered_parcels:
        print("No parcels fit in the smallest truck. Program will terminate.")
        return
    parcels = filtered_parcels  # ใช้กล่องที่ผ่านการกรอง

    # คำนวณขนาด state และ action space
    max_parcels = len(parcels)
    state_size = max_width * max_length * max_height + max_parcels * 4 + 4  # container + parcels + truck_info
    action_size = max_parcels * 6  # จำนวนกล่อง x การหมุน
    # สร้างเอเจนต์ DQN
    agent = DQNAgent(state_size, action_size)

    # ถามผู้ใช้ว่าโหลดโมเดลที่ฝึกไว้หรือไม่
    load_existing_model = input("Do you want to load a pre-trained model? (y/n): ").lower() == 'y'
    if load_existing_model:
        agent.load_model()
    else:
        # ฝึกโมเดลใหม่
        episodes = 2000  # จำนวน episode ทั้งหมด
        batch_size = 128  # ขนาด batch สำหรับ replay
        target_update_freq = 20  # ความถี่ในการอัปเดตโมเดลเป้าหมาย

        # วนลูปฝึกแต่ละ episode
        for e in range(episodes):
            # สุ่มเลือกขนาดรถสำหรับ episode นี้
            config = random.choice(truck_configs)
            truck = Truck(
                width=config['width'],
                length=config['length'],
                height=config['height'],
                max_weight=config['max_weight']
            )
            truck.reset()
            remaining_parcels = parcels.copy()  # คัดลอกลิสต์กล่อง
            # สร้าง state เริ่มต้น
            state = get_state(truck, remaining_parcels, max_parcels, max_width, max_length, max_height, max_weight)
            total_reward = 0  # รางวัลรวมของ episode
            done = False  # สถานะว่า episode จบหรือไม่
            step = 0  # นับจำนวนขั้นตอน

            # แสดงผลทุก 100 episode
            if e % 100 == 0:
                print(f"\nEpisode {e+1}/{episodes}, Epsilon: {agent.epsilon:.3f}, Truck: {config['width']}x{config['length']}x{config['height']}")

            # วนลูปใน episode จนกว่าจะเสร็จหรือถึงขีดจำกัดขั้นตอน
            while not done and step < 200:
                step += 1
                # เลือกแอคชัน
                action_pos = agent.act(state, truck, remaining_parcels)
                if action_pos == (-1, None):
                    # ไม่มีแอคชันที่ถูกต้อง
                    reward = -50  # ลงโทษ
                    done = True
                else:
                    action, pos = action_pos
                    parcel_idx = action // 6  # หาดัชนีกล่อง
                    rot_idx = action % 6  # หาดัชนีการหมุน
                    parcel = remaining_parcels[parcel_idx]
                    rotated_size = truck.get_rotated_size(parcel['size'], rot_idx)

                    # ตรวจสอบว่าวางได้และน้ำหนักไม่เกิน
                    if (truck.can_place(rotated_size, pos, parcel['weight']) and
                        truck.current_weight + parcel['weight'] <= truck.max_weight):
                        truck.place_box(parcel, pos, rot_idx)
                        reward = calculate_reward(truck, parcel, pos, rotated_size)
                        remaining_parcels.pop(parcel_idx)  # ลบกล่องที่วางแล้ว
                    else:
                        reward = -50  # ลงโทษถ้าวางไม่ได้

                # สร้าง state ถัดไป
                next_state = get_state(truck, remaining_parcels, max_parcels, max_width, max_length, max_height, max_weight)
                # ตรวจสอบว่า episode จบหรือไม่
                done = len(remaining_parcels) == 0 or action_pos == (-1, None)
                total_reward += reward
                # เก็บประสบการณ์
                agent.remember(state, action, reward, next_state, done)
                state = next_state

            # บันทึกข้อมูลการฝึก
            volume_usage = np.sum(truck.container)
            boxes_placed = len(truck.placed_boxes)
            agent.log_training(e + 1, total_reward, volume_usage, boxes_placed)

            # แสดงผลทุก 100 episode
            if e % 100 == 0:
                print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, Volume: {volume_usage}, Boxes: {boxes_placed}")

            # ฝึกโมเดลถ้ามีประสบการณ์เพียงพอ
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # อัปเดตโมเดลเป้าหมายทุก 20 episode
            if e % target_update_freq == 0:
                agent.update_target_model()

        # บันทึกโมเดลและพล็อตประวัติ
        agent.save_model()
        agent.plot_training_history()

    # จำลองการจัดวางครั้งสุดท้ายสำหรับรถทุกขนาด
    for config in truck_configs:
        # สร้างรถตามขนาดที่กำหนด
        truck = Truck(
            width=config['width'],
            length=config['length'],
            height=config['height'],
            max_weight=config['max_weight']
        )
        truck.reset()
        remaining_parcels = parcels.copy()
        # สร้าง state เริ่มต้น
        state = get_state(truck, remaining_parcels, max_parcels, max_width, max_length, max_height, max_weight)
        print(f"\nFinal packing simulation for truck {config['width']}x{config['length']}x{config['height']}:")
        step = 0
        # วนลูปจำลองการวางกล่อง
        while remaining_parcels and step < 150:
            step += 1
            action_pos = agent.act(state, truck, remaining_parcels)
            if action_pos == (-1, None):
                print(f"Step {step}: No more valid actions")
                break
            action, pos = action_pos
            parcel_idx = action // 6
            rot_idx = action % 6
            parcel = remaining_parcels[parcel_idx]
            rotated_size = truck.get_rotated_size(parcel['size'], rot_idx)

            # วางกล่องถ้าเป็นไปได้
            if (truck.can_place(rotated_size, pos, parcel['weight']) and
                truck.current_weight + parcel['weight'] <= truck.max_weight):
                truck.place_box(parcel, pos, rot_idx)
                remaining_parcels.pop(parcel_idx)
                print(f"Step {step}: Placed {parcel['id']} at {pos} with rotation {rot_idx}")
            state = get_state(truck, remaining_parcels, max_parcels, max_width, max_length, max_height, max_weight)

        # แสดงผลกล่องที่วาง
        print("\n=== Parcels Loaded in Truck ===")
        for box in truck.placed_boxes:
            print(f"ID: {box['id']}, Size: {box['size']}, Weight: {box['weight']}, Position: {box['position']}")
        print(f"Total Weight Loaded: {truck.current_weight}")
        print(f"Volume Used: {np.sum(truck.container)}/{truck.width * truck.length * truck.height} ({np.sum(truck.container)/(truck.width * truck.length * truck.height)*100:.2f}%)")
        print("\n=== Remaining Parcels ===")
        for p in remaining_parcels:
            print(f"ID: {p['id']}, Size: {p['size']}, Weight: {p['weight']}")

        # สร้างวิดีโอและ heatmap
        plot_3d_pyvista(truck)
        plot_weight_distribution(truck)

# รันโปรแกรม
if __name__ == "__main__":
    main()