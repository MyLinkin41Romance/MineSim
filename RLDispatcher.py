from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite
from openmines.src.dump_site import DumpSite
from openmines.src.charging_site import ChargingSite

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state'])

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class RLDispatcher(BaseDispatcher):
    def __init__(self, 
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32):
        super().__init__()
        self.name = "RLDispatcher"
        self.np_random = np.random.RandomState()
        
        # RL参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        # 网络参数
        self.learning_rate = learning_rate
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def init_networks(self, mine: "Mine"):
        """延迟初始化网络"""
        if self.q_network is not None:
            return
            
        # 状态空间：当前位置类型(3) + 各个站点队列长度 + 各个站点来车数量 + 卡车负载状态(1)
        self.state_size = 3 + len(mine.load_sites) + len(mine.dump_sites) + \
                         len(mine.load_sites) + len(mine.dump_sites) + 1
        # 动作空间：所有可能的目标站点
        self.action_size = max(len(mine.load_sites), len(mine.dump_sites))
        
        self.q_network = DQN(self.state_size, self.action_size)
        self.target_network = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.update_target_network()

    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_state(self, truck: "Truck", mine: "Mine") -> np.ndarray:
        """构建当前状态向量"""
        state = []
        
        # 当前位置类型（one-hot编码）
        location_type = [0, 0, 0]  # [ChargingSite, LoadSite, DumpSite]
        if isinstance(truck.current_location, ChargingSite):
            location_type[0] = 1
        elif isinstance(truck.current_location, LoadSite):
            location_type[1] = 1
        elif isinstance(truck.current_location, DumpSite):
            location_type[2] = 1
        state.extend(location_type)
        
        # 各站点队列长度
        for site in mine.load_sites:
            state.append(site.parking_lot.queue_status["total"][int(mine.env.now)])
        for site in mine.dump_sites:
            state.append(site.parking_lot.queue_status["total"][int(mine.env.now)])
            
        # 各站点来车数量
        for site in mine.load_sites:
            state.append(getattr(site, 'coming_truck_num', 0))
        for site in mine.dump_sites:
            state.append(getattr(site, 'coming_truck_num', 0))
        
        # 卡车装载状态
        state.append(1 if truck.truck_load > 0 else 0)
        
        return np.array(state, dtype=np.float32)

    def calculate_reward(self, truck: "Truck", old_state, new_state, mine: "Mine") -> float:
        """计算奖励"""
        reward = 0
        
        # 队列变化的奖励
        old_queues = old_state[3:3+len(mine.load_sites)+len(mine.dump_sites)]
        new_queues = new_state[3:3+len(mine.load_sites)+len(mine.dump_sites)]
        queue_change = np.mean(new_queues) - np.mean(old_queues)
        reward -= queue_change
        
        # 完成运输的奖励
        if truck.truck_load == 0 and isinstance(truck.current_location, DumpSite):
            reward += 10
            
        return reward

    def select_action(self, state: np.ndarray, valid_actions: list) -> int:
        """选择动作"""
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            valid_q_values = q_values[0][valid_actions]
            return valid_actions[torch.argmax(valid_q_values).item()]

    def train(self):
        """训练网络"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([exp.state for exp in batch])
        actions = torch.LongTensor([exp.action for exp in batch])
        rewards = torch.FloatTensor([exp.reward for exp in batch])
        next_states = torch.FloatTensor([exp.next_state for exp in batch])
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q
        
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        """初始分配装载点"""
        self.init_networks(mine)  # 确保网络已初始化
        
        assert isinstance(truck.current_location, ChargingSite), "Truck must be at charging site for initial dispatch"
        
        # 更新其他卡车目标信息
        other_trucks = [t for t in mine.trucks if t.name != truck.name]
        target_locations = [t.target_location for t in other_trucks 
                          if t.target_location is not None and isinstance(t.target_location, LoadSite)]
        
        # 更新装载点的即将到达卡车数
        for load_site in mine.load_sites:
            load_site.coming_truck_num = sum(1 for loc in target_locations if loc.name == load_site.name)
        
        state = self.get_state(truck, mine)
        valid_actions = list(range(len(mine.load_sites)))
        action = self.select_action(state, valid_actions)
        
        self.last_state = state
        self.last_action = action
        
        return action

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        """选择卸载点"""
        self.init_networks(mine)
        
        assert isinstance(truck.current_location, LoadSite), "Truck must be at load site for haul dispatch"
        assert truck.truck_load > 0, "Truck must be loaded for haul dispatch"
        
        # 更新其他卡车目标信息
        other_trucks = [t for t in mine.trucks if t.name != truck.name]
        target_locations = [t.target_location for t in other_trucks 
                          if t.target_location is not None and isinstance(t.target_location, DumpSite)]
        
        # 更新卸载点的即将到达卡车数
        for dump_site in mine.dump_sites:
            dump_site.coming_truck_num = sum(1 for loc in target_locations if loc.name == dump_site.name)
        
        state = self.get_state(truck, mine)
        valid_actions = list(range(len(mine.dump_sites)))
        
        if hasattr(self, 'last_state'):
            reward = self.calculate_reward(truck, self.last_state, state, mine)
            self.memory.append(Experience(self.last_state, self.last_action, reward, state))
            
        action = self.select_action(state, valid_actions)
        
        self.last_state = state
        self.last_action = action
        
        self.train()
        
        return action

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        """返回装载点"""
        self.init_networks(mine)
        
        assert isinstance(truck.current_location, DumpSite), "Truck must be at dump site for back dispatch"
        assert truck.truck_load == 0, "Truck must be empty for back dispatch"
        
        # 更新其他卡车目标信息
        other_trucks = [t for t in mine.trucks if t.name != truck.name]
        target_locations = [t.target_location for t in other_trucks 
                          if t.target_location is not None and isinstance(t.target_location, LoadSite)]
        
        # 更新装载点的即将到达卡车数
        for load_site in mine.load_sites:
            load_site.coming_truck_num = sum(1 for loc in target_locations if loc.name == load_site.name)
        
        state = self.get_state(truck, mine)
        valid_actions = list(range(len(mine.load_sites)))
        
        if hasattr(self, 'last_state'):
            reward = self.calculate_reward(truck, self.last_state, state, mine)
            self.memory.append(Experience(self.last_state, self.last_action, reward, state))
            
        action = self.select_action(state, valid_actions)
        
        self.last_state = state
        self.last_action = action
        
        self.train()
        
        return action


if __name__ == "__main__":
    dispatcher = RLDispatcher()
    print(dispatcher.give_init_order(1,2))
    print(dispatcher.give_haul_order(1,2))
    print(dispatcher.give_back_order(1,2))
    print(dispatcher.total_order_count, dispatcher.init_order_count)