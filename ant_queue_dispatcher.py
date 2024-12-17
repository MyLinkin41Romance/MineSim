from __future__ import annotations
import numpy as np
import random
import json
import time

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.dump_site import DumpSite, Dumper
from openmines.src.charging_site import ChargingSite
from openmines.src.road import Road
from openmines.src.truck import Truck


class ImprovedAntColonyDispatcher(BaseDispatcher):
    def __init__(self, n_ants=20, n_iterations=50, alpha=1.0, beta=2.0, rho=0.1, q0=0.9,
                 distance_weight=0.4, queue_weight=0.3, productivity_weight=0.3):
        super().__init__()
        self.name = "ImprovedAntColonyDispatcher"
        
        # ACO基本参数
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta    # 启发式因子重要程度
        self.rho = rho      # 信息素蒸发率
        self.q0 = q0        # 状态转移规则参数
        
        # 评估权重
        self.distance_weight = distance_weight      # 距离权重
        self.queue_weight = queue_weight           # 队列权重
        self.productivity_weight = productivity_weight  # 生产力权重
        
        # 解决方案相关
        self.pheromone = None
        self.distance_matrix = None
        self.heuristic_info = None
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.group_solution = None
        
        # 其他参数
        
        self.load_sites = None
        self.dump_sites = None
        self.np_random = np.random.RandomState()

    def calculate_distance_matrix(self, mine: "Mine"):
        """计算距离矩阵"""
        n_loads = len(mine.load_sites)
        n_dumps = len(mine.dump_sites)
        distance_matrix = np.zeros((n_loads, n_dumps))
        
        # 计算装载点到卸载点的距离
        for i, load_site in enumerate(mine.load_sites):
            for j, dump_site in enumerate(mine.dump_sites):
                # 使用路网距离而不是直线距离
                if hasattr(mine, 'road') and mine.road is not None:
                    distance = mine.road.road_matrix[i][j]
                else:
                    distance = np.linalg.norm(np.array(load_site.position) - np.array(dump_site.position))
                distance_matrix[i][j] = distance
        
        return distance_matrix

    def calculate_heuristic_info(self, mine: "Mine", current_time: int):
        """计算启发式信息矩阵"""
        n_loads = len(mine.load_sites)
        n_dumps = len(mine.dump_sites)
        heuristic_matrix = np.zeros((n_loads, n_dumps))
        
        for i, load_site in enumerate(mine.load_sites):
            load_queue = self.calculate_queue_factor(load_site, mine, current_time)
            load_productivity = load_site.load_site_productivity
            
            for j, dump_site in enumerate(mine.dump_sites):
                dump_queue = self.calculate_queue_factor(dump_site, mine, current_time)
                distance = self.distance_matrix[i][j]
                
                # 综合考虑距离、队列和生产力
                heuristic_value = (
                    self.distance_weight * (1.0 / (1.0 + distance)) +
                    self.queue_weight * (load_queue + dump_queue) / 2 +
                    self.productivity_weight * (1.0 / (1.0 + 1.0/load_productivity))
                )
                heuristic_matrix[i][j] = heuristic_value
        
        return heuristic_matrix

    def calculate_queue_factor(self, site, mine: "Mine", current_time: int):
        """计算队列因素"""
        # 当前队列长度
        current_queue = site.parking_lot.queue_status["total"][current_time]
        
        # 计算即将到达的车辆数
        coming_trucks = 0
        for truck in mine.trucks:
            if truck.target_location == site:
                coming_trucks += 1
        
        # 获取等待时间估计
        if hasattr(site, 'estimated_queue_wait_time'):
            wait_time = site.estimated_queue_wait_time
        else:
            wait_time = 0
        
        # 综合评分
        total_load = current_queue + coming_trucks
        wait_factor = 1.0 / (1.0 + wait_time/60)  # 转换为分钟
        queue_factor = 1.0 / (1.0 + total_load)
        
        return (queue_factor + wait_factor) / 2

    def initialize_pheromone(self, mine: "Mine"):
        """初始化信息素矩阵"""
        n_loads = len(mine.load_sites)
        n_dumps = len(mine.dump_sites)
        
        # 使用启发式信息初始化信息素
        initial_pheromone = np.ones((n_loads, n_dumps)) * 0.1
        
        # 根据距离设置初始值
        for i in range(n_loads):
            for j in range(n_dumps):
                distance = self.distance_matrix[i][j]
                initial_pheromone[i][j] *= (1.0 / (1.0 + distance))
        
        return initial_pheromone

    def calculate_solution_quality(self, solution, mine: "Mine", current_time: int):
        """计算解决方案质量"""
        total_score = 0
        
        for load_site_name, info in solution.items():
            if info["dumpsite"] is None:
                continue
            
            load_site = next(ls for ls in mine.load_sites if ls.name == load_site_name)
            dump_site = info["dumpsite"]
            
            # 1. 距离评分
            load_index = mine.load_sites.index(load_site)
            dump_index = mine.dump_sites.index(dump_site)
            distance = self.distance_matrix[load_index][dump_index]
            distance_score = 1.0 / (1.0 + distance)
            
            # 2. 队列评分
            load_queue_score = self.calculate_queue_factor(load_site, mine, current_time)
            dump_queue_score = self.calculate_queue_factor(dump_site, mine, current_time)
            queue_score = (load_queue_score + dump_queue_score) / 2
            
            # 3. 生产力匹配评分
            trucks_assigned = len(info["trucks"])
            expected_trucks = int(len(mine.trucks) * 
                                (load_site.load_site_productivity / 
                                 sum(ls.load_site_productivity for ls in mine.load_sites)))
            productivity_score = 1.0 / (1.0 + abs(trucks_assigned - expected_trucks))
            
            # 4. 计算综合评分
            site_score = (
                self.distance_weight * distance_score +
                self.queue_weight * queue_score +
                self.productivity_weight * productivity_score
            )
            
            # 5. 根据分配的卡车数量加权
            total_score += site_score * trucks_assigned
            
        return total_score
    def construct_ant_solution(self, mine: "Mine", current_time: int):
        """构建单个蚂蚁的解决方案"""
        solution = {}
        available_trucks = mine.trucks.copy()
        
        # 初始化解决方案结构
        for load_site in mine.load_sites:
            solution[load_site.name] = {
                "trucks": [],
                "dumpsite": None
            }
        
        # 为每个装载点选择卸载点和分配卡车
        for load_site in mine.load_sites:
            load_index = mine.load_sites.index(load_site)
            
            # 选择卸载点
            dump_site = self.choose_dump_site(load_index, mine)
            solution[load_site.name]["dumpsite"] = dump_site
            
            # 分配卡车
            if available_trucks:
                # 基于生产力和队列状态计算需要的卡车数量
                productivity_ratio = load_site.load_site_productivity / sum(ls.load_site_productivity for ls in mine.load_sites)
                queue_factor = self.calculate_queue_factor(load_site, mine, current_time)
                
                # 调整卡车分配数量
                base_trucks_needed = int(len(mine.trucks) * productivity_ratio)
                queue_adjustment = int(base_trucks_needed * (1 - queue_factor))
                trucks_needed = max(1, base_trucks_needed - queue_adjustment)
                
                # 分配卡车
                if available_trucks and trucks_needed > 0:
                    assigned_trucks = random.sample(available_trucks, 
                                                 min(trucks_needed, len(available_trucks)))
                    solution[load_site.name]["trucks"].extend(assigned_trucks)
                    for truck in assigned_trucks:
                        available_trucks.remove(truck)
        
        return solution

    def choose_dump_site(self, load_index: int, mine: "Mine"):
        """选择卸载点"""
        if random.random() < self.q0:
            # 利用最优选择
            dump_probs = self.calculate_dump_probabilities(load_index, mine)
            return mine.dump_sites[np.argmax(dump_probs)]
        else:
            # 概率性选择
            dump_probs = self.calculate_dump_probabilities(load_index, mine)
            dump_indices = range(len(mine.dump_sites))
            chosen_index = random.choices(dump_indices, weights=dump_probs)[0]
            return mine.dump_sites[chosen_index]

    def calculate_dump_probabilities(self, load_index: int, mine: "Mine"):
        """计算选择各个卸载点的概率"""
        n_dumps = len(mine.dump_sites)
        probabilities = np.zeros(n_dumps)
        
        for j in range(n_dumps):
            # 信息素
            tau = self.pheromone[load_index][j]
            # 启发式信息
            eta = self.heuristic_info[load_index][j]
            # 计算概率
            probabilities[j] = (tau ** self.alpha) * (eta ** self.beta)
        
        # 归一化
        sum_prob = np.sum(probabilities)
        if sum_prob > 0:
            probabilities = probabilities / sum_prob
            
        return probabilities

    def update_pheromone(self, solutions_quality, mine: "Mine"):
        """更新信息素"""
        # 信息素蒸发
        self.pheromone *= (1 - self.rho)
        
        # 添加新的信息素
        for solution, quality in solutions_quality:
            delta = 1.0 / (1.0 + 1.0/quality)  # 转换质量为信息素增量
            
            for load_site_name, info in solution.items():
                if info["dumpsite"] is not None:
                    load_index = mine.load_sites.index(
                        next(ls for ls in mine.load_sites if ls.name == load_site_name))
                    dump_index = mine.dump_sites.index(info["dumpsite"])
                    
                    # 根据分配的卡车数量加权
                    truck_weight = len(info["trucks"]) / len(mine.trucks)
                    self.pheromone[load_index][dump_index] += delta * truck_weight

    def compute_solution(self, mine: "Mine"):
        """计算最优解决方案"""
        if self.group_solution is not None:
            return
            
        current_time = int(mine.env.now)
        self.load_sites = mine.load_sites
        self.dump_sites = mine.dump_sites
        
        # 初始化计算
        self.distance_matrix = self.calculate_distance_matrix(mine)
        self.heuristic_info = self.calculate_heuristic_info(mine, current_time)
        self.pheromone = self.initialize_pheromone(mine)
        
        best_solution = None
        best_quality = float('-inf')
        
        # 迭代优化
        for iteration in range(self.n_iterations):
            solutions_quality = []
            
            # 每只蚂蚁构建解决方案
            for ant in range(self.n_ants):
                solution = self.construct_ant_solution(mine, current_time)
                quality = self.calculate_solution_quality(solution, mine, current_time)
                solutions_quality.append((solution, quality))
                
                # 更新最优解
                if quality > best_quality:
                    best_quality = quality
                    best_solution = solution.copy()
            
            # 更新信息素
            self.update_pheromone(solutions_quality, mine)
            
            # 动态调整参数
            self.adjust_parameters(iteration)
        
        self.group_solution = best_solution
        self.best_solution = best_solution
        self.best_fitness = best_quality

    def adjust_parameters(self, iteration):
        """动态调整算法参数"""
        # 随着迭代进行，增加局部搜索倾向
        self.q0 = min(0.9, 0.5 + iteration / self.n_iterations * 0.4)
        
        # 调整信息素权重
        self.alpha = max(0.5, 1.0 - iteration / self.n_iterations * 0.5)


    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        """初始分配装载点"""

            
        # 计算优化解决方案
        if self.group_solution is None:
            self.compute_solution(mine)
            
        # 获取固定分配的装载点
        assigned_load_site = None
        for load_site_name, info in self.group_solution.items():
            if truck in info["trucks"]:
                assigned_load_site = next(ls for ls in mine.load_sites if ls.name == load_site_name)
                break
        
        if assigned_load_site:
            # 检查队列状态
            current_time = int(mine.env.now)
            queue_factor = self.calculate_queue_factor(assigned_load_site, mine, current_time)
            
            if queue_factor < 0.3:  # 如果队列太长，寻找替代装载点
                alternative_loads = []
                for load_site in mine.load_sites:
                    score = (
                        self.queue_weight * self.calculate_queue_factor(load_site, mine, current_time) +
                        self.productivity_weight * (load_site.load_site_productivity / 
                                                  sum(ls.load_site_productivity for ls in mine.load_sites))
                    )
                    alternative_loads.append((load_site, score))
                
                # 选择得分最高的替代装载点
                best_alternative = max(alternative_loads, key=lambda x: x[1])[0]
                return mine.load_sites.index(best_alternative)
                
            return mine.load_sites.index(assigned_load_site)
            
        return 0

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        """选择卸载点"""

            
        # 计算优化解决方案
        if self.group_solution is None:
            self.compute_solution(mine)
            
        current_location = truck.current_location
        assert isinstance(current_location, LoadSite), "current_location is not a LoadSite"
        
        # 获取固定分配的卸载点
        assigned_dump_site = None
        for load_site_name, info in self.group_solution.items():
            if truck in info["trucks"]:
                assigned_dump_site = info["dumpsite"]
                break
        
        if assigned_dump_site:
            current_time = int(mine.env.now)
            queue_factor = self.calculate_queue_factor(assigned_dump_site, mine, current_time)
            
            if queue_factor < 0.3:  # 如果队列太长，寻找替代卸载点
                alternative_dumps = []
                for dump_site in mine.dump_sites:
                    # 计算综合得分
                    distance = np.linalg.norm(
                        np.array(current_location.position) - np.array(dump_site.position))
                    queue_score = self.calculate_queue_factor(dump_site, mine, current_time)
                    
                    score = (
                        self.distance_weight * (1.0 / (1.0 + distance)) +
                        self.queue_weight * queue_score
                    )
                    alternative_dumps.append((dump_site, score))
                
                # 选择得分最高的替代卸载点
                best_alternative = max(alternative_dumps, key=lambda x: x[1])[0]
                return mine.dump_sites.index(best_alternative)
                
            return mine.dump_sites.index(assigned_dump_site)
            
        return 0

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        """返回装载点"""

            
        # 计算优化解决方案
        if self.group_solution is None:
            self.compute_solution(mine)
            
        # 获取固定分配的装载点
        assigned_load_site = None
        for load_site_name, info in self.group_solution.items():
            if truck in info["trucks"]:
                assigned_load_site = next(ls for ls in mine.load_sites if ls.name == load_site_name)
                break
        
        if assigned_load_site:
            current_time = int(mine.env.now)
            queue_factor = self.calculate_queue_factor(assigned_load_site, mine, current_time)
            
            if queue_factor < 0.3:  # 如果队列太长，寻找替代装载点
                alternative_loads = []
                for load_site in mine.load_sites:
                    queue_score = self.calculate_queue_factor(load_site, mine, current_time)
                    productivity_score = load_site.load_site_productivity / sum(
                        ls.load_site_productivity for ls in mine.load_sites)
                    
                    score = (
                        self.queue_weight * queue_score +
                        self.productivity_weight * productivity_score
                    )
                    alternative_loads.append((load_site, score))
                
                # 选择得分最高的替代装载点
                best_alternative = max(alternative_loads, key=lambda x: x[1])[0]
                return mine.load_sites.index(best_alternative)
                
            return mine.load_sites.index(assigned_load_site)
            
        return 0


# 测试代码
if __name__ == "__main__":
    # 创建调度器实例
    dispatcher = ImprovedAntColonyDispatcher(
        n_ants=30,
        n_iterations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q0=0.9,
        distance_weight=0.4,
        queue_weight=0.3,
        productivity_weight=0.3
    )

    # 加载配置文件
    config_file = "openmines/src/conf/north_pit_mine.json"
    
    def load_config(filename):
        with open(filename, 'r') as file:
            return json.load(file)

    config = load_config(config_file)
    mine = Mine(config_file)

    # 初始化充电站和卡车
    charging_site = ChargingSite(config['charging_site']['name'], 
                                position=config['charging_site']['position'])
    for truck_config in config['charging_site']['trucks']:
        for _ in range(truck_config['count']):
            truck = Truck(
                name=f"{truck_config['type']}{_ + 1}",
                truck_capacity=truck_config['capacity'],
                truck_speed=truck_config['speed']
            )
            charging_site.add_truck(truck)

    # 初始化装载点和铲车
    for load_site_config in config['load_sites']:
        load_site = LoadSite(name=load_site_config['name'], 
                            position=load_site_config['position'])
        for shovel_config in load_site_config['shovels']:
            shovel = Shovel(
                name=shovel_config['name'],
                shovel_tons=shovel_config['tons'],
                shovel_cycle_time=shovel_config['cycle_time'],
                position_offset=shovel_config['position_offset']
            )
            load_site.add_shovel(shovel)
        load_site.add_parkinglot(
            position_offset=load_site_config['parkinglot']['position_offset'],
            name=load_site_config['parkinglot']['name']
        )
        mine.add_load_site(load_site)

    # 初始化卸载点和卸载机
    for dump_site_config in config['dump_sites']:
        dump_site = DumpSite(dump_site_config['name'], 
                            position=dump_site_config['position'])
        for dumper_config in dump_site_config['dumpers']:
            for _ in range(dumper_config['count']):
                dumper = Dumper(
                    name=f"{dump_site_config['name']}-点位{_}",
                    dumper_cycle_time=dumper_config['cycle_time'],
                    position_offset=dumper_config['position_offset']
                )
                dump_site.add_dumper(dumper)
        dump_site.add_parkinglot(
            position_offset=dump_site_config['parkinglot']['position_offset'],
            name=dump_site_config['parkinglot']['name']
        )
        mine.add_dump_site(dump_site)

    # 初始化道路
    road_matrix = np.array(config['road']['road_matrix'])
    road_event_params = config['road'].get('road_event_params', {})
    charging_to_load_road_matrix = config['road']['charging_to_load_road_matrix']
    
    road = Road(
        road_matrix=road_matrix,
        charging_to_load_road_matrix=charging_to_load_road_matrix,
        road_event_params=road_event_params
    )

    # 添加系统组件
    mine.add_road(road)
    mine.add_charging_site(charging_site)
    mine.add_dispatcher(dispatcher)

    # 运行调度器并计算解决方案
    dispatcher.compute_solution(mine)

    # 打印结果
    print("\n===== 改进蚁群算法调度结果 =====")
    print("最优解:", dispatcher.best_solution)
    print("最优适应度值:", dispatcher.best_fitness)

    # 详细分析结果
    for load_site_name, info in dispatcher.best_solution.items():
        print(f"\n装载点 {load_site_name} 的分配情况:")
        print(f"分配的卡车数量: {len(info['trucks'])}")
        print(f"分配的卸载点: {info['dumpsite'].name if info['dumpsite'] else 'None'}")
        print("分配的卡车:", [truck.name for truck in info['trucks']])

    # 性能统计
    print("\n===== 性能统计 =====")
    print(f"总调度次数: {dispatcher.total_order_count}")
    print(f"初始调度次数: {dispatcher.init_order_count}")
