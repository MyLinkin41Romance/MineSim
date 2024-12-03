from __future__ import annotations

import json
import random
import numpy as np

from openmines.src.charging_site import ChargingSite
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.dump_site import DumpSite, Dumper
from openmines.src.road import Road
from openmines.src.truck import Truck

class AntColonyDispatcher(BaseDispatcher):
    def __init__(self, n_ants=20, n_iterations=50, alpha=1.0, beta=2.0, rho=0.1, q0=0.9):
        super().__init__()
        self.name = "AntColonyDispatcher"
        self.group_solution = None
        # ACO parameters
        self.n_ants = n_ants  # 蚂蚁数量
        self.n_iterations = n_iterations  # 迭代次数
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta  # 启发式因子重要程度
        self.rho = rho  # 信息素蒸发率
        self.q0 = q0  # 状态转移规则的参数
        self.pheromone = None  # 信息素矩阵
        self.best_solution = None
        self.best_fitness = float('inf')
        self.load_sites = None
        self.dump_sites = None

    def calculate_distance_matrix(self, mine: "Mine"):
        n_loads = len(mine.load_sites)
        n_dumps = len(mine.dump_sites)
        distance_matrix = np.zeros((n_loads, n_dumps))
        
        for i, load_site in enumerate(mine.load_sites):
            for j, dump_site in enumerate(mine.dump_sites):
                distance = np.linalg.norm(np.array(load_site.position) - np.array(dump_site.position))
                distance_matrix[i][j] = distance
                
        return distance_matrix

    def initialize_pheromone(self, mine: "Mine"):
        n_loads = len(mine.load_sites)
        n_dumps = len(mine.dump_sites)
        return np.ones((n_loads, n_dumps)) * 0.1

    def compute_solution(self, mine: "Mine"):
        if self.group_solution is not None:
            return
        
        self.load_sites = mine.load_sites
        self.dump_sites = mine.dump_sites

        distance_matrix = self.calculate_distance_matrix(mine)
        self.pheromone = self.initialize_pheromone(mine)
        
        load_sites = mine.load_sites
        dump_sites = mine.dump_sites
        trucks = mine.trucks

        # Calculate load site productivity
        for load_site in load_sites:
            load_site_productivity = sum(
                shovel.shovel_tons / shovel.shovel_cycle_time for shovel in load_site.shovel_list)
            load_site.load_site_productivity = load_site_productivity

        best_solution = None
        best_fitness = float('inf')

        for iteration in range(self.n_iterations):
            ant_solutions = []
            
            for ant in range(self.n_ants):
                solution = {}
                available_trucks = trucks.copy()
                
                # Initialize solution structure
                for load_site in load_sites:
                    solution[load_site.name] = {
                        "trucks": [],
                        "dumpsite": None
                    }

                # Assign trucks to load sites using ACO
                for load_site in load_sites:
                    dump_probabilities = self.calculate_probabilities(load_site, dump_sites, distance_matrix)
                    chosen_dump = self.choose_dump_site(dump_probabilities, dump_sites)
                    solution[load_site.name]["dumpsite"] = chosen_dump

                    # Assign trucks based on load site productivity
                    trucks_needed = int(len(available_trucks) * 
                                     (load_site.load_site_productivity / sum(ls.load_site_productivity for ls in load_sites)))
                    
                    if available_trucks and trucks_needed > 0:
                        assigned_trucks = random.sample(available_trucks, min(trucks_needed, len(available_trucks)))
                        solution[load_site.name]["trucks"].extend(assigned_trucks)
                        for truck in assigned_trucks:
                            available_trucks.remove(truck)

                # Evaluate solution
                fitness = self.evaluate_solution(solution, distance_matrix, load_sites)
                ant_solutions.append((solution, fitness))

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = solution

            # Update pheromone
            self.update_pheromone(ant_solutions, distance_matrix)

        self.group_solution = best_solution
        self.best_solution = best_solution
        self.best_fitness = best_fitness

    def calculate_probabilities(self, load_site, dump_sites, distance_matrix):
        load_index = next(i for i, ls in enumerate(self.load_sites) if ls.name == load_site.name)
        probabilities = np.zeros(len(dump_sites))
        
        for j in range(len(dump_sites)):
            tau = self.pheromone[load_index][j]
            eta = 1.0 / (distance_matrix[load_index][j] + 1e-10)  # 避免除零
            probabilities[j] = (tau ** self.alpha) * (eta ** self.beta)
            
        return probabilities / (np.sum(probabilities) + 1e-10)

    def choose_dump_site(self, probabilities, dump_sites):
        if random.random() < self.q0:
            return dump_sites[np.argmax(probabilities)]
        else:
            return random.choices(dump_sites, weights=probabilities)[0]

    def evaluate_solution(self, solution, distance_matrix, load_sites):
        total_distance = 0
        for load_site in load_sites:
            if solution[load_site.name]["dumpsite"] is not None:
                load_index = next(i for i, ls in enumerate(load_sites) if ls.name == load_site.name)
                dump_index = next(i for i, ds in enumerate(self.dump_sites) 
                                if ds.name == solution[load_site.name]["dumpsite"].name)
                total_distance += distance_matrix[load_index][dump_index] * len(solution[load_site.name]["trucks"])
        return total_distance

    def update_pheromone(self, ant_solutions, distance_matrix):
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Add new pheromone
        for solution, fitness in ant_solutions:
            delta = 1.0 / (fitness + 1e-10)
            for load_site_name, info in solution.items():
                if info["dumpsite"] is not None:
                    load_index = next(i for i, ls in enumerate(self.load_sites) 
                                    if ls.name == load_site_name)
                    dump_index = next(i for i, ds in enumerate(self.dump_sites) 
                                    if ds.name == info["dumpsite"].name)
                    self.pheromone[load_index][dump_index] += delta

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        if self.group_solution is None:
            self.compute_solution(mine)

        for load_site_name, info in self.group_solution.items():
            if truck in info["trucks"]:
                return mine.load_sites.index([ls for ls in mine.load_sites if ls.name == load_site_name][0])
        return 0

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        if self.group_solution is None:
            self.compute_solution(mine)

        current_location = truck.current_location
        assert isinstance(current_location, LoadSite), "current_location is not a LoadSite"
        
        # Find the assigned dump site from the solution
        for load_site_name, info in self.group_solution.items():
            if truck in info["trucks"]:
                assigned_dump_site = info["dumpsite"]
                return mine.dump_sites.index(assigned_dump_site)
        return 0

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        if self.group_solution is None:
            self.compute_solution(mine)

        for load_site_name, info in self.group_solution.items():
            if truck in info["trucks"]:
                return mine.load_sites.index([ls for ls in mine.load_sites if ls.name == load_site_name][0])
        return 0

# Test code remains the same as in the original dispatcher
if __name__ == "__main__":
    dispatcher = AntColonyDispatcher()
    config_file = r"D:\git clone\openmines\openmines\src\conf\north_pit_mine.json"
    from openmines.src.mine import Mine

    def load_config(filename):
        with open(filename, 'r') as file:
            return json.load(file)

    config = load_config(config_file)
    mine = Mine(config_file)
    
# Test code continuation...

    # 初始化充电站和卡车
    charging_site = ChargingSite(config['charging_site']['name'], position=config['charging_site']['position'])
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
        load_site = LoadSite(name=load_site_config['name'], position=load_site_config['position'])
        for shovel_config in load_site_config['shovels']:
            shovel = Shovel(
                name=shovel_config['name'],
                shovel_tons=shovel_config['tons'],
                shovel_cycle_time=shovel_config['cycle_time'],
                position_offset=shovel_config['position_offset']
            )
            load_site.add_shovel(shovel)
        load_site.add_parkinglot(position_offset=load_site_config['parkinglot']['position_offset'],
                                name=load_site_config['parkinglot']['name'])
        mine.add_load_site(load_site)

    # 初始化卸载点和卸载机
    for dump_site_config in config['dump_sites']:
        dump_site = DumpSite(dump_site_config['name'], position=dump_site_config['position'])
        for dumper_config in dump_site_config['dumpers']:
            for _ in range(dumper_config['count']):
                dumper = Dumper(
                    name=f"{dump_site_config['name']}-点位{_}",
                    dumper_cycle_time=dumper_config['cycle_time'],
                    position_offset=dumper_config['position_offset']
                )
                dump_site.add_dumper(dumper)
        dump_site.add_parkinglot(position_offset=dump_site_config['parkinglot']['position_offset'],
                                name=dump_site_config['parkinglot']['name'])
        mine.add_dump_site(dump_site)

    # 初始化道路
    road_matrix = np.array(config['road']['road_matrix'])
    road_event_params = config['road'].get('road_event_params', {})

    charging_to_load_road_matrix = config['road']['charging_to_load_road_matrix']
    road = Road(road_matrix=road_matrix, 
                charging_to_load_road_matrix=charging_to_load_road_matrix,
                road_event_params=road_event_params)
    
    # 添加充电站和装载区卸载区
    mine.add_road(road)
    mine.add_charging_site(charging_site)
    
    # 添加调度器并计算解决方案
    mine.add_dispatcher(dispatcher)
    dispatcher.compute_solution(mine)
    
    # 打印结果
    print("蚁群算法调度结果:")
    print("最优解:", dispatcher.group_solution)
    print("最优适应度值:", dispatcher.best_fitness)
    
    # 可以添加更详细的结果分析
    for load_site_name, info in dispatcher.group_solution.items():
        print(f"\n装载点 {load_site_name}:")
        print(f"分配的卡车数量: {len(info['trucks'])}")
        print(f"分配的卸载点: {info['dumpsite'].name if info['dumpsite'] else 'None'}")
        print("分配的卡车:", [truck.name for truck in info['trucks']])