import numpy as np
import random

class Time_Sim:
    def __init__(self, mu=0.3) -> None:
        self.MU = mu # mean arrival rate
        # self.TAU = tau
        self.arrival_time = 0
    
    def ret_arr_time(self):
        u = np.random.uniform()
        self.arrival_time = self.arrival_time + (-np.log(1-u) / self.MU)
        arr_t = int(self.arrival_time * 10)
        return arr_t

    def ret_dep_time(self):
        # u = np.random.uniform()
        # dep_time = self.arrival_time + (-np.log(1-u) / self.TAU)
        # dep_t = int(dep_time * 10)
        # return dep_t
        return int(self.arrival_time * 10) + random.randint(500, 1000)
    
    def reset(self):
        self.arrival_time = 0

# time_sim = Time_Sim(mu=0.15)
# for x in range(100):
#     print(time_sim.ret_arr_time())
#     print(time_sim.ret_dep_time())
#     print('\n')