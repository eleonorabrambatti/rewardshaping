import gym
import numpy as np
from gym import spaces

class InventoryEnvGYConfig(gym.Env):
    def __init__(self, config):
        super(InventoryEnvGYConfig, self).__init__()
        # Action and observation spaces
        #self.action_space = spaces.Discrete(4) 
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([3]), dtype=np.float32)
        observation_length = (config['m'] + config['L'] - 1) + 2
        self.observation_space = spaces.Box(low=0, high=100, shape=(observation_length,), dtype=np.float32)
        self.rewards_history = []  # Initialize a list to track rewards
        
        # Inventory management parameters from config
        self.m = config['m']
        self.L = config['L']
        self.p = config['p']
        self.c = config['c']
        self.h = config['h']
        self.b = config['b']
        self.w = config['w']
        
        # Demand distribution parameters
        self.mean_demand = config['mean_demand']
        self.coef_of_var = config['coef_of_var']
        self.shape = 1 / (self.coef_of_var ** 2)
        self.scale = self.mean_demand / self.shape
        
        # Ensure all other necessary initializations are performed here
        self.reset()


    def reset(self):
        self.stock = np.zeros(self.m + self.L - 1)
        self.current_step = 0
        self.demand = 0  # Initialize demands for observation
        self.total_stock = 0  # Initialize total stock for observation
        self.rewards_history = []  # Clear rewards history for the new episode
        return self._next_observation()

    def _next_observation(self):
        obs = np.concatenate([
            self.stock,
            np.array([self.demand, self.current_step])
        ])
        return obs


    def seed(self, seed=None):
        np.random.seed(seed)
        
    
    def step(self, action):
        #order_quantity = action[0] 
        order_quantity = np.around(action).astype(int)
        # Generate total demand
        self.demand = np.round(np.random.gamma(self.shape, self.scale)).astype(int)

        if self.current_step < self.L:
            #if order_quantity != 3:
            #    print(f'order_quantity: {order_quantity}')
            self.demand = 0
        
        lost_demand = self.demand
        # Satisfy  demand
        for i in range(min(len(self.stock), self.m)):
            if lost_demand > 0:
                taken = min(self.stock[i], lost_demand)
                self.stock[i] -= taken
                lost_demand -= taken
                
      # Calculate rewards and metrics
        lost_sales = max(0, lost_demand)
        expired = self.stock[0] 
        satisfied_demand=(self.demand - lost_sales)
        self.total_stock = np.sum(self.stock[:self.m])  # Sum only the first m elements
        reward = (self.p * (self.demand - lost_sales) -
                  self.c * order_quantity - self.h * (self.total_stock) -
                  self.b * lost_sales -
                  self.w * (expired))
        
        reward /= 100.0  # Divide the reward by 100
        self.rewards_history.append(reward)  # Track the reward for each step
        
        # Update the stock for the next period
        self.stock = np.roll(self.stock, -1)
        self.stock[-1] = order_quantity  # Add new order at the end
        self.total_stock = np.sum(self.stock[:self.m])
        #print(f' ts dopo: {total_stock}')
        
        # Calculate metrics for the items
        info = {
            'demand': self.demand,
            'stock': np.sum(self.stock[:self.m]),
            'in_transit': np.sum(self.stock[-self.L:]),  # in transit
            'expired_items': expired,
            'lost_sales': lost_sales,
            'satisfied_demand': satisfied_demand,
            'orders': order_quantity,
            'reward': reward,  # Include current step's reward
            # Calculate and include the standard deviation of rewards up to the current step
            'rewards_std': np.std(self.rewards_history) if self.rewards_history else 0
        }
        #if self.current_step < 2:
        #    reward = 0 
        self.current_step += 1
        done = self.current_step >= 10  # End episode after 10 steps or define your own condition
        return self._next_observation(), reward, done, info
    

    def render(self, mode='console'):
        if mode == 'console':
            print(f"Step: {self.current_step}")
            print(f"Stock: {self.stock}")
            print(f"Initial Demand: {self.demand}")
            # Include additional print statements as needed for debugging or information purposes.
