import gym
import numpy as np
from gym import spaces

class BaseStockConfig(gym.Env):
    def __init__(self, config):
        super(BaseStockConfig, self).__init__()
        # Action and observation spaces
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
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
        self.base_stock_level = None
        # Ensure all other necessary initializations are performed here
        self.reset()


    def reset(self):
        self.stock = np.zeros(self.m + self.L - 1)
        self.current_step = 0
        self.initial_demand = 0  # Initialize demands for observation
        self.base_stock_level = None
        #print("Starting a new episode...")
        return self._next_observation()

    def _next_observation(self):
        obs = np.concatenate([
            self.stock,
            np.array([self.initial_demand, self.current_step]) 
        ])
        return obs


    def seed(self, seed=None):
        np.random.seed(seed)
        
    
    """ def step(self, action):
        order_quantity =0
        #if self.current_step == 0: # Set base stock level based on the action
        self.base_stock_level = np.ceil(action[0] * 50).astype(int)
            #print(f"Base stock level set to: {self.base_stock_level} for this episode.")
            # Calculate current inventory level
        
        current_inventory_level = np.sum(self.stock) 
        # Calculate order quantity needed to reach base stock level
        order_quantity = max(0, self.base_stock_level - current_inventory_level)
        #print(f"Action taken: {action[0]:.2f}, Current base stock level: {self.base_stock_level}, Order Quantity: {order_quantity}")
        # Generate total demand
        demand = np.round(np.random.gamma(self.shape, self.scale)).astype(int)
        
    
        # Satisfy demand
        for i in range(min(len(self.stock), self.m)):
            if demand > 0:
                taken = min(self.stock[i], demand)
                self.stock[i] -= taken
                demand -= taken
 
        # Calculate rewards and metrics
        lost_sales = max(0, demand)
        expired = self.stock[0] 
        satisfied_demand=(self.demand - lost_sales) # lost_sales rappresentano le vendite non soddifatte, cioe' la domanda non soddisfatta  
        total_stock = np.sum(self.stock[:self.m])  # Sum only the first m elements
        reward = (self.p * (self.demand - lost_sales)
                  -self.c * order_quantity - self.h * (total_stock )
                  -self.b * lost_sales
                  -self.w * (expired))
        
        reward /= 1.0  # Divide the reward by 100
        self.rewards_history.append(reward)  # Track the reward for each step
        print(f"Step: {self.current_step}, Base Stock Level: {self.base_stock_level}, Order Quantity: {order_quantity}, Current Inventory Level: {current_inventory_level}, Demand: {demand}, Reward: {reward}")
        # Update stock for the next period
        self.stock = np.roll(self.stock, -1)
        self.stock[-1] = order_quantity  # Add new order at the end
                

        # Calculate metrics for items
        info = {'stock': np.sum(self.stock[:self.m]),  # Current stock
                'expired': expired,  # given
                'lost_sales': lost_sales,  # given
                'satisfied': satisfied_demand,  # You need to calculate this
                'base_stock_level': self.base_stock_level, # The base stock level value
                'reward': reward,  # Include current step's reward
                # Calculate and include the standard deviation of rewards up to the current step
                'rewards_std': np.std(self.rewards_history) if self.rewards_history else 0
                }
        #print(info)
        self.current_step += 1
        done = self.current_step >= 10 # End episode after 10 steps or define your own condition
        return self._next_observation(), reward, done, info """

    def Simulate_step(self):
            
        current_inventory_level = np.sum(self.stock) 
        # Calculate order quantity needed to reach base stock level
        order_quantity = max(0, self.base_stock_level - current_inventory_level)
        #print(f"Action taken: {action[0]:.2f}, Current base stock level: {self.base_stock_level}, Order Quantity: {order_quantity}")
        # Generate total demand
        self.initial_demand = np.round(np.random.gamma(self.shape, self.scale)).astype(int)
        print(f'demand: {self.initial_demand}')

        demand = self.initial_demand
    
        # Satisfy demand
        for i in range(min(len(self.stock), self.m)):
            if demand > 0:
                taken = min(self.stock[i], demand)
                self.stock[i] -= taken
                demand -= taken
                      
      
        # Calculate rewards and metrics
        lost_sales = max(0, demand)
        expired = self.stock[0] 
        satisfied_demand=(self.initial_demand - lost_sales)
        total_stock = np.sum(self.stock[:self.m])  # Sum only the first m elements
        reward = (self.p * (satisfied_demand)
                  -self.c * order_quantity - self.h * (total_stock) 
                  -self.b * lost_sales
                  -self.w * (expired))
        reward /= 100.0  # Divide the reward by 100
        self.rewards_history.append(reward)  # Track the reward for each step
        print(f"Step: {self.current_step}, Base Stock Level: {self.base_stock_level}, Order Quantity: {order_quantity}, Current Inventory Level: {current_inventory_level}, Demand: {self.initial_demand}, Reward: {reward}")
        print(f"Step: {self.current_step}, stock: {np.sum(self.stock[:self.m]) },lost: {lost_sales}, outdate: { expired}, sales {self.p * (satisfied_demand) }, Reward: {reward}")
        # Update stock for the next period
        self.stock = np.roll(self.stock, -1)
        self.stock[-1] = order_quantity  # Add new order at the end
                

        # Calculate metrics for items
        info = {'stock': np.sum(self.stock[:self.m]),  # Current stock
                'expired': expired,  # given
                'lost_sales': lost_sales,  # given
                'satisfied': satisfied_demand,  # You need to calculate this
                'base_stock_level': self.base_stock_level, # The base stock level value
                'reward': reward,  # Include current step's reward
                'reward_components': {
                'holding_cost':self.h *(total_stock),
                'lost_sales_cost': self.b * lost_sales,
                'expired_stock_cost': self.w * (expired),
                'Satisfied demand':(self.p * (satisfied_demand))},
                # Calculate and include the standard deviation of rewards up to the current step
                'rewards_std': np.std(self.rewards_history) if self.rewards_history else 0
        
                }
        #print(info)
        self.current_step += 1
        done = self.current_step >= 10 # End episode after 10 steps or define your own condition
        if done:
             self.current_step = 0 
        return self._next_observation(), reward, done, info



    def render(self, mode='console'):
        if mode == 'console':
            print(f"Step: {self.current_step}")
            #print(f"Stock: {self.stock}}")
            print(f"Initial Demand: {self.initial_demand}")
            # Include additional print statements as needed for debugging or information purposes.
