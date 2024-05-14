import gym
import numpy as np
from gym import spaces
 
class InventoryEnvConfig(gym.Env):
    def __init__(self, config):
        super(InventoryEnvConfig, self).__init__()
        # Action and observation spaces
        self.action_space = spaces.Box(low=0, high=9,shape=(1,),dtype=np.int32) # ho 9 possibili azioni tra cui scegliere (da 0 a 8)
        observation_length = (config['m'] + config['L'] - 1) + 2
        self.observation_space = spaces.Box(low=0, high=100, shape=(observation_length,), dtype=np.float32)
        # all'interno della box c'è un vettore di dimensioni observation_length
        # ogni elemento nell'array è compreso tra low and high ed è di tipo dtype
        self.rewards_history = []  # Initialize a list to track rewards
       
        # Inventory management parameters from config
        self.m = config['m']
        self.L = config['L']
        self.p = config['p']
        self.price_transformed=0
        self.holding_transformed=0
        self.demand_not_satisfied=0
        self.perishability=0
        self.total_stock=0
        self.c = config['c']
        self.h = config['h']
        self.b = config['b']
        self.w = config['w']
        self.num_episodes= config['num_episodes']
        self.total_timesteps= config['total_timesteps']
        self.done=config['done']
       
        # Demand distribution parameters
        self.mean_demand = config['mean_demand']
        self.coef_of_var = config['coef_of_var']
        self.shape = 1 / (self.coef_of_var ** 2)
        self.scale = self.mean_demand / self.shape
 
        self.base_stock =config['best_base_stock']
       
        #seeds
        self.seed(42)
       
        # Ensure all other necessary initializations are performed here
        self.reset()
 
 
    def reset(self):
        self.stock = np.zeros(self.m + self.L - 1)
        self.stock_bs = np.zeros(self.m + self.L - 1)
        self.current_step = 0
        self.initial_demand = 0  # Initialize demands for observation
        self.rewards_history = []  # Clear rewards history for the new episode
        return self._next_observation()
 
    def _next_observation(self):
        obs = np.concatenate([
            self.stock,
            np.array([self.initial_demand, self.current_step])
        ])
        return obs
 
 
    def seed(self, seed=None):
        np.random.seed(42)
 
    def calculate_base_stock_action(self):
        return max(0, self.base_stock - self.total_stock) #l 'azioe che voglio e' quella per ripristinare lo stock in base stock fissato quindi devo agire ordinando la differenz
   
    def step(self, action):
        order_quantity=action
 
        # Generate total demand
        self.initial_demand = np.round(np.random.gamma(self.shape, self.scale)).astype(int)
        demand = self.initial_demand
   
        # Satisfy demand
        for i in range(min(len(self.stock), self.m)):
            if demand > 0:
                taken = min(self.stock[i], demand)
                self.stock[i] -= taken
                demand -= taken
        # Satisfy demand
        for i in range(min(len(self.stock_bs), self.m)):
            if demand > 0:
                taken = min(self.stock_bs[i], demand)
                self.stock_bs[i] -= taken
                demand -= taken
               
        # Calculate rewards and metrics
        lost_sales = max(0, demand)
        expired = self.stock[0]
 
        satisfied_demand=(self.initial_demand - lost_sales)
        self.total_stock = np.sum(self.stock[:self.m])  # Sum only the first m elements
        self.total_stock_bs = np.sum(self.stock_bs[:self.m])  # Sum only the first m elements
        reward = (self.p * (satisfied_demand)
                  - self.c * order_quantity - self.h * (self.total_stock)
                  - self.b * lost_sales
                  - self.w * (expired))
        reward /= 100.0  # Divide the reward by 100
        self.rewards_history.append(reward)  # Track the reward for each step
       
        self.total_stock=self.total_stock
 
        # Update stock for the next period
        self.stock = np.roll(self.stock, -1)
        self.stock_bs = np.roll(self.stock_bs, -1)
        self.stock[-1] = order_quantity  # Add new order at the end
        self.stock_bs[-1] =max(0, self.base_stock - self.total_stock)  # Add new order at the end
        self.price_transformed=(self.p * (self.initial_demand - lost_sales))
        self.holding_transformed=(self.h * (self.total_stock))
        self.demand_not_satisfied=(self.b * lost_sales)
        self.perishability=(self.w * (expired))#
 
       
       
 
       
        # Calculate metrics for green and yellow items
        info = {
            'stock': np.sum(self.stock[-self.m:]),
            'expired': expired,
            'lost_sales': lost_sales,
            'satisfied': satisfied_demand,
            'reward': reward,
            'rewards_std': np.std(self.rewards_history) if self.rewards_history else 0,
            'order_quantity': order_quantity, # Include order_quantity here,
            'base_stock_action': max(0, self.base_stock - self.total_stock),
            'stock_bs': np.sum(self.stock_bs[-self.m:])
            }
       
       
        self.current_step += 1
        done = self.current_step >= self.done # End episode after 100 steps or define your own condition
        return self._next_observation(), reward,  done, info
   
    def override_action_with_random(self):
        """
        This method selects a random action from the action space
        and applies it to the environment, bypassing the model's decision.
        This is intended for use during the warm-up period to encourage exploration.
        """
        random_action = self.action_space.sample()
        return self.step(random_action)
 
 
    def render(self, mode='console'):
        if mode == 'console':
            print(f"Step: {self.current_step}")
            print(f"Stock: {self.stock}")
            print(f"Initial Demand: {self.initial_demand}")
            # Include additional print statements as needed for debugging or information purposes.
 
# Registra l'ambiente personalizzato in Gym
 
gym.envs.register(
    id='Pippo-v0',
    entry_point='ppo_env:InventoryEnvConfig',  # Sostituisci 'your_module_name' con il nome del tuo modulo Python
)