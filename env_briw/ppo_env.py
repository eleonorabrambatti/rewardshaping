import gym
import numpy as np
from gym import spaces

class InventoryEnvGYConfig(gym.Env):
    def __init__(self, config):
        #super(InventoryEnvGYConfig, self).__init__()
        # Action and observation spaces
        self.action_space = spaces.Discrete(30) # ho 30 possibili azioni tra cui scegliere (da 0 a 29)
        observation_length = (config['m'] + config['L'] - 1)+2 #+ config['m'] + 3
        self.observation_space = spaces.Box(low=0, high=100, shape=(observation_length,), dtype=np.float32)
        # all'interno della box c'è un vettore di dimensioni observation_length
        # ogni elemento nell'array è compreso tra low and high ed è di tipo dtype
        self.rewards_history = []  # Initialize a list to track rewards
        
        # Inventory management parameters from config
        self.m = config['m']
        self.L = config['L']
        self.Alpha = config['Alpha']
        self.p1 = config['p1']
        self.price_transformed=0
        self.holding_transformed=0
        self.demand_not_satisfied=0
        self.perishability=0
        self.total_stock_green=0
        self.p2 = config['p2']
        self.c = config['c']
        self.h = config['h']
        self.b1 = config['b1']
        #self.b2 = config['b2']
        self.w = config['w']
        self.beta = config['beta']
        self.num_episodes= config['num_episodes']
        self.total_timesteps= config['total_timesteps']
        
        # Demand distribution parameters
        self.mean_demand = config['mean_demand']
        self.coef_of_var = config['coef_of_var']
        self.shape = 1 / (self.coef_of_var ** 2)
        self.scale = self.mean_demand / self.shape
        self.done=config['done']

        #seeds
        self.seed(42)
        
        # Ensure all other necessary initializations are performed here
        self.reset()


    def reset(self):
        self.green_stock = np.zeros(self.m + self.L - 1)
        #self.yellow_stock = np.zeros(self.m)
        self.current_step = 0
        self.initial_green_demand = 0  # Initialize demands for observation
        #self.initial_yellow_demand = 0
        self.rewards_history = []  # Clear rewards history for the new episode
        return self._next_observation()

    def _next_observation(self):
        obs = np.concatenate([
            self.green_stock,
            #self.yellow_stock,
            np.array([self.initial_green_demand, self.current_step]) #self.initial_yellow_demand, self.current_step])
        ])
        return obs


    def seed(self, seed=None):
        np.random.seed(42)
        
    
    def step(self, action):
        #order_quantity = action[0] 
        order_quantity=action
        #order_quantity = np.ceil(action).astype(int)
        #○print(f"Order Quantity: {order_quantity}")
        # Generate total demand
        
        total_demand = np.round(np.random.gamma(self.shape, self.scale)).astype(int)
        # Split the demand into green and yellow based on the attractiveness factor
        self.initial_green_demand = np.round(total_demand * (1 - self.beta)).astype(int)
        #self.initial_yellow_demand = total_demand - self.initial_green_demand
    
        
        green_demand = self.initial_green_demand
        #yellow_demand = self.initial_yellow_demand
        #yellow_stock_increase = np.zeros(self.m)
    
        # Satisfy green demand
        for i in range(min(len(self.green_stock), self.m)):
            if green_demand > 0:
                taken = min(self.green_stock[i], green_demand)
                self.green_stock[i] -= taken
                green_demand -= taken
                
        # Satisfy yellow demand
        """ for i in range(len(self.yellow_stock)):
            if yellow_demand > 0:
                taken = min(self.yellow_stock[i], yellow_demand)
                self.yellow_stock[i] -= taken
                yellow_demand -= taken """
        
                
        # Calculate rewards and metrics
        lost_sales_green = max(0, green_demand)
        #lost_sales_yellow = max(0, yellow_demand)
        expired_green = self.green_stock[0] 
        #expired_yellow = self.yellow_stock[0]  
        satisfied_green_demand=(self.initial_green_demand - lost_sales_green)
        #satisfied_yellow_demand=(self.initial_yellow_demand - lost_sales_yellow)
        total_stock_green = np.sum(self.green_stock[:self.m])  # Sum only the first m elements
        #total_stock_yellow = np.sum(self.yellow_stock)
        reward = (self.p1 * (self.initial_green_demand - lost_sales_green) 
                  #self.p2 * (self.initial_yellow_demand - lost_sales_yellow) -
                  - self.c * order_quantity - self.h * (total_stock_green) #+ total_stock_yellow) 
                  - self.b1 * lost_sales_green #- self.b2 * lost_sales_yellow -
                  - self.w * (expired_green)) #+ expired_yellow))
        
        reward /= 100.0  # Divide the reward by 100
        self.rewards_history.append(reward)  # Track the reward for each step
        self.total_stock_green=total_stock_green
        # Update green and yellow stocks for the next period
        self.green_stock = np.roll(self.green_stock, -1)
        self.green_stock[-1] = order_quantity  # Add new order at the end
        self.price_transformed=(self.p1 * (self.initial_green_demand - lost_sales_green))
        self.holding_transformed=(self.h * (total_stock_green))
        #self.total_stock_green=total_stock_green
        self.demand_not_satisfied=(self.b1 * lost_sales_green)
        self.perishability=(self.w * (expired_green))
        
        # Apply deterioration from green to yellow stock
        #yellow_stock_increase = self.green_stock[:self.m] * self.Alpha
        #self.green_stock[:self.m] -=  yellow_stock_increase
        
        
        
        # Age the yellow stock
        #self.yellow_stock = np.roll(self.yellow_stock, -1)
        
        # Since self.yellow_stock was initially m-1 elements and rolled, the last element is effectively "empty"
        # Before addition, ensure self.yellow_stock is prepared to receive the last element of increase
         # Expand self.yellow_stock to m elements
        #if len(self.yellow_stock) < len(yellow_stock_increase):
        #    self.yellow_stock = np.append(self.yellow_stock, 0)  
        
        # Now add yellow_stock_increase to self.yellow_stock, including the last element
        #self.yellow_stock += yellow_stock_increase
        
        
        # Calculate metrics for green and yellow items
        info = {
    'stock_green': np.sum(self.green_stock[-self.m:]),
    #'stock_yellow': np.sum(self.yellow_stock),
    'expired_green': expired_green,
    #'expired_yellow': expired_yellow,
    'lost_sales_green': lost_sales_green,
    #'lost_sales_yellow': lost_sales_yellow,
    'satisfied_green': satisfied_green_demand,
    #'satisfied_yellow': satisfied_yellow_demand,
    'reward': reward,
    'rewards_std': np.std(self.rewards_history) if self.rewards_history else 0,
    'order_quantity': order_quantity # Include order_quantity here,

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
            print(f"Green Stock: {self.green_stock}")
            print(f"Initial Green Demand: {self.initial_green_demand}")
            # Include additional print statements as needed for debugging or information purposes.
