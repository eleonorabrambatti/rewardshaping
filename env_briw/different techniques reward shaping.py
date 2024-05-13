
# TECHNIQUES APPLIED TO THE REWARD SHAPING

#Size of the action space fixed at 9,  k=50, gamma=0.99, (rewards - avg reward of episodes BS) + (avg_rewards_10- avg reward of episodes BS)

"""         new_obs, rewards, dones, infos = env.step(clipped_actions)

            #rewards_bs = evaluate_base_stock_performance(env_2, 16, 1000)

            #base_stock_action = infos[0]['base_stock_action']
            
            #print(f'action ppo: {clipped_actions}, action bs: {base_stock_action}')
            cur_val = -50 * abs(rewards - object_bs)
            F = cur_val - ((1/0.99)*prev_val)
            prev_val = cur_val
            rewards_10=rewards
            rewards = rewards + F
            
            reward_accumulator += np.sum(rewards_10)
            if num_accumulated_steps % 10 == 0 and num_accumulated_steps > 0:
                avg_reward = reward_accumulator / 10
                cur_val = -50 * abs(avg_reward - object_bs)
                F = cur_val - ((1/0.99) * prev_val)
                prev_val = cur_val
                reward_accumulator = 0
                num_accumulated_steps = 0    
                rewards_10 += F
                rewards=rewards_10

            num_accumulated_steps += env.num_envs
            self.num_timesteps += env.num_envs """

#Size of the action space fixed at 9,  k=50, gamma=0.99, (avg reward of prev episode PPO - avg reward of episodes BS)

""" new_obs, rewards, dones, infos = env.step(clipped_actions)

            #rewards_bs = evaluate_base_stock_performance(env_2, 16, 1000)

            #base_stock_action = infos[0]['base_stock_action']
            
            #print(f'action ppo: {clipped_actions}, action bs: {base_stock_action}')
            
            reward_accumulator += np.sum(rewards)
            if num_accumulated_steps % 10 == 0 and num_accumulated_steps > 0:
                avg_reward = reward_accumulator / 10
                cur_val = -50 * abs(avg_reward - object_bs)
                F = cur_val - ((1/0.99) * prev_val)
                prev_val = cur_val
                reward_accumulator = 0
                num_accumulated_steps = 0    
            rewards += F
                #rewards=rewards_10

            num_accumulated_steps += env.num_envs
            self.num_timesteps += env.num_envs
            callback.update_locals(locals()) """

#Size of the action space fixed at 9, k=variable, gamma= 0.99, (total stock ppo-total stock BS)


"""         new_obs, rewards, dones, infos = env.step(clipped_actions)

            #rewards_bs = evaluate_base_stock_performance(env_2, 16, 1000)

            #base_stock_action = infos[0]['base_stock_action']

            stock_bs = infos[0]['stock_bs']
            
            #print(f'action ppo: {clipped_actions}, action bs: {base_stock_action}')
            
            #reward_accumulator += np.sum(rewards)
            #if num_accumulated_steps % 10 == 0 and num_accumulated_steps > 0:
                #avg_reward = reward_accumulator / 10
            coefficient=0
            if self.steps_since_update == 1000:
                self.steps_since_update = 0
                coefficient = np.exp(-0.001 * (self.num_timesteps / 1000))
                print(coefficient)

            cur_val = -coefficient * abs(infos[0]['stock'] - stock_bs)
            #cur_val = -50 * abs(infos[0]['stock']- stock_bs)
            F = cur_val - ((1/0.99) * prev_val)
            prev_val = cur_val
            #reward_accumulator = 0
            #num_accumulated_steps = 0    
            rewards += F
            #rewards=rewards_10 
            
for this technique the step function has also been modified:
    
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
            }"""

#Size of the action space fixed at 9, k=10, gamma= 0.99, (total stock ppo-total stock BS)
"""             new_obs, rewards, dones, infos = env.step(clipped_actions)

            #rewards_bs = evaluate_base_stock_performance(env_2, 16, 1000)

            #base_stock_action = infos[0]['base_stock_action']

            stock_bs = infos[0]['stock_bs']
            
            #print(f'action ppo: {clipped_actions}, action bs: {base_stock_action}')
            
            #reward_accumulator += np.sum(rewards)
            #if num_accumulated_steps % 10 == 0 and num_accumulated_steps > 0:
                #avg_reward = reward_accumulator / 10
            coefficient=0
            coefficient = 50 - (30 / 1000) * self.steps_since_update  # Lineare da 50 a 20 in 1000 passi
            if coefficient < 20:
                coefficient = 20

            #coefficient=0
            #if self.steps_since_update == 1000:
            #    self.steps_since_update = 0
            #    coefficient = np.exp(-0.001 * (self.num_timesteps / 1000))
            #    print(coefficient)

            cur_val = -coefficient * abs(infos[0]['stock'] - stock_bs)
            #cur_val = -50 * abs(infos[0]['stock']- stock_bs)
            F = cur_val - ((1/0.99) * prev_val)
            prev_val = cur_val
            #reward_accumulator = 0
            #num_accumulated_steps = 0    
            rewards += F """

