
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