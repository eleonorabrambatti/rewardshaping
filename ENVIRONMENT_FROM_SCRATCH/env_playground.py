from supply_chain_env.demand import get_base_config
from supply_chain_env.supply_chain import SupplyChainEnvironment
import numpy as np
from supply_chain_env.algorithms import DQN_Agent
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

x = 0

#################################
MEAN_DEMAND = 4
CV = 0.5
LIFETIME = 12
LEADTIME = 12
C_LOST = 20
C_HOLD = 0.5
C_PERISH = 2
C_ORDER = 0
FIFO = True	
LIFO = False
#################################
MAX_ORDER = 50
TRAIN_TIME = 30
#################################
GAMMA = 0.99
EPSILON_DECAY = 0.997
PSI_DECAY = 0.99 # non usato
EPSILON_MIN = 0.01
LEARNING_RATE = 0.001

EPOCHS = 20
BATCH_SIZE = 32
UPDATE = 20
#################################
FACTOR = 50
BASESTOCK = 0
S1 = 0
S2 = 0
#b = 0
b = 1

all_scores = {}
columns = EPOCHS
rows = 5
def main():
    
    env = SupplyChainEnvironment(config=get_base_config(1,"uniform"))
    env.reset()
 
    """ human_actions = [0] * env.get_config().get("max_time")
    for i in range(len(human_actions)):
        if i % 3 == 0:
            human_actions[i] = 3 """
    
    #env_train = Retail_Environment(LIFETIME, lead_time, MEAN_DEMAND, CV, MAX_ORDER, C_ORDER, C_PERISH, C_LOST, C_HOLD, FIFO, LIFO, TRAIN_TIME)
    #if isinstance(env.observation_space, gym.spaces.Box):
    state_size = 25
    #elif isinstance(env.observation_space, gym.spaces.Discrete):
    #state_size = len(env.observation_space.n

    #if isinstance(env.action_space, gym.spaces.Box):
    action_size = 1
    #elif isinstance(env.action_space, gym.spaces.Discrete):
        #action_size = env.action_space.n

    scores = [[0 for _ in range(columns)]]
    
    #for i in range(rows):
    print("New agent created...")
    i=1
    agent = DQN_Agent(state_size, action_size, GAMMA, EPSILON_DECAY, EPSILON_MIN, LEARNING_RATE, EPOCHS, env, BATCH_SIZE, UPDATE, i, x)
    scores = agent.train()
    
    #df = pd.DataFrame({'0': scores})
    all_scores[f'{LEADTIME}'] = pd.DataFrame(scores)  # Converte la lista di liste in un DataFrame
    print(all_scores)
    #print(env) # SupplyChain inizializzato
    #for a in human_actions:
    #    state, reward, done, _, _ = env.step(a)
    #    print(env, reward)
    #    print("action is " + str(a))
    #    print("state is " + str(state))
for lead_time, scores_df in all_scores.items():
    print(f"Lead Time: {lead_time}")
    print(scores_df)
    
    # Creazione del grafico
    plt.figure(figsize=(10, 5))
    plt.plot(scores_df.index, scores_df[0], label=f'Lead Time {lead_time}')  # Plotting scores
    plt.title(f'Scores per Lead Time {lead_time}')
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid(True)
    plt.show()

 
 
if __name__ == "__main__":
    main()