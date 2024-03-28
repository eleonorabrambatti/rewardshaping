from ENV_TRAIN import Retail_Environment
from UNSHAPED_DQN import DQN_Agent
#from SHAPED_B import DQN_Agent
#from SHAPED_BLE import DQN_Agent
from BASESTOCK_POLICY import DQN_Agent
from pandas import DataFrame
import pandas as pd
import os
import matplotlib.pyplot as plt

x = 0

#################################
MEAN_DEMAND = 4
CV = 0.5
LIFETIME = 3
LEADTIME = 2
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

EPOCHS = 40
BATCH_SIZE = 32
UPDATE = 20
#################################
FACTOR = 50
BASESTOCK = 0
S1 = 0
S2 = 0
#b = 0
b = 1

#env_train = Retail_Environment(LIFETIME, LEADTIME, MEAN_DEMAND, CV, MAX_ORDER, C_ORDER, C_PERISH, C_LOST, C_HOLD, FIFO, LIFO, TRAIN_TIME)
#state_size = len(env_train.state)
#action_size = len(env_train.action_space)

columns = EPOCHS
rows = 5
all_scores = {}

for lead_time in range(2, 3):  # Iterating through lead times from 1 to 2
    
    env_train = Retail_Environment(LIFETIME, lead_time, MEAN_DEMAND, CV, MAX_ORDER, C_ORDER, C_PERISH, C_LOST, C_HOLD, FIFO, LIFO, TRAIN_TIME)
    state_size = len(env_train.state)
    action_size = len(env_train.action_space)
    scores = [[0 for _ in range(columns)]]
    
    #for i in range(rows):
    print("New agent created...")
    i=1
    agent = DQN_Agent(state_size, action_size, GAMMA, EPSILON_DECAY, EPSILON_MIN, LEARNING_RATE, EPOCHS, env_train, BATCH_SIZE, UPDATE, i, x)
    scores = agent.train()
    
    #df = pd.DataFrame({'0': scores})
    all_scores[f'{lead_time}'] = pd.DataFrame(scores)  # Converte la lista di liste in un DataFrame


# Now you have all scores in the dictionary `all_scores`
# You can do further processing or save them as needed
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
