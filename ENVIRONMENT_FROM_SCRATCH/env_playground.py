from supply_chain_env.demand import get_base_config
from supply_chain_env.supply_chain import SupplyChainEnvironment
import numpy as np

def main():
    env = SupplyChainEnvironment(config=get_base_config(1,"uniform"))
    env.reset()
 
    human_actions = [0] * env.get_config().get("max_time")
    for i in range(len(human_actions)):
        if i % 3 == 0:
            human_actions[i] = 3
 
    print(env) # SupplyChain inizializzato
    for a in human_actions:
        state, reward, done, _, _ = env.step(a)
        print(env, reward)
        print("action is " + str(a))
        print("state is " + str(state))
 
 
if __name__ == "__main__":
    main()