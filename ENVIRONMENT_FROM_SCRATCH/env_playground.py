from supply_chain_env.base_demand import get_base_config
from supply_chain_env.supply_chain import SupplyChainEnvironment

 
def main():
    env = SupplyChainEnvironment(config=get_base_config(1))
    env.reset()
 
    human_actions = [0] * env.get_config().get("max_time")
    for i in range(len(human_actions)):
        if i % 3 == 0:
            human_actions[i] = 3
 
    print(env) # SupplyChain inizializzato
    print(env.get_state()) # e' lo stato di partenza
    for a in human_actions:
        state, reward, done, _, _ = env.step(a)
        print(env, reward)
        print("action is " + str(a))
        print("state is " + str(env.get_state()))
 
 
if __name__ == "__main__":
    main()