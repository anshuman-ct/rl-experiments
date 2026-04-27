from env.game_2048 import Env2048
from agent.dqn_agent import DQNAgent
from utils.encoding import encode_state
import config

def train():
    env = Env2048()
    agent = DQNAgent()

    for episode in range(config.NUM_EPISODES):
        state = encode_state(env.reset())
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)

            next_state_raw, reward, done = env.step(action)
            next_state = encode_state(next_state_raw)

            agent.memory.push(state, action, reward, next_state, done)
            agent.optimize()

            state = next_state
            total_reward += reward

            if agent.steps_done % config.TARGET_UPDATE == 0:
                agent.update_target()

        print(f"Episode {episode} | Reward: {total_reward}")