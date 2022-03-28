'''
main script for training a Deep Q-Leanring RL Agent on the LunarLander problem from OpenAI's Gym.
Trains the agent for a number of rollouts. Then evaluates final performance and plots performance.
Finally, a number of rollouts are played while rendering the environment.
'''
import matplotlib.pyplot as plt

from agent import Agent

agent = Agent()
rewards, epsilons = agent.train(10000)
eval_rewards = agent.evaluate(100)

plt.figure()
plt.subplot(311)
plt.plot(range(len(rewards)), rewards)
plt.subplot(312)
plt.plot(range(len(epsilons)), epsilons)
plt.subplot(313)
plt.plot(range(len(eval_rewards)), eval_rewards)
plt.show()

for _ in range(10):
    agent.rollout(draw=True)
agent.env.close()
