import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from kicker_pong.model_pg_agent import PGAgent
from kicker_pong.model_memory_agent import *
from kicker_pong.plot_reward import RewardPlot
import kicker_pong.control_environment as Env


def discount_rewards(rewards, gamma=0.98):
    discounted_returns = [0 for _ in rewards]
    discounted_returns[-1] = rewards[-1]
    for t in range(len(rewards)-2, -1, -1):  # iterate backwards
        discounted_returns[t] = rewards[t] + discounted_returns[t+1]*gamma
    return discounted_returns


def main():
    # Configure Settings
    total_episodes = 150000
    total_steps_max = 10000
    epsilon_stop = 100000
    train_frequency = 1
    plot_frequency = 10
    max_episode_length = 5000
    render_start = 1000
    should_render = False
    should_plot = True
    episode_finished = False

    explore_exploit_setting = 'epsilon_greedy_annealed_0.25->0.001'

    s_path_model = "/home/prock/models/reinforce_kicker_pong_v2_"
    s_path_reward = "/home/prock/data/reinforce_reward_train_data_v2.txt"
    s_path_batch_loss = "/home/prock/data/reinforce_batch_loss_train_data_v2.txt"

    env = Env.EnvironmentController()
    state_size = 24
    num_actions = 3

    solved = False

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        agent = PGAgent(session=session, state_size=state_size, num_actions=num_actions, hidden_size_1=300,
                        hidden_size_2=600, hidden_size_3=600, hidden_size_4=600, hidden_size_5=300,
                        learning_rate=1e-5, explore_exploit_setting=explore_exploit_setting)

        session.run(tf.global_variables_initializer())

        episode_rewards = []
        batch_losses = []

        plt.ion()
        reward_memory = RewardsMemory()
        reward_plt = RewardPlot()

        global_memory = Memory()
        steps = 0
        train_it = np.uint(0)
        plot_it = np.uint(0)
        i = np.uint64(0)

        try:
            for i in tqdm.tqdm(range(total_episodes)):

                state, _, _ = env.reset()
                episode_reward = 0.0
                episode_history = EpisodeHistory()
                epsilon_percentage = float(min(i/float(epsilon_stop), 1.0))
                for j in range(max_episode_length):
                    action = agent.predict_action(state, epsilon_percentage)

                    state_prime, reward, terminal = env.step(action)
                    state_prime = state_prime
                    if (render_start > 0 and i > render_start and should_render):  # or (solved and should_render):
                        env.render()
                    episode_history.add_to_history(state, action, reward, state_prime)
                    state = state_prime
                    episode_reward += reward

                    steps += 1
                    if terminal:
                        episode_history.discounted_returns = discount_rewards(episode_history.rewards, 0.99)
                        global_memory.add_episode(episode_history)
                        train_it += 1

                        i += 1
                        plot_it += 1
                        episode_rewards.append(episode_reward)
                        episode_reward = 0.0

                        if np.mod(train_it, train_frequency) == 0:
                            feed_dict = {agent.reward_input: np.array(global_memory.discounted_returns),
                                         agent.action_input: np.array(global_memory.actions),
                                         agent.state: np.array(global_memory.states)}
                            _, batch_loss = session.run([agent.train_step, agent.loss], feed_dict=feed_dict)
                            batch_losses.append(batch_loss)
                            global_memory.reset_memory()
                            train_it = 0

                            episode_finished = True

                        if i > plot_frequency + 1:
                            if np.mod(plot_it, plot_frequency) == 0 and should_plot:
                                reward_memory.rewards.append(np.mean(episode_rewards))
                                reward_memory.episodes.append(i)
                                reward_plt.update(reward_memory.episodes, reward_memory.rewards)

                        if episode_finished:
                            break

                if np.mod(i, 1000) == 0:
                    print('Mean Reward: ', np.mean(episode_rewards), '  Games: ', i)
                    print('Anzahl der gewonnen Spiele: ', env.get_goal_counter(), '/ 1000 Spielen')
                    if env.get_goal_counter() > 600.0:
                        solved = True
                        save_path = agent.saver.save(session, s_path_model + explore_exploit_setting + '.ckpt')
                        print("Model saved in path: %s" % save_path)
                    else:
                        solved = False
                    env.set_goal_counter(0)

        except KeyboardInterrupt:
            pass
        finally:
            print('Solved:', solved, '!\nMean Reward', np.mean(episode_rewards), '  Episodes: ', i)
            save_path = agent.saver.save(session, s_path_model + explore_exploit_setting + '.ckpt')
            print("Model saved in path: %s" % save_path)
            with open(s_path_reward, "w") as fp:
                fp.writelines('Trainingsdaten kicker_pong mit REINFORCE Algorithmus (Belohnung pro Episode) ' +
                              explore_exploit_setting + '\n')
                for k in episode_rewards:
                    fp.write("%s\n" % k)
            with open(s_path_batch_loss, "w") as fp:
                fp.writelines('Trainingsdaten kicker_pong mit REINFORCE Algorithmus (Kosten pro Episode) ' +
                              explore_exploit_setting + '\n')
                for k in batch_losses:
                    fp.write("%s\n" % k)


if __name__ == '__main__':
    main()
