import tensorflow as tf
# import numpy as np
# import random
import pygame

from kicker_pong.model_pg_agent import PGAgent
import kicker_pong.control_environment as Env


def main():

    env = Env.EnvironmentController()
    state_size = 24
    num_actions = 3

    explore_exploit_setting = 'no_greedy'

    with tf.Session() as session:
        agent = PGAgent(session=session, state_size=state_size, num_actions=num_actions, hidden_size_1=300,
                        hidden_size_2=600, hidden_size_3=600, hidden_size_4=600, hidden_size_5=300,
                        learning_rate=1e-5, explore_exploit_setting=explore_exploit_setting)

        agent.session.run(tf.global_variables_initializer())

        agent.saver.restore(agent.session,
                            "/home/prock/models/___reinforce_kicker_pong_epsilon_greedy_annealed_0.25->0.001.ckpt")
        print("Model restored.")

        state, _, _ = env.reset()
        state = state
        print(state)

        clock = pygame.time.Clock()

        running = True
        while running:

            action = agent.predict_action(state, 1.0)

            state, _, terminal = env.step(action)

            if terminal:
                state = env.reset()

            state = state

            env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            clock.tick_busy_loop(30)


if __name__ == '__main__':
    main()