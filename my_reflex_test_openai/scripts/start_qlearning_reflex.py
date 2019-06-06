#!/usr/bin/env python2

import sys
# if sys.version_info[0] < 3:
#         raise Exception("Must be using Python 3 on ROS")

import gym
import numpy
import time
# import qlearn
import random
from gym import wrappers
from gym.envs.registration import register
# ROS packages required
import rospy
import rospkg
# import our training environment
# from openai_ros.task_envs.iiwa_tasks import iiwa_move
# from openai_ros.task_envs.hopper import hopper_stay_up
# import pickle, os

# from baselines import PPO2
# from run_algo import start_learning
import subprocess

# For the launch 
import roslaunch
import os
import git
import sys

def random_action():
    a = []
    for i in range(0, 6):
        a.append(random.uniform(-0.1, 0.1))
        # i = i+1
    a[2]=-0.05
    return a


if __name__ == '__main__':
        # Parameters
        timestep_limit_per_episode = 10000
        max_episode = 10

        # Begining of the script
        print("Python version: ", sys.version)
        rospy.init_node('start_qlearning_reflex', anonymous=True, log_level=rospy.WARN)
    
        print("Start node: start_qlearning_reflex file: start_qlearning_reflex.py")
        # print ("BIMM")

        # print("BIMM 2")
        # Cheating with the registration 
        # (shortcut: look at openai_ros_common.py and task_envs_list.py)
        register(
            id="iiwaMoveEnv-v0",
            entry_point='openai_ros.task_envs.iiwa_tasks.iiwa_move:iiwaMoveEnv',
        #     timestep_limit=timestep_limit_per_episode, #old one...
            max_episode_steps=timestep_limit_per_episode,
        )

        # Create the Gym environment
        env = gym.make('iiwaMoveEnv-v0')
        # rospy.loginfo("Gym environment done")
        print("Gym environment done")

        # print("Before reset the env" )
        # env.reset()
        # print("After reset the env" )

        # print("Action 1" )
        # # action = [-0.1, -0.1, -0.1, 1.3, 0.1, 0.5]
        # action = [-0.1, -0.1, -0.1, 0.0, 0.0, 0.0]
        # observation, reward, done, info = env.step(action)
        # print("*********************************************")
        # print("Observation: ", observation)
        # print("Reward: ", reward)
        # print("Done: ", done)
        # print("Info: ", info)
        # print("Action: ",  action)
        # print("*********************************************")
        # # print("Set Action: " + str(action))
        # # env.step(action)


        # rospy.sleep(10.0)

        # print("Action 2" )
        # # action2 = [0.2, 0.2, 0.2, -1.3, -0.1, -0.5]
        # action2 = [0.3, -0.1, -0.2, 0.0, 0.0, 0.0]
        # observation, reward, done, info = env.step(action2)
        # print("*********************************************")
        # print("Observation: ", observation)
        # print("Reward: ", reward)
        # print("Done: ", done)
        # print("Info: ", info)
        # print("Action: ",  action2)
        # print("*********************************************")
        # # print("Set Action: " + str(action2))
        # # env.step(action2)

        # print("Action are sent")


        # print("Before reset the env" )
        # env.reset()
        # print("After reset the env" )
        
        
        
        # agent = DQNRobotSolver(environment_name,
        #                         n_observations,
        #                         n_actions,
        #                         n_win_ticks,
        #                         min_episodes,
        #                         max_env_steps,
        #                         gamma,
        #                         epsilon,
        #                         epsilon_min,
        #                         epsilon_log_decay,
        #                         alpha,
        #                         alpha_decay,
        #                         batch_size,
        #                         monitor,
        #                         quiet)
        # agent.run(num_episodes=n_episodes_training, do_train=True)
        
        

        # Define and train a model in one line of code !
        # trained_model = PPO2('MlpPolicy', 'CartPole-v1').learn(total_timesteps=10000)
        # you can then access the gym env using trained_model.get_env()
        
        
        
        
        # env._set_action(action)
        # # Set the logging system
        # rospack = rospkg.RosPack()
        # pkg_path = rospack.get_path('my_hopper_openai_example')
        # outdir = pkg_path + '/training_results'
        # env = wrappers.Monitor(env, outdir, force=True)
        # rospy.loginfo("Monitor Wrapper started")


        #Where I thest EVERYTHING
        # observation, reward, done, info
        # for i in range(0, 9):
        #     # raw_input("Press Enter to continue...")
        #     a=random_action()
        #     # env.step(a)
        #     observation, reward, done, info = env.step(a)
        #     print("*********************************************")
        #     print("Observation: ", observation)
        #     print("Reward: ", reward)
        #     print("Done: ", done)
        #     print("Info: ", info)
        #     print("Action: ",  a)
        #     print("*********************************************")
    
        # start_learning()
        # script = ["python3.6", "/home/roboticlab14/catkin_ws/src/openai_examples_projects/my_reflex_test_openai/scripts/run_algo.py"]    
        # process = subprocess.Popen(" ".join(script),
        #                                 shell=True 
        #                                 # env={"PYTHONPATH": "."}
        #                                 )

        # python3_command = ["python3.6", "/home/roboticlab14/catkin_ws/src/openai_examples_projects/my_reflex_test_openai/scripts/run_algo.py"]  # launch your python2 script using bash

        # process = subprocess.Popen(python3_command, stdout=subprocess.PIPE, shell=True)
        # output, error = process.communicate()  # receive output from the python2 script


        # print("Before reset the env" )
        # env.reset()
        # print("After reset the env" )

        # for i in range(0,10):
        #     a=random_action()
        #     env.step(a)
        #     print(a)

        # print("Before reset the env" )
        # env.reset()
        # print("After reset the env" )
        

        # To never finish
        while True:
            a=1
            # a=random_action()
            # env.step(a)
            # print()

        # For testing 
        # for episode in range(max_episode):
        #     observation = env.reset()
        #     print(episode)
        print("Close node: start_qlearning_reflex.py")
        print("Close node: start_qlearning_reflex.py")
        print("Close node: start_qlearning_reflex.py")
        print("Close node: start_qlearning_reflex.py")
        print("Close node: start_qlearning_reflex.py")
        print("Close node: start_qlearning_reflex.py")
        print("Close node: start_qlearning_reflex.py")
        print("Close node: start_qlearning_reflex.py")
        # env.close()