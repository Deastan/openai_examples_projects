#!/usr/bin/env python2

import sys
# if sys.version_info[0] < 3:
#         raise Exception("Must be using Python 3 on ROS")

import gym
import numpy
import time
import qlearn
import random
from gym import wrappers
from gym.envs.registration import register
# ROS packages required
import rospy
import rospkg
# import our training environment
from openai_ros.task_envs.iiwa_tasks import iiwa_move
# from openai_ros.task_envs.hopper import hopper_stay_up
# import pickle, os

# For the launch 
import roslaunch
import os
import git
import sys

def random_action():
    a = []
    for i in range(0, 6):
        a.append(random.uniform(-0.01, 0.01))
        # i = i+1
    return a


if __name__ == '__main__':
        # Parameters
        timestep_limit_per_episode = 10000
        max_episode = 10

        # Begining of the script
        print("Python version: ", sys.version)
        rospy.init_node('start_qlearning_reflex', anonymous=True, log_level=rospy.WARN)
    
        print("Start node: start_qlearning_reflex file: start_qlearning_reflex.py")

        # Cheating with the registration 
        # (shortcut: look at openai_ros_common.py and task_envs_list.py)
        register(
            id="iiwaMoveEnv-v0",
            entry_point='openai_ros.task_envs.iiwa_tasks.iiwa_move:iiwaMoveEnv',
        #     timestep_limit=timestep_limit_per_episode,
            max_episode_steps=timestep_limit_per_episode,
        )

        # Create the Gym environment
        env = gym.make('iiwaMoveEnv-v0')
        rospy.loginfo("Gym environment done")
        print("Gym environment done")

        print("Before reset the env" )
        env.reset()
        print("After reset the env" )

        print("Action 1" )
        action = [-0.5, -0.5, -0.5, 1.3, 0.1, 0.5]
        print("Set Action: " + str(action))
        env.step(action)

        print("Action 2" )
        # action2 = [0.2, 0.2, 0.2, -1.3, -0.1, -0.5]
        action2 = [0.2, 0.2, 0.2, 0.0, 0.0, 0.0]
        print("Set Action: " + str(action2))
        env.step(action2)

        # print("Action are sent")
        print("Before reset the env" )
        env.reset()
        print("After reset the env" )
        # env._set_action(action)
        # # Set the logging system
        # rospack = rospkg.RosPack()
        # pkg_path = rospack.get_path('my_hopper_openai_example')
        # outdir = pkg_path + '/training_results'
        # env = wrappers.Monitor(env, outdir, force=True)
        # rospy.loginfo("Monitor Wrapper started")
        for i in range(0,15):
            a=random_action()
            env.step(a)
            print(a)

        print("Before reset the env" )
        env.reset()
        print("After reset the env" )

        for i in range(0,10):
            a=random_action()
            env.step(a)
            print(a)

        print("Before reset the env" )
        env.reset()
        print("After reset the env" )
        
        # while True:
        #     a=random_action()
        #     env.step(a)
        #     print(a)

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
        env.close()