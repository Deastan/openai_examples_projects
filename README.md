Come from the constructcore but modify by me to add a robot.

#Prerequire
Depend of openai from deastan:
https://github.com/Deastan/openai_ros.git


#Installation
Go to your catkin workspace, here catkin_ws
$ cd catkin_ws
$ cd src
$ git clone https://github.com/Deastan/openai_ros.git
$ git clone https://github.com/Deastan/openai_examples_projects.git
$ cd ..
$ catkin_make 
or 
$ catkin build
$ source devel/setup.bash

#Run (to test if it work):

roslaunch my_reflex_test_openai start_training_reflex.launch