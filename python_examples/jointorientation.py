import numpy as np
import time
import json
import redis
import math
from enum import Enum, auto
from dataclasses import dataclass

DEG_TO_RAD = math.pi / 180.0

ROBOT_NAME = "Titania"

class State(Enum):
  GOING_TO_JOINT_CONFIG = auto()
  INIT = auto()

@dataclass
class RedisKeys:
  joint_task_goal_position: str = "opensai::controllers::" + ROBOT_NAME + "::joint_controller::joint_task::goal_position"
  joint_task_current_position: str = "opensai::controllers::" + ROBOT_NAME + "::joint_controller::joint_task::current_position"
  active_controller: str = "opensai::controllers::" + ROBOT_NAME + "::active_controller_name"
  config_file_name: str = "::sai-interfaces-webui::config_file_name"

redis_keys = RedisKeys()

config_file_for_this_example = "suturebot_grav_real.xml"
joint_controller_to_use = "joint_controller"

first_joint_config = np.array([0.0, 20.0, 0.0, 10.0, 0.0, 10.0, 0.0]) * DEG_TO_RAD
final_joint_config = np.array([46.0, -98.0, -115.0, 83.0, 101.0, -24.0, -32.0]) * DEG_TO_RAD

# redis client
redis_client = redis.Redis()

# check that the config file is correct
config_file_name = redis_client.get(redis_keys.config_file_name).decode("utf-8")
if config_file_name != config_file_for_this_example:
    print("This example is meant to be used with the config file: ", config_file_for_this_example)
    exit(0)

# start with the joint controller and send the first joint configuration
while redis_client.get(redis_keys.active_controller).decode("utf-8") != joint_controller_to_use:
	redis_client.set(redis_keys.active_controller, joint_controller_to_use)
print("active controller:", redis_client.get(redis_keys.active_controller).decode("utf-8"))
time.sleep(0.1)
redis_client.set(redis_keys.joint_task_goal_position, json.dumps(first_joint_config.tolist()))

# loop at 100 Hz
loop_time = 0.0
dt = 0.01
internal_step = 0
state = State.GOING_TO_JOINT_CONFIG

time.sleep(0.01)
init_time = time.perf_counter_ns() * 1e-9

try:
  while True:
    loop_time += dt
    time.sleep(max(0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time)))
    
    if state == State.GOING_TO_JOINT_CONFIG:
      current_joint_position = np.array(json.loads(redis_client.get(redis_keys.joint_task_current_position)))
      joint_error = np.linalg.norm(first_joint_config - current_joint_position)
   
      if joint_error < .5:
        
        state = State.INIT
        print("Joint configuration reached. Switching to cartesian task controller.")


      print("joint error: ", joint_error)

    elif state == State.INIT:
      time.sleep(1)
      redis_client.set(redis_keys.joint_task_goal_position, json.dumps(final_joint_config.tolist()))
      current_joint_position = np.array(json.loads(redis_client.get(redis_keys.joint_task_current_position)))
      joint_error = np.linalg.norm(final_joint_config - current_joint_position)

      if joint_error < .5:
        
        
        state = State.INIT
        print("Joint configuration reached. Switching to cartesian task controller.")

      print("joint error: ", joint_error)

except KeyboardInterrupt:
  print("Keyboard interrupt")
  pass
except Exception as e:
  print(e)
  pass



