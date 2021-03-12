import gym
import numpy as np
from pathlib import Path
import argparse
import os

import ray
import ray.tune as tune

from EnvWrapper import MalmoEnv

# For creating OpenAI gym environment (custom MalmoEnv)
def create_env(config):
    xml = Path(config["mission_file"]).read_text()
    env = MalmoEnv(
        xml=xml,
        width=config["width"],
        height=config["height"],
        millisec_per_tick=config["millisec_per_tick"])
    return env

# For stopping a learner for successful training
def stop_check(trial_id, result):
    return result["episode_reward_mean"] >= 85

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mission_path",
        help="full path to the mission file lava_maze_malmo.xml",
        type=str)
    parser.add_argument("--num_gpus",
        type=int,
        required=False,
        default=0,
        help="number of gpus")
    args = parser.parse_args()

    tune.register_env("testenv01", create_env)

    ray.init()

    tune.run(
        run_or_experiment="DQN",
        config={
            "log_level": "WARN",
            "env": "testenv01",
            "env_config": {
                "mission_file": args.mission_path,
                "width": 84,
                "height": 84,
                "millisec_per_tick": 20
            },
            "framework": "tf",
            "num_gpus": args.num_gpus,
            "num_workers": 1,
            "double_q": True,
            "dueling": True,
            "explore": True,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 500000
            }
        },
        stop=stop_check,
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir='./logs'
    )

    print('training has done !')
    ray.shutdown()
