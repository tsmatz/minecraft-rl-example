import ray
import ray.tune as tune
from ray.rllib import rollout
from ray.tune.registry import get_trainable_cls
import gym

import argparse
from pathlib import Path
import random
import time

from EnvWrapper import MalmoEnv

def create_env(config):
    xml = Path(config["mission_file"]).read_text()
    env = MalmoEnv(
        xml=xml,
        width=config["width"],
        height=config["height"],
        millisec_per_tick=config["millisec_per_tick"],
        mazeseed=config["maze_seed"],
        enable_action_history=config["enable_action_history"])
    return env

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mission_path",
        help="full path to the mission file lava_maze_malmo.xml",
        type=str)
    parser.add_argument('--checkpoint_file',
        required=False,
        default="./checkpoint/checkpoint-622",
        help="trained checkpoint file path")
    parser.add_argument("--num_gpus",
        type=int,
        required=False,
        default=0,
        help="number of gpus")
    args = parser.parse_args()

    ray.init()

    run = True
    while run:

        #
        # Generate a seed for maze
        #

        print("Generating new seed ...")
        maze_seed = random.randint(1, 9999)

        #
        # Run agent with trained checkpoint
        #

        print("An agent is running ...")
        tune.register_env("testenv01", create_env)
        cls = get_trainable_cls("DQN")
        config={
            "env_config": {
                "mission_file": args.mission_path,
                "width": 84,
                "height": 84,
                "millisec_per_tick": 20,
                "maze_seed": maze_seed,
                "enable_action_history": True
            },
            "framework": "tf",
            "num_gpus": args.num_gpus,
            "num_workers": 0,
            "double_q": True,
            "dueling": True,
            "explore": False
        }
        agent = cls(env="testenv01", config=config)
        #agent.optimizer.stop()
        agent.restore(args.checkpoint_file)
        env1 = agent.workers.local_worker().env
        obs = env1.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.compute_action(obs)
            obs, reward, done, info = env1.step(action)
            total_reward += reward
        env1.close()
        agent.stop()
        print("Done with reward ", total_reward)

        #
        # Simulate same result with wide screen
        #

        xml = Path(args.mission_path).read_text()
        env2 = MalmoEnv(
            xml=xml,
            width=800,
            height=600,
            millisec_per_tick=50,
            mazeseed=maze_seed)
        env2.reset()
        print("The world is loaded.\nPress Enter and F5 key in Minecraft to show third-person view.")
        input("Enter keyboard to start simulation")
        for action in env1.action_history:
            time.sleep(0.5)
            obs, reward, done, info = env2.step(action)
        user_choice = input("Enter 'N' to exit [Y/n]: ").lower()
        if user_choice in ['n', 'no']:
            run = False
        env2.close()

    ray.shutdown()
