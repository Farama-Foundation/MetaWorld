"""Use this script to control the env with your keyboard.

For this script to work, you need to have the PyGame window in focus.

See/modify `char_to_action` to set the key-to-action mapping.
"""
import sys

import numpy as np
import pygame  # type: ignore
from pygame.locals import KEYDOWN, QUIT  # type: ignore

from metaworld.envs.v2 import SawyerPickPlaceEnvV2

pygame.init()
screen = pygame.display.set_mode((400, 300))


char_to_action = {
    "w": np.array([0, -1, 0, 0]),
    "a": np.array([1, 0, 0, 0]),
    "s": np.array([0, 1, 0, 0]),
    "d": np.array([-1, 0, 0, 0]),
    "q": np.array([1, -1, 0, 0]),
    "e": np.array([-1, -1, 0, 0]),
    "z": np.array([1, 1, 0, 0]),
    "c": np.array([-1, 1, 0, 0]),
    "k": np.array([0, 0, 1, 0]),
    "j": np.array([0, 0, -1, 0]),
    "h": "close",
    "l": "open",
    "x": "toggle",
    "r": "reset",
    "p": "put obj in hand",
}


env = SawyerPickPlaceEnvV2()
env._partially_observable = False
env._freeze_rand_vec = False
env._set_task_called = True
env.reset()
env._freeze_rand_vec = True
lock_action = False
random_action = False
obs = env.reset()
action = np.zeros(4, dtype=np.float32)
while True:
    done = False
    if not lock_action:
        action[:3] = 0
    if not random_action:
        for event in pygame.event.get():
            event_happened = True
            if event.type == QUIT:
                sys.exit()
            if event.type == KEYDOWN:
                char = event.dict["key"]
                new_action = char_to_action.get(chr(char), None)
                if new_action == "toggle":
                    lock_action = not lock_action
                elif new_action == "reset":
                    done = True
                elif new_action == "close":
                    action[3] = 1
                elif new_action == "open":
                    action[3] = -1
                elif new_action is not None and isinstance(new_action, np.ndarray):
                    action[:3] = new_action[:3]
                else:
                    action = np.zeros(3, dtype=np.float32)
                print(action)
    else:
        action = np.array(env.action_space.sample(), dtype=np.float32)
    ob, reward, done, infos = env.step(action)
    # time.sleep(1)
    if done:
        obs = env.reset()
    env.render()
