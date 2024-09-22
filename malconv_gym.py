import hashlib
import os
import random
import sys
from collections import OrderedDict

import gym
import numpy as np
from gym import spaces
from malware_rl.envs.controls import modifier
from malware_rl.envs.utils import interface, malconv

random.seed(0)
module_path = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]

ACTION_LOOKUP = {i: act for i, act in enumerate(modifier.ACTION_TABLE.keys())}

mc = malconv.MalConv()
malicious_threshold = mc.malicious_threshold


class MalConvEnv(gym.Env):
    """Create MalConv gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        sha256list,
        random_sample=True,
        maxturns=5,
        output_path="data/evaded/malconv",
    ):
        super().__init__()
        self.available_sha256 = sha256list
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        self.observation_space = spaces.Box(
            low=0,
            high=256,
            shape=(1048576,),
            dtype=np.int16,
        )
        self.maxturns = maxturns
        self.feature_extractor = mc.extract
        self.output_path = output_path
        self.random_sample = random_sample
        self.history = OrderedDict()
        self.sample_iteration_index = 0

        self.output_path = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__),
                    ),
                ),
            ),
            output_path,
        )

    def step(self, action_ix):
        print(self.sha256)
        # Execute one time step within the environment
        self.turns += 1
        self._take_action(action_ix)
        self.observation_space = self.feature_extractor(self.bytez)
        self.score = mc.predict_sample(self.observation_space)

        if self.score < malicious_threshold:
            reward = 10.0
            episode_over = True
            self.history[self.sha256]["evaded"] = True
            self.history[self.sha256]["reward"] = reward

            # save off file to evasion directory
            m = hashlib.sha256()
            m.update(self.bytez)
            sha256 = m.hexdigest()
            evade_path = os.path.join(self.output_path, sha256)

            with open(evade_path, "wb") as out:
                out.write(self.bytez)

            self.history[self.sha256]["evade_path"] = evade_path

        elif self.turns >= self.maxturns:
            # game over - max turns hit
            reward = self.original_score - self.score
            episode_over = True
            self.history[self.sha256]["evaded"] = False
            self.history[self.sha256]["reward"] = reward

        else:
            reward = float(self.original_score - self.score)
            episode_over = False

        if episode_over:
            print(f"Episode over: reward = {reward}")

        return self.observation_space, reward, episode_over, self.history[self.sha256]

    def _take_action(self, action_ix):
        action = ACTION_LOOKUP[action_ix]
        # print("ACTION:", action)
        self.history[self.sha256]["actions"].append(action)
        self.bytez = modifier.modify_sample(self.bytez, action)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.turns = 0
        while True:
            # grab a new sample (TODO)
            if self.random_sample:
                self.sha256 = random.choice(self.available_sha256)
            else:
                self.sha256 = self.available_sha256[
                    self.sample_iteration_index % len(self.available_sha256)
                ]
                self.sample_iteration_index += 1

            self.history[self.sha256] = {"actions": [], "evaded": False}
            self.bytez = interface.fetch_file(
                os.path.join(
                    module_path,
                    "utils/samples/",
                )
                + self.sha256,
            )

            self.observation_space = self.feature_extractor(self.bytez)
            self.original_score = mc.predict_sample(self.observation_space)
            if self.original_score < malicious_threshold:
                # already labeled benign, skip
                continue

            break
        print(f"Sample: {self.sha256}")

        return self.observation_space

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        pass
