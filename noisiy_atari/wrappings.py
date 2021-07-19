import gym
from collections import deque
import numpy as np
from PIL import Image


def normalize_image(array_img):
    return  (array_img - 127.5) /127.5


def add_noise(array_img):
    array_img = array_img + np.random.normal(0, 0.25, array_img.shape)

    array_img = np.clip(array_img, -1, 1)
    return array_img


def normalize_and_add_noise(array_img):
    return add_noise(normalize_image(array_img))


class NoisyAtariWrapper(gym.Env):
    def __init__(self, atari_name: str, frame_stacks=8, noise_ratio=0.2):
        self.wrapped = gym.make(atari_name)
        self.__frame_stack_len = frame_stacks
        self.frame_stacks = deque(maxlen=self.__frame_stack_len)
        self.noise_ratio = noise_ratio
        self.zero_obses = np.zeros(shape=self.wrapped.observation_space.shape)
        print(self.zero_obses.shape)

    @property
    def observation_space(self):
        shape = self.wrapped.observation_space.shape
        w = shape[0]
        h = shape[1]
        c = shape[2]
        return gym.spaces.Box(low=-1, high=1, shape=(w, h, c * self.__frame_stack_len))

    @property
    def action_space(self):
        return self.wrapped.action_space

    def render(self, mode='human'):
        return self.wrapped.render(mode)

    def reset(self):
        frame = self.wrapped.reset()
        for _ in range(self.__frame_stack_len):
            self.frame_stacks.append(self.zero_obses.copy())
        self.frame_stacks.append(normalize_and_add_noise(frame))
        return np.stack(self.frame_stacks, axis=2)

    def step(self, action):
        next_frame, reward, done, info = self.wrapped.step(action)
        self.frame_stacks.append(normalize_and_add_noise(next_frame))
        return np.stack(self.frame_stacks, axis=2), reward, done, info


if __name__ == '__main__':
    env = NoisyAtariWrapper('Breakout-v0')
    atari = gym.make("Breakout-v0")
    obs = env.reset()
    print(obs.shape)
