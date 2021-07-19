import gym
import numpy as np

from scipy.ndimage import gaussian_filter
from collections import deque
from abc import abstractmethod, ABC


def normalize_image(array_img):
    return  (array_img - 127.5) /127.5


def add_noise(array_img):
    array_img = array_img + np.random.normal(0, 0.25, array_img.shape)

    array_img = np.clip(array_img, -1, 1)
    return array_img


def normalize_and_add_noise(array_img):
    return add_noise(normalize_image(array_img))


class AbstractNoisePlanner(ABC):

    @abstractmethod
    def __call__(self, frame, *args, **kwargs):
        pass


class ConstantGaussianNoisePlanner(ABC):
    def __init__(self, sigma=1):
        self._sigma = sigma

    def __call__(self, frame, *args, **kwargs):
        return gaussian_filter(frame.copy(), sigma=self._sigma)


class NoisyAtariWrapper(gym.Env):
    def __init__(
            self,
            atari_name: str,
            frame_stacks=8,
            noise_planner=ConstantGaussianNoisePlanner(sigma=1)
        ):
        self.wrapped = gym.make(atari_name)
        self.__frame_stack_len = frame_stacks
        self.frame_stacks = deque(maxlen=self.__frame_stack_len)
        self.noise_planner = noise_planner
        self.zero_obses = np.zeros(shape=self.wrapped.observation_space.shape)

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
        self.frame_stacks.append(self.noise_planner(frame.copy()))
        return np.transpose(np.concatenate(self.frame_stacks, axis=2), (2, 0, 1))

    def step(self, action):
        next_frame, reward, done, info = self.wrapped.step(action)
        self.frame_stacks.append(self.noise_planner(next_frame.copy()))
        return np.transpose(np.concatenate(self.frame_stacks, axis=2), (2, 0, 1)), reward, done, info


if __name__ == '__main__':
    from PIL import Image

    env = NoisyAtariWrapper('Breakout-v0', frame_stacks=4, noise_planner=ConstantGaussianNoisePlanner(sigma=1))
    env.reset()
    for i in range(8):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
    obs = obs[:, :, 21:24]
    obs = gaussian_filter(obs, sigma=1)


    img = Image.fromarray(obs)
    img.show()
