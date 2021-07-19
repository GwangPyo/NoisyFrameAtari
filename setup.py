from setuptools import setup

setup(
   name='noisy_atari',
   version='1.0',
   description='A useful module',
   author='Han il seok',
   author_email='x2ever.han@gmail.com',
   packages=['noisy_atari'],
   install_requires=['stable_baselines3[extra]', 'numpy', 'atari-py'],
)