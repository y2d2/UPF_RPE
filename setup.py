
# -*- coding: utf-8 -*-


from setuptools import find_packages, setup

setup(
    name='UPF_RPE',
    packages=find_packages(include=['ParticleFilter', 'BaseLines']),
    version='0.1.2',
    description='UWB egomotion PF for Pose estimation.',
    author='Yuri Durodie',
    license='MIT',
    install_requires=["numpy",],
)