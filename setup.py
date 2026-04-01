
# -*- coding: utf-8 -*-


from setuptools import find_packages, setup

setup(
    name='UPF_RPE',
    packages=find_packages(include=['Code', 'Code.*']),
    version='0.0.1',
    description='UPF for relative pose estimation using odometry and UWB measurements',
    author='Yuri Durodie',
    license='GPL3',
    install_requires=["numpy","filterpy"],
)