from setuptools import setup
import setuptools

setup(
    name='gym_microrts',
    include_package_data=True,
    version='0.1.2',
    install_requires=['gym', 'dacite', 'jPype1', 'Pillow'],
    packages=setuptools.find_packages(),
)
