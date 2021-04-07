from setuptools import setup
import setuptools
import versioneer
with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='gym_microrts',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=['gym', 'dacite', 'jPype1', 'Pillow'],
    packages=setuptools.find_packages(),
)
