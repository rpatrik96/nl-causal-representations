from setuptools import setup, find_packages

setup(
    name='causalnlica',
    version='0.1.0',
    packages=find_packages(include=['pytorch_flows', 'pytorch_flows.*', 'src', 'src.*'])
)