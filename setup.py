
from setuptools import setup, find_packages

setup(
    name='psyclab',
    version='1.0.0',
    url='https://github.com/venachescu/psyclab.git',
    author='Vince Enachescu',
    author_email='enachesc@usc.edu',
    description='Psychophysics and Simulations Laboratory',
    packages=find_packages(),
    scripts=['scripts/notebook'],
    install_requires=[
        'numpy >= 1.11.1',
        'matplotlib >= 1.5.1',
        'python-osc',
        'six',
        'glumpy',
        'sysv_ipc',
        'CppHeaderParser'
    ],
)
