from setuptools import find_packages, setup

setup(
    name="gpu-runner",
    version="0.0.1",
    description="Run a command requiring GPU resources",
    python_requires=">=3.7",
    install_requires=[
        "filelock",
        "nvidia-ml-py3",
    ],
    entry_points={"console_scripts": ["grun=grun.__main__:main"]},
    url="https://github.com/cosmoquester/gpu-runner.git",
    author="Park Sangjun",
    packages=find_packages(exclude=["tests"]),
)
