from setuptools import find_packages, setup
from grun import __version__

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="gpu-runner",
    version=__version__,
    description="Run a command requiring GPU resources",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=[
        "filelock",
        "nvidia-ml-py3",
        "psutil",
    ],
    entry_points={"console_scripts": ["grun=grun.__main__:main"]},
    url="https://github.com/cosmoquester/gpu-runner.git",
    author="Park Sangjun",
    packages=find_packages(exclude=["tests"]),
)
