# GPU Runner

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![cosmoquester](https://circleci.com/gh/cosmoquester/python3-template.svg?style=svg)](https://app.circleci.com/pipelines/github/cosmoquester/python3-template)

GPU Runner is a Python package that provides a simple way to run a function on a GPU.

## Installation

```bash
$ pip install gpu-runner
```

I recommend using [pipx](https://pipx.pypa.io/stable/installation/) to install this package with independency.

## Usage

### Run command on a GPU

```bash
$ CUDA_VISIBLE_DEVICES=0 python train.py --args 1 2 3
~~~~~~~~~~
```
- this is a traditional way to run a function on a GPU.
- but, it's hassle to check the available GPU and set the `CUDA_VISIBLE_DEVICES` environment variable.

```bash
$ grun python train.py --args 1 2 3 --kwargs key1 value1 key2 value2

[GRUN] Number of GPUs: 8
[GRUN] Non-Utilized GPUs: [0, 2]
[GRUN] Selected GPUs: [0]
[GRUN] Run: python train.py --args 1 2 3 --kwargs key1 value1 key2 value2
~~~~~~~~~~
[GRUN] Done.
```
- do not use `CUDA_VISIBLE_DEVICES` directly.
- just use `grun` command if you want to run a function on a GPU.
- `grun` will automatically select the available GPU.

### Run command on multiple GPUs

```bash
$ grun -n 2 python train.py --args 1 2 3 --kwargs key1 value1 key2 value2 --device 0

[GRUN] Number of GPUs: 8
[GRUN] Non-Utilized GPUs: [0, 2]
[GRUN] Selected GPUs: [0, 2]
[GRUN] Run: python train.py --args 1 2 3 --kwargs key1 value1 key2 value2 --device 0
~~~~~~~~~~~
[GRUN] Done.
```
- `-n` option is the number of GPUs to use.

### Run prolonged command

```bash
# Without wating option (-w)
$ grun -n 8 python train.py --args 1 2 3 --kwargs key1 value1 key2 value2 --device 0 1

[GRUN] Number of GPUs: 8
[GRUN] Non-Utilized GPUs: [0, 2]
[GRUN] 8 GPUs requested, but only 2 gpus [0, 2] available.

# With wating option (-w)
$ grun -n 8 -w python train.py --args 1 2 3 --kwargs key1 value1 key2 value2 --device 0 1
[GRUN] Number of GPUs: 8
[GRUN] Non-Utilized GPUs: [0, 2]
[GRUN] 8 GPUs requested, but only 2 gpus [0, 2] available.
[GRUN] Start Waiting for more GPUs...
[GRUN]  [2024-08-06 20:27:20] Waiting for 8 GPUs...
[GRUN] Selected GPUs: [0, 1, 2, 3, 4, 5, 6, 7]
[GRUN] Run: python train.py --args 1 2 3 --kwargs key1 value1 key2 value2 --device 0 1
~~~~~~~~~~~
[GRUN] Done.
```
- with `-w` option, grun will wait to get the GPU resources until they are available.
- it automatically waits for the resources to be available and then runs the command.
- the execution order of the commands with `-w` option is guaranteed.

### Freeze current working directory

```bash
$ grun -f python train.py --args 1 2 3 --kwargs key1 value1 key2 value2 --device 0
[GRUN] Freeze current working directory to /tmp/grun_fx1p1ljv
[GRUN] Number of GPUs: 8
[GRUN] Non-Utilized GPUs: [0, 2]
[GRUN] Selected GPUs: [0]
[GRUN] Run: python train.py --args 1 2 3 --kwargs key1 value1 key2 value2 --device 0
~~~~~~~~~~~
[GRUN] Done.
```
- it is easy to change the current working directory after executing the command with `-w` option.
- by default, the command will be executed with the changed code in the future.
- if you want to execute the command with the current code, use `-f` option.
- this option copies the current code to the temporary directory and executes the command.
