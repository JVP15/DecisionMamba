# Decision Mamba

Basically, this is [decision transformers](https://github.com/kzl/decision-transformer), but with the Mamba architecture. 
Since Mamba has the 'convolutional' mode for parallelizable training, it is a prime candidate for offline RL.
Since it also has a 'recurrent' mode, it can do fast inference, which would be ideal for online training. 

## Roadmap

1. Train Mamba on offline Mujoco dataset [ ] (working on training code right now)
2. Evaluate Mamba against base Decision Transformer [ ]
3. Fine-tune Mamba using an online RL algorithm (probably PPO) [ ]

Contributions welcome

## Installing

Install `requirements.txt` with `pip install -r requirements.txt`. I also ran these commands to installs these packages in Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf \
    xvfb
```

Honestly I don't know what any of them do, but they were in the list of packages to install in the [train your first decision transformer](https://github.com/huggingface/blog/blob/main/notebooks/101_train-decision-transformers.ipynb) notebook so I installed them.

### Installing Mujoco

It's super easy to install the newer version of mujoco. Barely an inconvenience. But to access older (v3 and below) versions of the mujoco envs...
You have to install `mujoco-py` and that is a *pain*. Here's what I found:

1. If you get an error trying to compile `mujoco-py` the first time it runs, try Cython < 3 (which I have in the requirements.txt file). See [this issue](https://github.com/openai/mujoco-py/issues/773) for more info.
2. If you get an error like `GLIBCXX_3.4.29 not found`, read [this StackOverflow post](https://stackoverflow.com/questions/72205522/glibcxx-3-4-29-not-found). Basically, delete `libstdc++.so.6` (or whatever file is causing the error) from your `anaconda_install_dir/envs/env_name/lib/` folder. 
If it is coming from a system path, not Anaconda, you probably shouldn't delete it.

### Scratchpad

Okay so the recurrent mode is slower than the normal mode for some reason, even though when the sizes are about the same for text generation, inference for mamba is about as fast as inference for DT
and it is a lot faster when the graph is cached (like, 6x).