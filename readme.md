# Tjack

## setup

First time:

```sh
# clone repository
git clone --recurse-submodules git@gitlab.liu.se:TDDE19-2021-1/tjack.git
# or `git submodule init && git submodule update` after regular clone

# use virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# install requirements
pip install -r requirements.txt

# setup build directory
meson build
# to choose non-default compiler like clang, do `CXX=clang++ meson <builddir>`
# to select build type, do meson <builddir> --buildtype={debug,release}
```

To build:

```sh
cd build
ninja
```

### olympen

Make sure the machine you are on has the correct versions of g++ and CUDA:

```sh
g++ version
# look for 'g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0'

nvidia-smi
# look for 'CUDA Version: 11.2'
```

olympen1-117.ad.liu.se is known to not satisfy these requirements and will not work.

Install PyTorch with CUDA support:

```sh
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Ensure that CUDA is available:

```sh
./scripts/has_cuda.py
```

## engines

To create an engine, you can copy the [example](example) directory and edit the files accordingly. You also have to add the engine to the [build script](meson.build).

## lichess

Play with bot on lichess via [lichess-bot](https://github.com/ShailChoksi/lichess-bot):

```bash
# download lichess-bot somewhere
git clone https://github.com/ShailChoksi/lichess-bot.git <path/to/lichess-bot>

# export bot token token
export LICHESS_BOT_TOKEN=<token>

# run engine
cd <engine>
python3 <path/to/lichess-bot>/lichess-bot.py
```

Then challenge here:

https://lichess.org/?user=tjack-bot#friend

## evaluate

Estimate Elo rating of of engine:

```
python3 scripts/elo_est.py build/<engine>
```

## links

- [Wiki](https://gitlab.liu.se/groups/tdde19-group-1/-/wikis/home) (for detailed documentation and other resources)
