# sigmazero

Implementation of the [AlphaZero](https://arxiv.org/abs/1712.01815) algorithm.

Feature highlights:

- Fast generation of legal moves using [magic bitboards](https://www.chessprogramming.org/Magic_Bitboards).
- Distributed training with one training process and multiple self-play processes.
- Batched Monte Carlo tree search.
- Communication with user interfaces and [lichess.org](https://lichess.org/) using [UCI](https://www.chessprogramming.org/UCI).

Originally a group project in [TDDE19](https://www.ida.liu.se/~TDDE19/) at Linköping University.

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
# to select build type, do `meson <builddir> --buildtype={debug,release}`
# if the build fails with some torch error, try `meson <builddir> -D_GLIBCXX_USE_CXX11_ABI=0
```

To build:

```sh
cd build
ninja
```

## running

Run one selfplayer and one trainer:

```bash
model=model.pt
./selfplay $model | ./training $model
```

Run multiple selfplayers and one trainer:

```bash
model=model.pt
./training $model <(./selfplay $model) <(./selfplay $model) <(./selfplay $model)
```

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
