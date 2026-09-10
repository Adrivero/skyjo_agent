# SKYJO MARL agent

This project aims to train learn a policy via **multi-agentic reinforcement learning (MARL)** to play the card game SKYJO™. 

> [!NOTE]
> This project is still in development.

## Setup

Project managed with [uv packet manager](https://github.com/astral-sh/uv) 

## Train

The production defaults train for ten million learner decisions:

```sh
uv run python train.py --seed 1
```

Run three independent seeds by giving each run its own output directories:

```sh
uv run python train.py --seed 1 --run-dir runs/seed_1 --output policies/seed_1.pt
uv run python train.py --seed 2 --run-dir runs/seed_2 --output policies/seed_2.pt
uv run python train.py --seed 3 --run-dir runs/seed_3 --output policies/seed_3.pt
```
