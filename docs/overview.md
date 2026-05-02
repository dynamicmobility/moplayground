---
layout: default
title: Overview
nav_order: 1
---

# Overview

**MO-Playground** is a Python library for massively parallelized **multi-objective reinforcement learning (MORL)** for robotics, built on [JAX](https://github.com/google/jax) and [MuJoCo Playground](https://playground.mujoco.org). It provides GPU-accelerated environments, training utilities (MORLAX, a multi-objective extension of PPO), and a set of pre-defined robotics tasks — `cheetah`, `hopper`, `walker`, `ant`, `humanoid`, and the `bruce` humanoid — for benchmarking and developing MORL policies that trade off between competing reward objectives.

## Installation

There are two ways to install moplayground: via `pip` or `git`.

1. To install using `pip`, simply use `pip install moplayground`.

2. For `git`, copy the SSH or HTTPS link from [the moplayground repository](https://github.com/dynamicmobility/moplayground) and clone the repoistory using `git clone`. Next, using the provided ymls, create a new conda environment.

    1. If you want to enable GPU-based training or are using Linux (tested on Ubuntu 22.04), run:

        ```bash
        # Navigate to the moplayground directory
        conda env create -f environment.yml
        conda activate moplayground
        pip install -e .
        ```

    2. If you just want to evaluate policies and explore the code (i.e. running on a Mac):

        ```bash
        # Navigate to the moplayground directory
        conda env create -f mac_environment.yml
        conda activate moplayground
        pip install -e .
        ```

    That last line locally installs the package into your Python environment in editable mode, so you can change `moplayground`'s code and it will be reflected when you import `moplayground` in your scripts.

## Basic Usage

Several scripts have been provided that let you train and rollout policies. The following pages walk through the primary usage of MO-Playground:

- [Simulating]({% link simulating.md %}) — download pre-trained policies and roll them out.
- [Training]({% link training.md %}) — train a policy on an existing environment.
- [Custom Environments]({% link custom-environments.md %}) — define your own environment.
- [Plotting]({% link plotting.md %}) — visualize results.

## General notes for development

When working on MO-Playground, it is generally assumed that you are familiar with [MuJoCo Playground](https://playground.mujoco.org). Specifically, how the `MjxEnv` class works, including the `step` and `reset` functions. 

The main change from [MuJoCo Playground](https://playground.mujoco.org) is that we use a `SwappableBase` class implemented in `moplayground`'s RL backend: [`minimal-mjx`](https://github.com/dynamicmobility/minimal-mjx). The most important thing here is that all numpy operations are done using `self._np` (i.e. `self._np.zeros(3)`), since we change the `numpy` and `mujoco` backends depending on if `MJX` is being used or not (i.e. CPU vs GPU tasks).

It is also useful to take a look at how PPO works in [brax](https://github.com/google/brax) before diving into MORLAX.
