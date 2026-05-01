---
layout: default
title: Overview
nav_order: 1
---

# Overview

**MO-Playground** is a Python library for massively parallelized **multi-objective reinforcement learning (MORL)** for robotics, built on [JAX](https://github.com/google/jax) and [MuJoCo Playground](https://playground.mujoco.org). It provides GPU-accelerated environments, training utilities (MORLAX, a multi-objective extension of PPO), and a set of pre-defined robotics tasks — `cheetah`, `hopper`, `walker`, `ant`, `humanoid`, and the `bruce` humanoid — for benchmarking and developing MORL policies that trade off between competing reward objectives.

## Installation

First, clone the repository via SSH or HTTPS from [the moplayground repository](https://github.com/dynamicmobility/moplayground).

Note that a `pip`-installable package is coming, and will be released after double-blind review!

Next, using the provided ymls, create a new conda environment.

**If you want to enable GPU based training, run:**

```bash
# Navigate to the moplayground directory
conda env create -f environment.yml
conda activate moplayground
```

**If you just want to evaluate policies and explore the code (i.e. running on a Mac):**

```bash
# Navigate to the moplayground directory
conda env create -f mac_environment.yml
conda activate moplayground
```

## Basic Usage

Several scripts have been provided that let you train and rollout policies. The following pages walk through the primary usage of MO-Playground:

- [Simulating]({% link simulating.md %}) — download pre-trained policies and roll them out.
- [Training]({% link training.md %}) — train a policy on an existing environment.
- [Custom Environments]({% link custom-environments.md %}) — define your own environment.
- [Plotting]({% link plotting.md %}) — visualize results.

## Prerequisites for advanced usage

For advanced usage of MO-Playground, it is generally assumed that you are familiar with [MuJoCo Playground](https://playground.mujoco.org). Specifically, how the `MjxEnv` class works, including the `step` and `reset` functions.

It is also useful to take a look at how PPO works in [brax](https://github.com/google/brax) before diving into MORLAX.
