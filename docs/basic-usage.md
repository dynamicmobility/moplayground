---
layout: default
title: Basic Usage
nav_order: 2
---

# Basic Usage

Several scripts have been provided that let you train and rollout policies. The following pages walk through the primary usage of MO-Playground:

- [Simulating]({% link simulating.md %}) — download pre-trained policies and roll them out.
- [Training]({% link training.md %}) — train a policy on an existing environment.
- [Custom Environments]({% link custom-environments.md %}) — define your own environment.
- [Plotting]({% link plotting.md %}) — visualize results.

## Prerequisites for advanced usage

For advanced usage of MO-Playground, it is generally assumed that you are familiar with [MuJoCo Playground](https://playground.mujoco.org). Specifically, how the `MjxEnv` class works, including the `step` and `reset` functions.

It is also useful to take a look at how PPO works in [brax](https://github.com/google/brax) before diving into MORLAX.
