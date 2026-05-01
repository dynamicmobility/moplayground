---
layout: default
title: Installation
nav_order: 1
---

# Installation

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
