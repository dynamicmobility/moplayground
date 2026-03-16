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

# Basic Usage
Several scripts have been provided that let you train and rollout policies.
Below, we'll show the primary usage of MO-Playground, evalauting and training policies.
## Downloading and evaluating policies
To download and evaluate policies, simply run 
```python
# example code here
```

## Training
Add a description and code example here.

```python
print('hello')
```

# Advanced Usage
## Writing your own evaluation scripts
TODO
## Writing your own training scripts
TODO
## Creating custom environments

Add a description and code example here.

```python
# example code here
```
