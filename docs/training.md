---
layout: default
title: Training
nav_order: 4
---

# Training

To train a pre-existing environment, check out the configuration files in `config/`. These files specify everything from model architecture and MORLAX parameters to reward and environment constants.

Choose the config file you want, edit the parameters to your liking, and run:

```bash
python3 -m scripts.train config_path
```

where `config_path` is the path to the config of your choice.

If you downloaded a policy in the past, you can also use those configs to run an identical training run on your system.

## Writing your own training scripts

_Coming soon._
