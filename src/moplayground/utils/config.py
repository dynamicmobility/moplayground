"""Single access point between YAML config files and the ``moplayground`` package.
"""
import minimal_mjx as mm
import numpy as np


class MOConfig(mm.Config):
    
    @property
    def default_scalarization(self) -> dict:
        """Sampling settings: ``alpha``, ``k``, ``sampling``, ``warmup_frac``."""
        return np.array(self.cfg.env_config.reward.optimization.default_scalarization)

    @property
    def sampling_params(self) -> dict:
        """Sampling settings: ``alpha``, ``k``, ``sampling``, ``warmup_frac``."""
        return self.cfg.learning_params.sampling_params.to_dict()
    
    @property
    def ppo_network_params(self) -> dict:
        """Network settings for PPO."""
        return self.cfg.learning_params['ppo_network_params'].to_dict()
    
    @property
    def morlax_network_params(self) -> dict:
        """Network settings for the configured algorithm."""
        return self.cfg.learning_params[f'morlax_network_params'].to_dict()
    
    @property
    def amor_network_params(self) -> dict:
        """Network settings for the configured algorithm."""
        return self.cfg.learning_params[f'amor_network_params'].to_dict()
    
    @property
    def objectives(self) -> list:
        """Reward keys per objective, one list per reward dimension."""
        return list(self.cfg.env_config.reward.optimization.objectives)

    @property
    def num_objectives(self) -> int:
        """Number of objectives (length of the reward vector)."""
        return len(self.cfg.env_config.reward.optimization.objectives)

    @property
    def objective_labels(self):
        """Display names for the objectives, or ``None`` if not set."""
        labels = self.cfg.env_config.reward.optimization.get('labels')
        return None if labels is None else list(labels)


# TODO: move to mm
def check_same_env(cfgs):
    """Raise ``ValueError`` unless every config deploys the same environment.

    Compares ``env`` and the full ``env_config`` section. Use this before
    evaluating several runs together (e.g. ``eval.pareto.get_morlax_fronts``
    with a list of run directories).
    """
    cfgs = list(cfgs)
    if len(cfgs) <= 1:
        return
    ref = cfgs[0]
    for i, c in enumerate(cfgs[1:], start=1):
        if c.env != ref.env:
            raise ValueError(
                f"Config[{i}] env={c.env!r} does not match config[0] env={ref.env!r}; "
                f"all configs must deploy the same environment."
            )
        if c.env_config.to_dict() != ref.env_config.to_dict():
            raise ValueError(
                f"Config[{i}] env_config differs from config[0]; "
                f"all configs must deploy the same environment."
            )