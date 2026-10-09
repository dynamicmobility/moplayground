"""Multi-objective JAX/MuJoCo environments.

Provides the registered MO-Playground environments (``MOCheetah``,
``MOHopper``, ``MOAnt``, ``MOWalker``, ``MOHumanoid``, ``NaviGait``) and
the base classes (``MultiObjectiveBase``, ``Multi2SingleObjective``)
they share. Use ``create_environment(env_name, env_params, ...)`` to
construct one.
"""
from . import dmcontrol, generic, locomotion
from .create import create_environment
from .generic import MultiObjectiveBase, Multi2SingleObjective