"""MO-Playground: massively parallel multi-objective RL for robotics.

Every public name lives at the top level, so you never need to know the
package layout::

    import moplayground as mop

    env, params = mop.create_environment(config)
    networks    = mop.make_morlax_networks(...)
    mop.plot_pareto(front)

The subpackages (``mop.envs``, ``mop.moppo``, ...) remain importable if you
want the fully-qualified path.

The concrete environments resolve lazily (PEP 562), so importing
``moplayground`` does not pull in every environment -- in particular the
Bruce/locomotion modules, which load MuJoCo models at import time, are only
touched when actually used.
"""
import importlib
from typing import TYPE_CHECKING

# Imported eagerly (as before): these define the import order the package
# relies on -- ``moppo.acting`` imports ``learning.wrappers``, so ``learning``
# has to be in place first.
from . import envs, eval, learning, moppo, utils

_SUBPACKAGES = ('envs', 'eval', 'learning', 'moppo', 'utils')

# Public name -> module it lives in (or ``(module, original_name)`` when the
# surfaced name is an alias). Order mirrors the package layout.
_EXPORTS = {
    # --- envs ------------------------------------------------------------
    'create_environment': 'moplayground.envs.create',
    'MultiObjectiveBase': 'moplayground.envs.generic.mobase',
    'Multi2SingleObjective': 'moplayground.envs.generic.mobase',
    'MOAnt': 'moplayground.envs.dmcontrol.ant',
    'MOCheetah': 'moplayground.envs.dmcontrol.cheetah',
    'MOHopper': 'moplayground.envs.dmcontrol.hopper',
    'MOHumanoid': 'moplayground.envs.dmcontrol.humanoid',
    'MOWalker': 'moplayground.envs.dmcontrol.walker',
    'AntInterface': 'moplayground.envs.dmcontrol.interface',
    'CheetahInterface': 'moplayground.envs.dmcontrol.interface',
    'HopperInterface': 'moplayground.envs.dmcontrol.interface',
    'HumanoidInterface': 'moplayground.envs.dmcontrol.interface',
    'WalkerInterface': 'moplayground.envs.dmcontrol.interface',
    'BipedalBase': 'moplayground.envs.locomotion.generic.bipedal',
    'NaviGait': 'moplayground.envs.locomotion.generic.navigait',
    'Bruce': 'moplayground.envs.locomotion.bruce.navigait',
    # the Bruce constants/kinematics module, surfaced whole (it is a big flat
    # namespace of MuJoCo ids and conversions, not a handful of symbols)
    'bruce_interface': ('moplayground.envs.locomotion.bruce.interface_westwood', None),
    'bezier_basis_matrix': 'moplayground.envs.locomotion.control.bezier',
    'Leg': 'moplayground.envs.locomotion.control.bezier',
    'P1Bezier': 'moplayground.envs.locomotion.control.bezier',
    'GaitLibrary': 'moplayground.envs.locomotion.control.gait',
    'MIN_SWING_PHASE': 'moplayground.envs.locomotion.control.gait',

    # --- eval ------------------------------------------------------------
    # NOTE: ``learning.inference`` also defines a ``rollout_policy``; the eval
    # one wins here, matching what ``moplayground.eval`` already exported.
    'rollout_policy': 'moplayground.eval.rollout',
    'get_pareto_rollout': 'moplayground.eval.pareto',
    'compute_fronts': 'moplayground.eval.pareto',
    'get_morlax_fronts': 'moplayground.eval.pareto',
    'get_amor_fronts': 'moplayground.eval.pareto',

    # --- learning --------------------------------------------------------
    'train_policy': 'moplayground.learning.training',
    'setup_morlax': 'moplayground.learning.training',
    'setup_amor': 'moplayground.learning.training',
    'create_training_directory': 'moplayground.learning.training',
    'mo_wrapper': 'moplayground.learning.training',
    'load_mo_policy': 'moplayground.learning.inference',
    'load_hypernetworks': 'moplayground.learning.inference',
    'load_hypernetwork_inference_fn': 'moplayground.learning.inference',
    'load_amor_networks': 'moplayground.learning.inference',
    'load_make_amor_inference_fn': 'moplayground.learning.inference',
    'get_num_objectives': 'moplayground.learning.inference',
    'MultiObjectiveEpisodeWrapper': 'moplayground.learning.wrappers',
    'MultiObjectiveEvalWrapper': 'moplayground.learning.wrappers',

    # --- moppo -----------------------------------------------------------
    'MultiObjectiveTransition': 'moplayground.moppo.acting',
    'actor_step': 'moplayground.moppo.acting',
    'generate_unroll': 'moplayground.moppo.acting',
    'Evaluator': 'moplayground.moppo.acting',
    # ``morlax.train`` and ``amor.train`` share a name, so they surface under
    # disambiguated aliases.
    'train_morlax': ('moplayground.moppo.morlax', 'train'),
    'train_amor': ('moplayground.moppo.amor', 'train'),
    'MOTrainingState': 'moplayground.moppo.morlax',
    'sample_preferences': 'moplayground.moppo.morlax',
    'AMORTrainingState': 'moplayground.moppo.amor',
    'FeedForwardHypernetwork': 'moplayground.moppo.factory',
    'MORLAXNetworks': 'moplayground.moppo.factory',
    'AMORNetworks': 'moplayground.moppo.factory',
    'make_hypernetwork': 'moplayground.moppo.factory',
    'make_hypernetwork_inference_fn': 'moplayground.moppo.factory',
    'make_morlax_networks': 'moplayground.moppo.factory',
    'make_amor_networks': 'moplayground.moppo.factory',
    'make_amor_policy_network': 'moplayground.moppo.factory',
    'make_amor_value_network': 'moplayground.moppo.factory',
    'make_amor_inference_fn': 'moplayground.moppo.factory',
    'MORLAXNetworkParams': 'moplayground.moppo.losses',
    'AMORNetworkParams': 'moplayground.moppo.losses',
    'compute_mo_gae': 'moplayground.moppo.losses',
    'compute_morlax_loss': 'moplayground.moppo.losses',
    'compute_amor_loss': 'moplayground.moppo.losses',
    'MLP': 'moplayground.moppo.networks',
    'Hypernet': 'moplayground.moppo.networks',
    'HypernetMLP': 'moplayground.moppo.networks',
    'FakeHypernet': 'moplayground.moppo.networks',
    'ActorCriticHypernet': 'moplayground.moppo.networks',
    'DualA2CHypernet': 'moplayground.moppo.networks',
    'flatten_model': 'moplayground.moppo.networks',
    'count_params': 'moplayground.moppo.networks',

    # --- utils -----------------------------------------------------------
    'get_nondominated': 'moplayground.utils.pareto',
    'project_to_simplex': 'moplayground.utils.pareto',
    'closest_points': 'moplayground.utils.pareto',
    'corner_tradeoffs': 'moplayground.utils.pareto',
    'get_pareto_statistics': 'moplayground.utils.pareto',
    'hypervolume_from_nondominated': 'moplayground.utils.pareto',
    'sparsity_from_normalized_nondominated': 'moplayground.utils.pareto',
    'MOTrainingPlottingInfo': 'moplayground.utils.plotting',
    'plot_mo_progress': 'moplayground.utils.plotting',
    'plot_pareto': 'moplayground.utils.plotting',
    'plot_sequential_paretos': 'moplayground.utils.plotting',
    'plot_sequential_hypervolume': 'moplayground.utils.plotting',
    'default_coloring': 'moplayground.utils.plotting',
    'FREE3D_POS': 'moplayground.utils.geometry',
    'FREE3D_VEL': 'moplayground.utils.geometry',
    'euler2quat': 'moplayground.utils.geometry',
    'quat2euler': 'moplayground.utils.geometry',
    'rotx': 'moplayground.utils.geometry',
    'roty': 'moplayground.utils.geometry',
    'rotz': 'moplayground.utils.geometry',
    'rotmat': 'moplayground.utils.geometry',
    'quat_mul': 'moplayground.utils.geometry',
    'angle2quat': 'moplayground.utils.geometry',
    'quat_conjugate': 'moplayground.utils.geometry',
    'quat_rotate': 'moplayground.utils.geometry',
    'quat_rotate_vector': 'moplayground.utils.geometry',
    'quat_dist': 'moplayground.utils.geometry',
    'decide_quat': 'moplayground.utils.geometry',
    'extract_yaw': 'moplayground.utils.geometry',
    'solve_transform': 'moplayground.utils.geometry',
    'apply_transform': 'moplayground.utils.geometry',
    'inv_transform': 'moplayground.utils.geometry',
    # re-exported from minimal_mjx, as ``moplayground.utils`` already did
    'read_config': 'minimal_mjx.utils.config',
    'save_metrics': 'minimal_mjx.utils.plotting',
    'save_video': 'minimal_mjx.utils.plotting',
}

__all__ = sorted(set(_EXPORTS) | set(_SUBPACKAGES))


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module, attr = target if isinstance(target, tuple) else (target, name)
    imported = importlib.import_module(module)
    return imported if attr is None else getattr(imported, attr)


def __dir__():
    return __all__


if TYPE_CHECKING:  # static analysers / IDE completion
    from minimal_mjx.utils.config import read_config as read_config
    from minimal_mjx.utils.plotting import save_metrics as save_metrics, save_video as save_video

    from . import envs as envs, eval as eval, learning as learning, moppo as moppo, utils as utils
    from .envs.create import create_environment as create_environment
    from .envs.dmcontrol.ant import MOAnt as MOAnt
    from .envs.dmcontrol.cheetah import MOCheetah as MOCheetah
    from .envs.dmcontrol.hopper import MOHopper as MOHopper
    from .envs.dmcontrol.humanoid import MOHumanoid as MOHumanoid
    from .envs.dmcontrol.interface import (
        AntInterface as AntInterface,
        CheetahInterface as CheetahInterface,
        HopperInterface as HopperInterface,
        HumanoidInterface as HumanoidInterface,
        WalkerInterface as WalkerInterface,
    )
    from .envs.dmcontrol.walker import MOWalker as MOWalker
    from .envs.generic.mobase import Multi2SingleObjective as Multi2SingleObjective, MultiObjectiveBase as MultiObjectiveBase
    from .envs.locomotion.bruce import interface_westwood as bruce_interface
    from .envs.locomotion.bruce.navigait import Bruce as Bruce
    from .envs.locomotion.control.bezier import Leg as Leg, P1Bezier as P1Bezier, bezier_basis_matrix as bezier_basis_matrix
    from .envs.locomotion.control.gait import MIN_SWING_PHASE as MIN_SWING_PHASE, GaitLibrary as GaitLibrary
    from .envs.locomotion.generic.bipedal import BipedalBase as BipedalBase
    from .envs.locomotion.generic.navigait import NaviGait as NaviGait
    from .eval.pareto import (
        compute_fronts as compute_fronts,
        get_amor_fronts as get_amor_fronts,
        get_morlax_fronts as get_morlax_fronts,
        get_pareto_rollout as get_pareto_rollout,
    )
    from .eval.rollout import rollout_policy as rollout_policy
    from .learning.inference import (
        get_num_objectives as get_num_objectives,
        load_amor_networks as load_amor_networks,
        load_hypernetwork_inference_fn as load_hypernetwork_inference_fn,
        load_hypernetworks as load_hypernetworks,
        load_make_amor_inference_fn as load_make_amor_inference_fn,
        load_mo_policy as load_mo_policy,
    )
    from .learning.training import (
        create_training_directory as create_training_directory,
        mo_wrapper as mo_wrapper,
        setup_amor as setup_amor,
        setup_morlax as setup_morlax,
        train_policy as train_policy,
    )
    from .learning.wrappers import (
        MultiObjectiveEpisodeWrapper as MultiObjectiveEpisodeWrapper,
        MultiObjectiveEvalWrapper as MultiObjectiveEvalWrapper,
    )
    from .moppo.acting import (
        Evaluator as Evaluator,
        MultiObjectiveTransition as MultiObjectiveTransition,
        actor_step as actor_step,
        generate_unroll as generate_unroll,
    )
    from .moppo.amor import AMORTrainingState as AMORTrainingState
    from .moppo.amor import train as train_amor
    from .moppo.factory import (
        AMORNetworks as AMORNetworks,
        FeedForwardHypernetwork as FeedForwardHypernetwork,
        MORLAXNetworks as MORLAXNetworks,
        make_amor_inference_fn as make_amor_inference_fn,
        make_amor_networks as make_amor_networks,
        make_amor_policy_network as make_amor_policy_network,
        make_amor_value_network as make_amor_value_network,
        make_hypernetwork as make_hypernetwork,
        make_hypernetwork_inference_fn as make_hypernetwork_inference_fn,
        make_morlax_networks as make_morlax_networks,
    )
    from .moppo.losses import (
        AMORNetworkParams as AMORNetworkParams,
        MORLAXNetworkParams as MORLAXNetworkParams,
        compute_amor_loss as compute_amor_loss,
        compute_mo_gae as compute_mo_gae,
        compute_morlax_loss as compute_morlax_loss,
    )
    from .moppo.morlax import MOTrainingState as MOTrainingState, sample_preferences as sample_preferences
    from .moppo.morlax import train as train_morlax
    from .moppo.networks import (
        MLP as MLP,
        ActorCriticHypernet as ActorCriticHypernet,
        DualA2CHypernet as DualA2CHypernet,
        FakeHypernet as FakeHypernet,
        Hypernet as Hypernet,
        HypernetMLP as HypernetMLP,
        count_params as count_params,
        flatten_model as flatten_model,
    )
    from .utils.geometry import (
        FREE3D_POS as FREE3D_POS,
        FREE3D_VEL as FREE3D_VEL,
        angle2quat as angle2quat,
        apply_transform as apply_transform,
        decide_quat as decide_quat,
        euler2quat as euler2quat,
        extract_yaw as extract_yaw,
        inv_transform as inv_transform,
        quat2euler as quat2euler,
        quat_conjugate as quat_conjugate,
        quat_dist as quat_dist,
        quat_mul as quat_mul,
        quat_rotate as quat_rotate,
        quat_rotate_vector as quat_rotate_vector,
        rotmat as rotmat,
        rotx as rotx,
        roty as roty,
        rotz as rotz,
        solve_transform as solve_transform,
    )
    from .utils.pareto import (
        closest_points as closest_points,
        corner_tradeoffs as corner_tradeoffs,
        get_nondominated as get_nondominated,
        project_to_simplex as project_to_simplex,
        get_pareto_statistics as get_pareto_statistics,
        hypervolume_from_nondominated as hypervolume_from_nondominated,
        sparsity_from_normalized_nondominated as sparsity_from_normalized_nondominated,
    )
    from .utils.plotting import (
        MOTrainingPlottingInfo as MOTrainingPlottingInfo,
        default_coloring as default_coloring,
        plot_mo_progress as plot_mo_progress,
        plot_pareto as plot_pareto,
        plot_sequential_hypervolume as plot_sequential_hypervolume,
        plot_sequential_paretos as plot_sequential_paretos,
    )
