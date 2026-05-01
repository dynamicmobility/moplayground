import numpy as np
import jax
import minimal_mjx as mm
import moplayground as mop

def rollout_policy(
    env,
    config,
    tradeoff = None,
    T=10.0,
    camera = 'track',
    width  = 1080,
    height = 720
):
    """Roll out a trained policy on ``env`` and return rendered frames.

    Loads the policy specified by ``config`` (single-objective if
    ``config.mo2so.enabled`` else multi-objective with the given
    ``tradeoff``), JIT-compiles the inference function, and delegates the
    actual stepping/rendering to ``minimal_mjx.eval.rollout_policy``.

    Args:
        env: The environment to roll out in (typically built with
            ``create_environment``).
        config: Run config (ConfigDict) for the trained policy.
        tradeoff: Per-objective weighting used to condition the
            multi-objective policy. Defaults to all-ones with length equal
            to the number of objectives in ``config``. Ignored when
            ``config.mo2so.enabled``.
        T: Rollout duration in seconds.
        camera: MuJoCo camera name to render from.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Whatever ``minimal_mjx.eval.rollout_policy`` returns — at the time
        of writing a 4-tuple ``(frames, reward_plotter, ...)``.
    """
    if tradeoff is None:
        tradeoff = np.ones(len(config.env_config.reward.optimization.objectives))
    if config['mo2so']['enabled']:
        print('Loading single objective policy')
        inference_fn = mm.learning.inference.load_policy(config, deterministic=True)
    else:
        print('Loading multi-objective policy')
        inference_fn = mop.learning.inference.load_mo_policy(
            config          = config,
            tradeoff        = tradeoff,
            deterministic   = True
        )
    
    inference_fn = jax.jit(inference_fn)
    return mm.eval.rollout_policy(
        inference_fn    = inference_fn,
        env             = env,
        T               = T,
        height          = height,
        width           = width,
        camera          = camera
    )