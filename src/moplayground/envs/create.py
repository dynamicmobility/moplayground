import minimal_mjx as mm

def create_environment(config, for_training=False, **env_kwargs):
    """Instantiate a MO-Playground environment from a config.

    Uses ``config['env']`` to construct one of the registered
    multi-objective environments (``MOCheetah``, ``MOHopper``, ``MOAnt``,
    ``MOWalker``, ``MOHumanoid``, ``NaviGait``).

    Args:
        config: Config dict (typically loaded from a YAML file in ``config/``)
            with at least ``env`` (str) and ``env_config`` (dict) entries.
            ``backend`` is read when ``for_training`` is False.
        for_training: If True, force the JAX (``'jnp'``) backend regardless of
            ``config['backend']``. Set this when building the environment for
            GPU training; leave False for evaluation/rollout.
        **env_kwargs: Extra keyword arguments forwarded to the environment
            constructor. Currently only consumed by ``NaviGait`` (Bruce).

    Returns:
        Tuple ``(env, env_params)`` where ``env`` is the constructed
        environment instance and ``env_params`` is the resolved
        ``ConfigDict`` of environment parameters.

    Raises:
        Exception: If ``config['env']`` does not match a registered
            environment name.
    """
    env_params = mm.utils.config.create_config_dict(config['env_config'])
    backend = 'jnp' if for_training else config['backend']
    common_kwargs = {
        'backend': backend,
        'env_params': env_params
    }
    
    match config['env']:
        case 'MOCheetah':
            from moplayground.envs.dmcontrol.cheetah import MOCheetah
            env = MOCheetah(**common_kwargs)
        case 'MOHopper':
            from moplayground.envs.dmcontrol.hopper import MOHopper
            env = MOHopper(**common_kwargs)
        case 'MOAnt':
            from moplayground.envs.dmcontrol.ant import MOAnt
            env = MOAnt(**common_kwargs)
        case 'MOWalker':
            from moplayground.envs.dmcontrol.walker import MOWalker
            env = MOWalker(**common_kwargs)
        case 'MOHumanoid':
            from moplayground.envs.dmcontrol.humanoid import MOHumanoid
            env = MOHumanoid(**common_kwargs)
        case 'NaviGait':
            from moplayground.envs.locomotion.bruce.navigait import Bruce
            env = Bruce(
                gaitlib_path    = config['gaitlib_path'],
                gait_type       = 'P2',
                animate         = False,
                **common_kwargs,
                **env_kwargs
            )
        case _:
            raise Exception(f'Unknown enviornment {config["env"]}.')
    return env, env_params