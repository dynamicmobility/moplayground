def create_environment(env_name, env_params, backend='jnp', gaitlib_path=None, **env_kwargs):
    """Instantiate a MO-Playground environment.

    Constructs one of the registered multi-objective environments
    (``MOCheetah``, ``MOHopper``, ``MOAnt``, ``MOWalker``, ``MOHumanoid``,
    ``NaviGait``).

    Args:
        env_name: Registered environment name, e.g. ``'MOCheetah'``.
        env_params: ``ConfigDict`` of environment parameters (the
            ``env_config`` section of a run config).
        backend: ``'jnp'`` (JAX, for training) or ``'np'`` (NumPy, for
            evaluation/rollout).
        gaitlib_path: Gait library path. Required for ``NaviGait``.
        **env_kwargs: Extra keyword arguments forwarded to the environment
            constructor. Currently only consumed by ``NaviGait`` (Bruce).

    Returns:
        Tuple ``(env, env_params)`` where ``env`` is the constructed
        environment instance and ``env_params`` is the ``ConfigDict`` of
        environment parameters passed in.

    Raises:
        Exception: If ``env_name`` does not match a registered environment.
    """
    common_kwargs = {
        'backend': backend,
        'env_params': env_params
    }
    
    match env_name:
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
                gaitlib_path    = gaitlib_path,
                gait_type       = 'P2',
                animate         = False,
                **common_kwargs,
                **env_kwargs
            )
        case _:
            raise Exception(f'Unknown enviornment {env_name}.')
    return env, env_params