from minimal_mjx.learning.startup import create_config_dict

def create_environment(config, for_training=False):
    env_params = create_config_dict(config['env_config'])
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
                idealistic      = True,
                animate         = False,
                **common_kwargs
            )
        case _:
            raise Exception(f'Unknown enviornment {config["env"]}.')
    return env, env_params