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
        case _:
            raise Exception(f'Unknown enviornment {config["env"]}.')
    return env, env_params