import matplotlib
matplotlib.use('Agg')
import moplayground as mop
import minimal_mjx as mm
import argparse
import functools

def setup_morlax(config):
    """Sets up the morlax training given a config file."""
    from moplayground.moppo import factory, morlax
    general_ppo_params = config.learning_params.ppo_params
    morlax_algo_params = config.learning_params.morlax_params.train_fn_params
    network_params = config.learning_params.morlax_params.network_params
    
    train_fn_params = dict(general_ppo_params) | dict(morlax_algo_params)
    
    network_factory = functools.partial(
        factory.make_morlax_networks,
        **network_params
    )

    train_fn = functools.partial(
        morlax.train, **dict(train_fn_params),
        network_factory=network_factory,
    )
        
    return train_fn, network_factory

def setup_amor(config):
    """Sets up the amor training given a config file."""
    from moplayground.moppo import factory, amor
    general_ppo_params = config.learning_params.base_ppo_params
    amor_algo_params   = config.learning_params.amor_params.train_fn_params
    network_params     = config.learning_params.amor_params.network_params

    train_fn_params = dict(general_ppo_params) | dict(amor_algo_params)

    network_factory = functools.partial(
        factory.make_amor_networks,
        **network_params
    )

    train_fn = functools.partial(
        amor.train, **dict(train_fn_params),
        network_factory=network_factory,
    )

    return train_fn, network_factory

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Config to train on")
    args = parser.parse_args()
    TRAIN_KWARGS = {}
    EVAL_KWARGS  = {}

    # Read in configs
    train_config = mop.read_config(args.config)
    eval_config  = mop.read_config(args.config)

    # Environment-specific config handling...
    match train_config.env:
        case 'BRUCE':
            EVAL_KWARGS = {'manual_speed': [0.0, 0.0, 0.0], 'idealistic': True}

    # Create environments
    print('Training', args.config)
    env, env_cfg = mop.create_environment(train_config, for_training=True, **TRAIN_KWARGS)
    eval_env, _  = mop.create_environment(eval_config, for_training=True, **EVAL_KWARGS)

    match train_config['algorithm'].lower():
        case 'morlax':
            handle_params_fn = setup_morlax
        case 'amor':
            handle_params_fn = setup_amor
        case _algo:
            raise Exception(f'Unknown algorithm {_algo}')
    
    name = train_config['save_dir'] + '/' + train_config['name']
    run = mm.initialize_wandb(
        name    = name.replace('/', ''),
        entity  = 'njanwani-gatech',
        project = 'MO-Playground-2',
        config  = dict(train_config)
    )

    mop.train_policy(
        config=train_config, 
        env=env, 
        eval_env=eval_env, 
        run=run, 
        handle_params=handle_params_fn,
        normalize_observations=train_config.learning_params.ppo_params.normalize_observations
    )

if __name__ == '__main__':
    main()