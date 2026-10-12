import matplotlib
matplotlib.use('Agg')
import moplayground as mop
import minimal_mjx as mm
import argparse

def get_handle_params(cfg: mop.MOConfig):
    match cfg.algorithm:
        case 'ppo':
            return mm.setup_ppo, cfg.ppo_params, cfg.ppo_network_params
        case 'morlax':
            return mop.setup_morlax, cfg.ppo_params | cfg.sampling_params, cfg.morlax_network_params
        case 'amor':
            return mop.setup_amor, cfg.ppo_params | cfg.sampling_params, cfg.amor_network_params
        case _:
            return Exception('Unknown Algorithm')
        
def get_progress(cfg: mop.MOConfig):
    match cfg.algorithm:
        case 'ppo':
            return mm.Progress
        case _:
            return mop.MOProgress

def main(config_path):
    TRAIN_KWARGS = {}
    EVAL_KWARGS  = {}

    # Read in config
    cfg = mop.MOConfig(config_path)

    # Environment-specific config handling...
    match cfg.env_name:
        case 'BRUCE':
            EVAL_KWARGS = {'manual_speed': [0.0, 0.0, 0.0], 'idealistic': True}

    # Create environments
    print('Training', args.config_path)
    env_kwargs = dict(
        env_name     = cfg.env_name,
        backend      = 'jnp',
        # gaitlib_path = cfg.gaitlib_path(cfg),
    )
    env, env_cfg = mop.create_environment(env_params=cfg.env_params, **env_kwargs, **TRAIN_KWARGS)
    eval_env, _  = mop.create_environment(env_params=cfg.env_params, **env_kwargs, **EVAL_KWARGS)
    if cfg.algorithm == 'ppo':
        env         = mop.Multi2SingleObjective(env, cfg.default_scalarization)
        eval_env    = mop.Multi2SingleObjective(eval_env, cfg.default_scalarization)

    # Create the run directory and save the config (adds git_hash unless name is 'test')
    cfg.save(cfg.run_dir / 'config.yaml', warn_github_changes=False)

    run = mm.initialize_wandb(
        name    = str(cfg.run_dir).replace('/', ''),
        entity  = 'njanwani-gatech',
        project = 'MO-Playground-2',
        config  = cfg.cfg.to_dict()
    )
    
    handle_params, train_fn_params, network_factory_params = get_handle_params(cfg)
    progress = get_progress(cfg)

    mm.train(
        env                     = env,
        eval_env                = eval_env,
        run_dir                 = cfg.run_dir,
        handle_params           = handle_params,
        train_fn_params         = train_fn_params,
        network_factory_params  = network_factory_params,
        progress_cls            = progress,
        run                     = run,
        resume                  = False,
        overwrite               = cfg.name == 'test'
    )


if __name__ == '__main__':
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="YAML config to train with")
    args = parser.parse_args()
    main(args.config_path)