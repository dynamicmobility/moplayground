from pathlib import Path
from minimal_mjx.eval.simulate import make_dummy_inference_fn
from minimal_mjx.learning.startup import read_config
from minimal_mjx.learning.inference import rollout
from minimal_mjx.utils.plotting import save_video, save_metrics
from moplayground.envs.create import create_environment

def main(save_dir):
    config            = read_config()
    env, env_params   = create_environment(config)
    inference_fn      = make_dummy_inference_fn(env, mode='zero')
    
    T = env.dt * 500
    frames, reward_plotter, data_plotter, info_plotter = rollout(
        inference_fn    = inference_fn,
        env             = env,
        T               = T,
        height          = 640,
        width           = 480
    )

    save_video(
        frames    = frames,
        dt        = env.dt,
        path      = save_dir / f'{config['env']}-simulate.mp4'
    )
    save_metrics(
        plotter   = reward_plotter,
        path      = save_dir / f'{config['env']}-reward.pdf'
    )

if __name__ == '__main__':
    save_dir = Path('output/videos')
    main(save_dir)