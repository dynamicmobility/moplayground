from minimal_mjx.learning.startup import read_config
from moplayground.envs.create import create_environment
from moplayground.eval.pareto import run_experiments
from moplayground.utils.plotting import plot_squential_paretos
import jax

config = read_config()
env, env_config = create_environment(
    config, 
    for_training = True, 
    manual_speed = True,
    idealistic   = True
)
rewards_over_iters, directives = run_experiments(
    config    = config,
    rng       = jax.random.PRNGKey(95),
    env       = env,
    N_STEPS   = 500,
    NUM_ENVS  = 1024
)
fig, ax = plot_squential_paretos(
    ax_titles   = ['titles'],
    paretos     = [rewards_over_iters[-1]],
    directives  = [directives[-1]],
    # objectives = config.env_params
)
fig.savefig(f'output/plots/{config['env']}_frontier.pdf')
