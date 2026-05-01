from minimal_mjx.utils import read_config
from moplayground.envs.create import create_environment
from moplayground.eval.pareto import run_experiments
from moplayground.utils.plotting import plot_pareto
from matplotlib import pyplot as plt
import jax

config = read_config()
env, env_config = create_environment(
    config, 
    for_training = True, 
    manual_speed = True,
    idealistic   = True
)
rewards_over_iters, directives = run_experiments(
    config          = config,
    rng             = jax.random.PRNGKey(95),
    env             = env,
    N_STEPS         = 500,
    NUM_ENVS        = 1024,
    save_results    = True,
    only_final      = True
)
print(rewards_over_iters.shape)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax = plot_pareto(
    ax          = ax,
    pareto      = rewards_over_iters[-1],
    directive   = directives[-1],
    # objective  = config.env_params
)

plt.show()
# fig.savefig(f'output/plots/{config['env']}_frontier.pdf')
