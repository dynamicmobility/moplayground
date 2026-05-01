from minimal_mjx.utils import read_config
from moplayground.envs.create import create_environment
from moplayground.eval.pareto import run_experiments
from moplayground.utils.plotting import plot_pareto
from moplayground.utils.pareto import get_nondominated
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
nd_idx = get_nondominated(rewards_over_iters[-1], epsilon=10)
fig, ax = plt.subplots()
ax = plot_pareto(
    ax          = ax,
    pareto      = rewards_over_iters[-1],
    directive   = directives[-1],
    objective   = config.env_config.reward.optimization.objectives,
    nondominated= nd_idx
)

plt.show()
# fig.savefig(f'output/plots/{config['env']}_frontier.pdf')
