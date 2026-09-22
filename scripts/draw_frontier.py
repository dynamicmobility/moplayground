from minimal_mjx.utils import read_config
import moplayground as mop
from moplayground.eval.pareto import run_experiments
from matplotlib import pyplot as plt
import jax

config = read_config()
env, env_config = mop.create_environment(
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
nd_idx = mop.get_nondominated(rewards_over_iters[-1], epsilon=10)
num_objectives = len(config.env_config.reward.optimization.objectives)

if(num_objectives == 3):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
elif(num_objectives == 2):
    fig, ax = plt.subplots()
else:
    raise ValueError("Can only plot 2 or 3 objective pareto frontiers")
    

ax = mop.plot_pareto(
    ax          = ax,
    pareto      = rewards_over_iters[-1],
    colors      = mop.default_coloring(directives[-1]),
    objective   = config.env_config.reward.optimization.labels,
)

plt.show()
# fig.savefig(f'output/plots/{config['env']}_frontier.pdf')
