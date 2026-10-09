import argparse
import moplayground as mop
from moplayground import config
from matplotlib import pyplot as plt
import jax

parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="YAML config of the run to evaluate")
args = parser.parse_args()

cfg = config.load(args.config_path)
env, env_config = mop.create_environment(
    env_name     = config.env_name(cfg),
    env_params   = config.env_params(cfg),
    backend      = config.backend(cfg, for_training=True),
    gaitlib_path = config.gaitlib_path(cfg),
    manual_speed = True,
    idealistic   = True
)
get_fronts = {
    'morlax': mop.get_morlax_fronts,
    'amor':   mop.get_amor_fronts,
}[config.algorithm(cfg)]
num_objectives = config.num_objectives(cfg)
rewards_over_iters, directives = get_fronts(
    run_dirs        = config.run_dir(cfg),
    network_params  = config.network_params(cfg),
    num_objectives  = num_objectives,
    rng             = jax.random.PRNGKey(95),
    env             = env,
    N_STEPS         = 500,
    NUM_ENVS        = 1024,
    save_results    = True,
    only_final      = True
)
print(rewards_over_iters.shape)
nd_idx = mop.get_nondominated(rewards_over_iters[-1], epsilon=10)

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
    objective   = config.objective_labels(cfg),
)

plt.show()
# fig.savefig(f'output/plots/{config.env_name(cfg)}_frontier.pdf')
