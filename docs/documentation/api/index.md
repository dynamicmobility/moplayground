# API reference

Every public function and class in `moplayground`, generated from the docstrings in
`src/`. Each page covers one subpackage, split by the module the code lives in. Use the
search bar to jump to any name.

| Subpackage | Modules | What it covers |
| --- | --- | --- |
| [`moplayground.envs`](envs.md) | `create`, `generic.mobase`, `dmcontrol.*`, `locomotion.*` | The multi-objective base class, the registered environments, and `create_environment` |
| [`moplayground.learning`](learning.md) | `training`, `inference`, `wrappers` | Setup functions for `mm.train`, loading checkpoints and policies, multi-objective Brax wrappers |
| [`moplayground.moppo`](moppo.md) | `morlax`, `amor`, `losses`, `factory`, `acting`, `networks` | The MORLAX and AMOR trainers and their shared components |
| [`moplayground.eval`](eval.md) | `pareto` | Batched rollouts over many tradeoffs to compute Pareto fronts |
| [`moplayground.utils`](utils.md) | `config`, `pareto`, `plotting`, `geometry` | `MOConfig`, Pareto statistics, progress callbacks and plots, rotations |

## Top-level shortcuts

Every public name is also importable from the package itself, so these are the same
function:

```python
import moplayground as mop

mop.load_mo_policy
mop.learning.inference.load_mo_policy
```

The two trainers share the name `train`, so they are exported as `mop.train_morlax`
and `mop.train_amor`. Concrete environments load lazily, so `import moplayground`
does not load every MuJoCo model.

| Shortcut | Defined in |
| --- | --- |
| `mop.create_environment`, `mop.MultiObjectiveBase`, `mop.Multi2SingleObjective`, `mop.MOCheetah`, ... | [`envs`](envs.md) |
| `mop.setup_morlax`, `mop.setup_amor`, `mop.mo_wrapper` | [`learning.training`](learning.md#moplayground.learning.training) |
| `mop.load_mo_policy`, `mop.load_hypernetworks`, `mop.load_amor_networks`, `mop.get_last_model`, ... | [`learning.inference`](learning.md#moplayground.learning.inference) |
| `mop.train_morlax`, `mop.sample_preferences`, `mop.train_amor` | [`moppo.morlax`](moppo.md#moplayground.moppo.morlax), [`moppo.amor`](moppo.md#moplayground.moppo.amor) |
| `mop.compute_morlax_loss`, `mop.compute_amor_loss`, `mop.compute_mo_gae` | [`moppo.losses`](moppo.md#moplayground.moppo.losses) |
| `mop.make_morlax_networks`, `mop.make_amor_networks`, `mop.make_amor_inference_fn`, ... | [`moppo.factory`](moppo.md#moplayground.moppo.factory) |
| `mop.generate_unroll`, `mop.Evaluator`, `mop.MultiObjectiveTransition` | [`moppo.acting`](moppo.md#moplayground.moppo.acting) |
| `mop.get_morlax_fronts`, `mop.get_amor_fronts`, `mop.compute_fronts` | [`eval.pareto`](eval.md) |
| `mop.MOConfig` | [`utils.config`](utils.md#moplayground.utils.config) |
| `mop.get_nondominated`, `mop.get_pareto_statistics`, ... | [`utils.pareto`](utils.md#moplayground.utils.pareto) |
| `mop.MOProgress`, `mop.plot_pareto`, ... | [`utils.plotting`](utils.md#moplayground.utils.plotting) |
