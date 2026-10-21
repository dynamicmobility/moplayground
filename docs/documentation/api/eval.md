# moplayground.eval

Evaluating trained runs over many tradeoffs at once. Each function returns the
per-objective episode returns for every checkpoint, from which Pareto fronts and
their statistics ([`utils.pareto`](utils.md#moplayground.utils.pareto)) are computed.

::: moplayground.eval.pareto
