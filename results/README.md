# Results

- Detailed report: [rl_navigation_experiment_report.md](rl_navigation_experiment_report.md)

Key takeaways:

- `objective_ablation_v1` shows that `efficiency_v1` is not a viable main objective on the critical upstream benchmark.
- `efficiency_gain_sweep_v1` shows that weak safety shaping works better than strong safety or energy shaping.
- The current recommended objective preset is `efficiency_v2`, which corresponds to:
  - `energy_cost_gain = 0.0`
  - `safety_cost_gain = 0.25`
- Some configurations look materially better at intermediate checkpoints than at the final checkpoint, so best-checkpoint evaluation should be part of the formal protocol.
