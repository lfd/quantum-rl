This validation run builds on the results of 
[the previous run](../reparam-xval-1/README.md).

While reparameterization does help with the convergence of VQC parameters, the
limiting factor is now the `scale` parameters, which approach their targets 
slowly, yet monotonically. To this end, we try another reparameterization where
the scaling factor is estimated on log-scale (i.e. being exponentiated before
multiplying with the VQC measuremets). 

We again validate around learning rates ranging from `1e-3` to `1e-1` to account
for potential changes in learning dynamics. Since shorter training times yielded
good (if not better results) in the previous training run, we maintain the 
reduced number of training steps (i.e., 25,000) and explore even more aggressive
epsilon decay schedules.

## Hyperparameters

| Config            | `learning_rate` | `epsilon_duration` | Return (top/last) |
| ----------------- | ---------------:| --------------:| :---------------: |
| [run_0](run_0.py) |            1e-3 |          10000 | tbd               |
| [run_1](run_1.py) |            1e-3 |           5000 | tbd               |
| [run_2](run_2.py) |            1e-3 |           1000 | tbd               |
| [run_3](run_3.py) |            1e-2 |          10000 | tbd               |
| [run_4](run_4.py) |            1e-2 |           5000 | tbd               |
| [run_5](run_5.py) |            1e-2 |           1000 | tbd               |
| [run_6](run_6.py) |            1e-1 |          10000 | tbd               |
| [run_7](run_7.py) |            1e-1 |           5000 | tbd               |
| [run_8](run_8.py) |            1e-1 |           1000 | tbd               |