This validation run explores the viability of reparameterizing the learned
VQC weights by using a sigmoid nonlinearity.

## Hyperparameters

| Config            | `learning_rate` | `update_every` | Return (top/last) |
| ----------------- | ---------------:| --------------:| :---------------: |
| [run_0](run_0.py) |            1e-4 |            1e4 | 27.54 / 24.82     |
| [run_1](run_1.py) |            1e-4 |            1e2 | 12.79 / 9.346     |
| [run_2](run_2.py) |            1e-4 |            1e0 | 109.5 / 9.375     |
| [run_3](run_3.py) |            1e-2 |            1e4 | 94.21 / 21.48     |
| [run_4](run_4.py) |            1e-2 |            1e2 | 164.9 / 112.2     |
| [run_5](run_5.py) |            1e-2 |            1e0 | 198.9 / 45.45     |
| [run_6](run_6.py) |             1e0 |            1e4 | 21.43 / 9.387     |
| [run_7](run_7.py) |             1e0 |            1e2 | 41.92 / 9.342     |
| [run_8](run_8.py) |             1e0 |            1e0 | 38.83 / 9.381     |