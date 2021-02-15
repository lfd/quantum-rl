This validation run explores the viability of reparameterizing the learned
VQC weights by using a sigmoid nonlinearity.

## Hyperparameters

| Config            | `learning_rate` | `update_every` |
| ----------------- | ---------------:| --------------:|
| [run_0](run_0.py) |            1e-4 |            1e4 |
| [run_1](run_1.py) |            1e-4 |            1e2 |
| [run_2](run_2.py) |            1e-4 |            1e0 |
| [run_3](run_3.py) |            1e-2 |            1e4 |
| [run_4](run_4.py) |            1e-2 |            1e2 |
| [run_5](run_5.py) |            1e-2 |            1e0 |
| [run_6](run_6.py) |             1e0 |            1e4 |
| [run_7](run_7.py) |             1e0 |            1e2 |
| [run_8](run_8.py) |             1e0 |            1e0 |