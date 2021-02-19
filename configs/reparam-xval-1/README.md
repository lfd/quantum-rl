This validation run builds on the results of 
[the previous run](../reparam-xval-0/README.md). 

We used the best-performing learning rate (`1e-2`) and refine the validation 
grid around this value. We also fix `update_every` to `1e0`, thereby effectively
disabling the target network. We conjecture that this is viable, since 
distributional shift is primarily a property of the environment and most likely
absent on `CartPole-v0` due to the small state space.  

Since peak performance (val. ret.) was reached on epoch 4 (= 4000 steps), we 
reduce total training time by half (25,000 steps) and explore different 
`epsilon_duration` settings, specifically 20,000 (equal to the previous run),
10,000 (equal to the previous run in terms of the fraction of total training
time) and 5,000 (explorative). Conversely, we double the frequency of 
validation epochs for finer temporal resolution.

## Hyperparameters

| Config            | `learning_rate` | `epsilon_duration` | Return (top/last) |
| ----------------- | ---------------:| --------------:| :---------------: |
| [run_0](run_0.py) |            1e-3 |          20000 | tbd |
| [run_1](run_1.py) |            1e-3 |          10000 | tbd |
| [run_2](run_2.py) |            1e-3 |           5000 | tbd |
| [run_3](run_3.py) |            1e-2 |          20000 | tbd |
| [run_4](run_4.py) |            1e-2 |          10000 | tbd |
| [run_5](run_5.py) |            1e-2 |           5000 | tbd |
| [run_6](run_6.py) |            1e-1 |          20000 | tbd |
| [run_7](run_7.py) |            1e-1 |          10000 | tbd |
| [run_8](run_8.py) |            1e-1 |           5000 | tbd |