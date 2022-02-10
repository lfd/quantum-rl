# configs/cartpole-dnn.py

import torch
import gym
import copy

from PT.models_qiskit.vqc_model import VQC_Model
from wrappers import ScaledContEncodingCP
from PT.models import Scale

## Environment
env = gym.make('CartPole-v0')
env = ScaledContEncodingCP(env)

val_env = gym.make('CartPole-v0')
val_env = ScaledContEncodingCP(val_env)

## Model
policy_model = VQC_Model(num_qubits=4, num_layers=5, 
                    in_scale=Scale(4),
                    out_scale=Scale(2),
                    layertype='skolik',
                    encoding_ops='skolik')

target_model = copy.deepcopy(policy_model)

## Optimization
optimizer = torch.optim.Adam(policy_model.qnn.parameters(), lr=1e-3)
optimizer_input = torch.optim.Adam(policy_model.in_scale.parameters(), lr=1e-3)
optimizer_output = torch.optim.Adam(policy_model.out_scale.parameters(), lr=1e-1)

loss = torch.nn.MSELoss()

## Hyperparameter
num_steps = 50000
train_after = 1000
train_every = 10
update_every = 30
validate_every = 100
batch_size = 16
replay_capacity=50000
gamma = 0.99
num_val_trials = 1
num_val_steps_acceptance = 25
acceptance_threshold = 196
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_duration = 20000

framework = 'pytorch'

