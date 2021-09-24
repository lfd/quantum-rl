# configs/cartpole-dnn.py

from tfq_models.vqc_model import VQC_Model
from wrappers import BlackjackEncodingContinous

import gym
from tensorflow import keras

# Setup Keras to use 32-bit floats
keras.backend.set_floatx('float32')

## Environment
env = gym.make('Blackjack-v0')
env = BlackjackEncodingContinous(env)

val_env = gym.make('Blackjack-v0')
val_env = BlackjackEncodingContinous(val_env)

## Model
policy_model = VQC_Model(num_qubits=3, num_layers=3, 
                    hybrid=True,
                    readout_op=None)
target_model = VQC_Model(num_qubits=3, num_layers=3, 
                    hybrid=True,
                    readout_op=None)
target_model.set_weights(policy_model.get_weights())

## Optimization
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss = keras.losses.MSE

## Hyperparameter
num_steps = 50000
train_after = 1000
train_every = 1
update_every = 500
validate_every = 1000
batch_size = 32
replay_capacity=50000
gamma = 0.99
num_val_steps = 5000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_duration = 20000

