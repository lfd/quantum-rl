# configs/cartpole-dnn.py

from tfq_models.vqc_layers import Pooling_Layer_V1
from tfq_models.vqc_model import VQC_Model
from wrappers import CartPoleEncodingMix

import gym
from tensorflow import keras
from models import SingleScale

# Setup Keras to use 32-bit floats
keras.backend.set_floatx('float32')

## Environment
env = gym.make('CartPole-v0')
env = CartPoleEncodingMix(env)

val_env = gym.make('CartPole-v0')
val_env = CartPoleEncodingMix(val_env)

## Model
policy_model = VQC_Model(num_qubits=4, num_layers=3, 
                            scale=SingleScale(name="scale"),
                            readout_op=None,
                            pooling_layertype=Pooling_Layer_V1)
target_model = VQC_Model(num_qubits=4, num_layers=3, 
                            scale=SingleScale(name="scale"),
                            readout_op=None,
                            pooling_layertype=Pooling_Layer_V1)
target_model.set_weights(policy_model.get_weights())

## Optimization
optimizer = keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.PolynomialDecay(5e-2, 2000, end_learning_rate=5e-4))
optimizer_output = keras.optimizers.Adam(learning_rate=0.1)
loss = keras.losses.MSE

## Hyperparameter
num_steps = 50000
train_after = 1000
train_every = 10
update_every_start = 30
update_every_end = 500
update_every_duration = 35000
validate_every = 500
batch_size = 32
replay_capacity=50000
gamma = 0.99
num_val_trials = 25
num_val_steps_acceptance = 25
acceptance_threshold = 196
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_duration = 20000

