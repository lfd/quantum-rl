# configs/cartpole-dnn.py

from TF.models_tfq.utils import encoding_ops_lockwood
from TF.models_tfq.vqc_layers import VQC_Layer_Lockwood
from TF.models_tfq.vqc_model import VQC_Model
from wrappers import ScaledDirectionalEncodingCP

import gym
from tensorflow import keras
from TF.models import SingleScale

# Setup Keras to use 32-bit floats
keras.backend.set_floatx('float32')

## Environment
env = gym.make('CartPole-v0')
env = ScaledDirectionalEncodingCP(env)

val_env = gym.make('CartPole-v0')
val_env = ScaledDirectionalEncodingCP(val_env)

## Model
policy_model = VQC_Model(num_qubits=4, num_layers=3, 
                        out_scale=SingleScale(name="scale"),
                        layertype=VQC_Layer_Lockwood,
                        encoding_ops=encoding_ops_lockwood,
                        readout_op='pooling',
                        activation='sigmoid')
target_model = VQC_Model(num_qubits=4, num_layers=3, 
                        out_scale=SingleScale(name="scale"),
                        layertype=VQC_Layer_Lockwood,
                        encoding_ops=encoding_ops_lockwood,
                        readout_op='pooling',
                        activation='sigmoid')

target_model.set_weights(policy_model.get_weights())

## Optimization
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
optimizer_output = keras.optimizers.Adam(learning_rate=1e-1)
loss = keras.losses.MSE

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