import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense

from tensorflow.keras.initializers import GlorotUniform
## Environment
env = gym.make('Blackjack-v0')

val_env = gym.make('Blackjack-v0')

inputs = keras.Input(shape=(3,), dtype=tf.float64, name='input')

## Model
policy_model = keras.Sequential([inputs,
                                Dense(6, kernel_initializer=GlorotUniform(seed=27)),
                                Activation('relu'),
                                Dense(2, kernel_initializer=GlorotUniform(seed=27)),
                                Activation('linear')])

target_model = keras.Sequential([inputs,
                                Dense(6),
                                Activation('relu'),
                                Dense(2),
                                Activation('linear')])
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

