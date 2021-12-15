# configs/cartpole-dnn.py

import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.initializers import GlorotUniform

## Environment
env = gym.make('CartPole-v0')
val_env = gym.make('CartPole-v0')

inputs = keras.Input(shape=(4,), dtype=tf.float64, name='input')

## Model
policy_model = keras.Sequential([inputs,
                                Dense(8, kernel_initializer=GlorotUniform(seed=27)),
                                Activation('relu'),
                                Dense(2, kernel_initializer=GlorotUniform(seed=27)),
                                Activation('linear')])

target_model = keras.Sequential([inputs,
                                Dense(8),
                                Activation('relu'),
                                Dense(2),
                                Activation('linear')])

target_model.set_weights(policy_model.get_weights())

## Optimization
loss = keras.losses.MSE
lr=0.1
lr_steps=4000
optimizer = keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.PolynomialDecay(lr, lr_steps, end_learning_rate=lr*0.01))

## Hyperparameter
num_steps = 50000
train_after = 1000
train_every = 10
update_every_start = 30
update_every_end = 500
update_every_duration = 35000
validate_every = 100
batch_size=32
replay_capacity=50000
num_val_trials = 1
num_val_steps_acceptance = 25
acceptance_threshold = 196
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_duration=20000
gamma=0.99
