# configs/cartpole-dnn.py

import gym
from tensorflow import keras

def model():

    inputs = keras.Input(shape=(4,))
    hidden = keras.layers.Dense(64, activation='relu')(inputs)
    outputs = keras.layers.Dense(2)(hidden)

    return keras.Model(inputs=inputs, outputs=outputs)

## Environment
env = gym.make('CartPole-v0')
val_env = gym.make('CartPole-v0')

## Model
policy_model = model()
target_model = model()
target_model.set_weights(policy_model.get_weights())

## Optimization
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss = keras.losses.MSE

## Hyperparameter
num_steps = 100000
train_after = 1000
train_every = 1
update_every = 500
validate_every = 500
batch_size = 32
replay_capacity=50000
gamma = 0.99
num_val_steps = 10000
epsilon_start = 1.0
epsilon_end = 0.02
epsilon_duration = 10000