# configs/cartpole-dnn.py

from wrappers import CartPoleEncoding, ToDoubleTensorFloat32

from tfq_models.skolik.base_model import VQC_Model
import gym
from tensorflow import keras
from tfq_models.skolik.vqc_layers import VQC_Layer_V2

# Setup Keras to use 32-bit floats
keras.backend.set_floatx('float32')

## Environment
env = gym.make('CartPole-v0')
env = CartPoleEncoding(env)
env = ToDoubleTensorFloat32(env)

val_env = gym.make('CartPole-v0')
val_env = CartPoleEncoding(val_env)
val_env = ToDoubleTensorFloat32(val_env)

## Model
policy_model = VQC_Model(num_qubits=4, num_layers=8, hybrid=True, layertype=VQC_Layer_V2)
target_model = VQC_Model(num_qubits=4, num_layers=8, hybrid=True, layertype=VQC_Layer_V2)
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
num_steps_per_layer = 3000

