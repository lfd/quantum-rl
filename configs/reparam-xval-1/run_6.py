# configs/cartpole-dnn.py

from wrappers import CartPoleEncoding, ToDoubleTensor

from models import Scale, pool, vqc_layer
import gym
import pennylane as qml
import tensorflow as tf
from tensorflow import keras

# Setup Keras to use 64-bit floats (required by PennyLane)
keras.backend.set_floatx('float64')

def model():

    NUM_QUBITS = 4
    NUM_LAYERS = 3
    
    qml.enable_tape()

    device = qml.device('default.qubit.tf', wires=NUM_QUBITS)

    @qml.qnode(device, interface='tf', diff_method='backprop')
    def circuit(inputs, layer_weights, pooling_weights):

        # Encode input state
        for idx, angle in enumerate(inputs):
            qml.RX(angle, wires=idx)
            qml.RZ(angle, wires=idx)

        for weights in layer_weights:
            vqc_layer(weights, nonlinearity='sigmoid')

        pool(pooling_weights[0], source=0, sink=2, nonlinearity='sigmoid')
        pool(pooling_weights[1], source=1, sink=3, nonlinearity='sigmoid')

        return [qml.expval(qml.PauliZ(i)) for i in (2, 3)]

    inputs = keras.Input(shape=(NUM_QUBITS,), dtype=tf.float64, name='input')

    vqc = qml.qnn.KerasLayer(
        qnode=circuit,
        weight_shapes={
            # layer, qubit, rotation angles
            'layer_weights': (NUM_LAYERS, NUM_QUBITS, 3),
            # layer, source/sink, rotation angles
            'pooling_weights': (2, 2, 3),
        },
        output_dim=2,
        name='VQC'
    )(inputs)

    outputs = Scale(name='scale')(vqc)

    return keras.Model(inputs=inputs, outputs=outputs)


## Environment
env = gym.make('CartPole-v0')
env = CartPoleEncoding(env)
env = ToDoubleTensor(env)

val_env = gym.make('CartPole-v0')
val_env = CartPoleEncoding(val_env)
val_env = ToDoubleTensor(val_env)

## Model
policy_model = model()
target_model = model()
target_model.set_weights(policy_model.get_weights())

## Optimization
optimizer = keras.optimizers.Adam(learning_rate=1e-1)
loss = keras.losses.MSE

## Hyperparameter
num_steps = 25000
train_after = 500
train_every = 1
update_every = int(1e0)
validate_every = 500
batch_size = 32
replay_capacity=25000
gamma = 0.99
num_val_steps = 5000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_duration = 20000
