from tensorflow import keras
import numpy as np
import pennylane as qml

def _reparameterize(weights, nonlinearity):

    if nonlinearity == 'sigmoid':
        weights = keras.activations.sigmoid(weights) * 2 * np.pi

    return weights


# weight_shape: 2, 3
def vqc_layer(weights, nonlinearity=None):
    
    weights = _reparameterize(weights, nonlinearity)

    # Create entanglement
    for idx in range(1, len(weights)):
        qml.CNOT(wires=[idx - 1, idx])

    # Apply learned qubit rotations
    for idx, theta in enumerate(weights):
        qml.Rot(theta[0], theta[1], theta[2], wires=idx)


def pool(weights, source, sink, nonlinearity=None):
    
    weights = _reparameterize(weights, nonlinearity)

    qml.Rot(weights[0, 0], weights[0, 1], weights[0, 2], wires=source)
    qml.Rot(weights[1, 0], weights[1, 1], weights[1, 2], wires=sink)
    qml.CNOT(wires=[source, sink])
    qml.Rot(weights[1, 0], weights[1, 1], weights[1, 2], wires=sink).inv()