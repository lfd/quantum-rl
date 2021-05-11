"""Utilities for creating VQC models"""

import numpy as np
import pennylane as qml
import tensorflow as tf
from tensorflow import keras

import tensorflow_quantum as tfq
import cirq
import sympy

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

# VQC as Keras Model (with tfq)
class QVC_Model(keras.Model):

    def __init__(self,  num_qubits, num_layers):
        inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)
        super(QVC_Model, self).__init__()

        self.qubits = cirq.GridQubit.rect(1, num_qubits)
        
        circuit = self.create_circuit(num_qubits, num_layers)
        readout_op = [cirq.Z(self.qubits[i]) for i in (2,3)]

        self.vqc = tfq.layers.PQC(circuit, readout_op)

        
    def call(self, inputs, trainig=False):
        x = [ self.encode_data(input, asTensor=True) for input in inputs ]
        x = tf.concat([self.vqc(i) for i in x], axis=0)
        return x
        

    def create_circuit(self, num_qubits, num_layers):
        circuit = cirq.Circuit()

        symbols = sympy.symbols('param0:' + str(num_qubits*num_layers*3 + 2*3*3)) # 4 qubits * 3 layers * 3 weights + 2 * 3 * 3 pooling = 36 + 18 = 54

        for idx in range(num_layers):
            circuit += self.vqc_layer(symbols=symbols[idx*num_qubits*num_layers : (idx+1)*num_qubits*num_layers], nonlinearity='sigmoid')

        circuit += self.pool(source=self.qubits[0], sink=self.qubits[2], 
                            symbols=symbols[num_qubits*num_layers*3 : num_qubits*num_layers*3+9], nonlinearity='sigmoid')
        circuit += self.pool(source=self.qubits[1], sink=self.qubits[3], 
                            symbols=symbols[num_qubits*num_layers*3+9 : ], nonlinearity='sigmoid')
        return circuit


    def vqc_layer(self, symbols, nonlinearity=None):
        
        circuit = cirq.Circuit()

        # Create entanglement
        for idx in range(1, len(self.qubits)):
            circuit.append(cirq.CNOT(control=self.qubits[idx], target=self.qubits[idx-1]))

        # Apply qubit rotations
        for idx in range(len(self.qubits)):
            weight_symbols = symbols[idx*3:(idx+1)*3]
            circuit.append([cirq.rx(weight_symbols[0]).on(self.qubits[idx]), 
                            cirq.ry(weight_symbols[1]).on(self.qubits[idx]), 
                            cirq.rz(weight_symbols[2]).on(self.qubits[idx])])

        return circuit


    def pool(self, source, sink, symbols, nonlinearity=None):
        circuit = cirq.Circuit()
        
        circuit.append([cirq.rx(symbols[0]).on(source), 
                        cirq.ry(symbols[1]).on(source), 
                        cirq.rz(symbols[2]).on(source)])

        circuit.append([cirq.rx(symbols[3]).on(sink), 
                        cirq.ry(symbols[4]).on(sink), 
                        cirq.rz(symbols[5]).on(sink)])

        circuit.append(cirq.CNOT(control=source, target=sink))

        circuit.append([cirq.rx(-symbols[6]).on(sink), 
                        cirq.ry(-symbols[7]).on(sink), 
                        cirq.rz(-symbols[8]).on(sink)])

        return circuit


    # Directional encoding of classical data -> quantum data
    def encode_data(self, input, asTensor=True):
        circuit = cirq.Circuit()
        for i, angle in enumerate(input): 
            angle = 0 if angle < 0 else np.pi
            circuit.append(cirq.rx(angle).on(self.qubits[i]))
            circuit.append(cirq.rz(angle).on(self.qubits[i]))
        if asTensor:
            return tfq.convert_to_tensor([circuit])
        else:
            return circuit


class Scale(keras.layers.Layer):

    def __init__(self, name=None):
        super(Scale, self).__init__(name=name)


    def build(self, input_shape):

        self.factor = self.add_weight(
            name='factor',
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(1.),
            trainable=True,
            dtype=keras.backend.floatx()
        )


    def call(self, inputs):
        return self.factor * inputs


class ConstantScale(keras.layers.Layer):

    def __init__(self, factor, name=None):
        super(ConstantScale, self).__init__(name=name)
        self.factor = factor

    def call(self, inputs):
        return self.factor * inputs

class ExpScale(keras.layers.Layer):

    def __init__(self, name=None):
        super(ExpScale, self).__init__(name=name)


    def build(self, input_shape):

        self.factor = self.add_weight(
            name='factor',
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(1.),
            trainable=True,
            dtype=keras.backend.floatx()
        )


    def call(self, inputs):
        return tf.exp(self.factor) * inputs
