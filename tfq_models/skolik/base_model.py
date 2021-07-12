from abc import ABC, abstractmethod
from numpy import random
import tensorflow as tf
from tensorflow import keras
import tensorflow_quantum as tfq
import cirq
import numpy as np
import sympy

# VQC as Keras Model
class VQC_Model(keras.Model, ABC):

    def __init__(self,  num_qubits, 
                        num_layers, 
                        activation='linear', 
                        scale=None, 
                        hybrid=False):
        super(VQC_Model, self).__init__()

        if hybrid and scale is None:
            raise ValueError("Hybrid netwok can not be initialized when scale is given.")

        circuit = cirq.Circuit()

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.qubits = cirq.GridQubit.rect(1, self.num_qubits)
        self.activation=keras.layers.Activation(activation)

        circuit += self.create_circuit()

        if hybrid:
            self.scale = keras.layers.Dense(2, activation=activation) 
            readout_op = [cirq.Z(self.qubits[i]) for i in range(num_qubits)]
        else:
            # when no hybrid model is used measure output of last two qubits
            self.scale = scale
            readout_op = [cirq.Z(self.qubits[i]) for i in range(self.num_qubits-2, self.num_qubits)]

        self.vqc = tfq.layers.ControlledPQC(circuit, readout_op, 
            differentiator=tfq.differentiators.ParameterShift())

        self.scale = scale

    def call(self, inputs, trainig=False):
        x = [ self.encode_data(input, asTensor=True) for input in inputs ]
        weights = self.reparameterize(self.w)
        x = tf.concat([self.vqc([i, weights]) for i in x], axis=0)
        if(self.scale is not None):
            x = self.scale(x)
        return x

    def encode_data(self, input, asTensor=True):
        circuit = cirq.Circuit()
        for i, angle in enumerate(input):
            angle = angle.numpy()
            circuit.append(cirq.rx(angle).on(self.qubits[i]))
        if asTensor:
            return tfq.convert_to_tensor([circuit])
        else:
            return circuit

    def reparameterize(self, weights):
        return self.activation(weights) * 2. * np.pi
        
    @abstractmethod
    def create_circuit(self):
        pass

    @abstractmethod
    def vqc_layer(self, symbols):
        pass


class Small_VQC_Model(VQC_Model):

    def create_circuit(self):
        circuit = cirq.Circuit()

        num_weights = self.num_qubits*self.num_layers

        weight_symbols = sympy.symbols(f'weights0:{num_weights}')
        self.w = tf.Variable(initial_value=np.random.uniform(0, 1, (1, num_weights)), dtype="float32", trainable=True, name="weights")

        for idx in range(self.num_layers):
            circuit += self.vqc_layer(symbols=weight_symbols[idx*self.num_qubits : (idx+1)*self.num_qubits])

        return circuit

    def vqc_layer(self, symbols):
        circuit = cirq.Circuit()

        # Apply qubit rotations
        for idx in range(self.num_qubits):
            circuit.append(self._generate_gate(idx, symbols))

        # Create entanglement
        for i in range(self.num_qubits-1):
            for j in range(i+1, self.num_qubits):
                circuit.append(cirq.CZ.on(self.qubits[i], self.qubits[j]))

        return circuit

    def _generate_gate(self, qubit_idx, symbols):
        rotation_gates = [cirq.rx(symbols[qubit_idx]).on(self.qubits[qubit_idx]), 
                        cirq.ry(symbols[qubit_idx]).on(self.qubits[qubit_idx]),
                        cirq.rz(symbols[qubit_idx]).on(self.qubits[qubit_idx])]
        return random.choice(rotation_gates)


class Big_VQC_Model(VQC_Model):

    def create_circuit(self):
        circuit = cirq.Circuit()

        num_weights = self.num_qubits*self.num_layers*3

        weight_symbols = sympy.symbols(f'weights0:{num_weights}')
        self.w = tf.Variable(initial_value=np.random.uniform(0, 1, (1, num_weights)), dtype="float32", trainable=True, name="weights")

        for idx in range(self.num_layers):
            circuit += self.vqc_layer(symbols=weight_symbols[idx*self.num_qubits*3 : (idx+1)*self.num_qubits*3])

        return circuit

    def vqc_layer(self, symbols):
        circuit = cirq.Circuit()

        # Apply qubit rotations
        for idx in range(len(self.qubits)):
            weight_symbols = symbols[idx*3:(idx+1)*3]
            circuit.append([cirq.rx(weight_symbols[0]).on(self.qubits[idx]), 
                            cirq.ry(weight_symbols[1]).on(self.qubits[idx]), 
                            cirq.rz(weight_symbols[2]).on(self.qubits[idx])])

        # Create entanglement
        for i in range(self.num_qubits-1):
            for j in range(i+1, self.num_qubits):
                circuit.append(cirq.CZ.on(self.qubits[i], self.qubits[j]))

        return circuit
