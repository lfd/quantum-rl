from abc import ABC, abstractmethod
from numpy import random
import tensorflow as tf
from tensorflow import keras
import tensorflow_quantum as tfq
import cirq
import numpy as np
import sympy

from tfq_models.vqc_model import VQC_Model_Base

# VQC as Keras Model
class VQC_Model(VQC_Model_Base, ABC):

    def _encoding_ops(self, input, qubit):
        return cirq.rx(input).on(qubit)

    def build_readout_op(self):
        return [cirq.Z(self.qubits[i]) for i in range(self.num_qubits-2, self.num_qubits)]

    @abstractmethod
    def create_circuit(self):
        pass

class Small_VQC_Model(VQC_Model):

    def create_circuit(self):
        circuit = cirq.Circuit()

        num_weights = self.num_qubits*self.num_layers

        weight_symbols = sympy.symbols(f'weights0:{num_weights}')
        self.w = tf.Variable(initial_value=np.random.uniform(0, 1, (1, num_weights)), dtype="float32", trainable=True, name="weights")

        for idx in range(self.num_layers):
            circuit += self._vqc_layer(symbols=weight_symbols[idx*self.num_qubits : (idx+1)*self.num_qubits])

        return circuit

    def _vqc_layer(self, symbols):
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
            circuit += self._vqc_layer(symbols=weight_symbols[idx*self.num_qubits*3 : (idx+1)*self.num_qubits*3])

        return circuit

    def _vqc_layer(self, symbols):
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
