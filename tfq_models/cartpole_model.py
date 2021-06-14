from abc import ABC, abstractmethod
import numpy as np 
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
from math import exp
from tfq_models.base_model import VQC_Model, Small_VQC_Model, Full_Param_VQC_Model


class Small_Cartpole_Model(Small_VQC_Model):
    def __init__(self, num_qubits, num_layers, activation='linear', scale=None):
        super(Small_Cartpole_Model, self).__init__(num_qubits, num_layers, activation, scale)

    def build_readout_op(self):
        return [cirq.Z(self.qubits[i]) for i in (2,3)]

    def build_pooling_layer(self):
        circuit = cirq.Circuit()

        num_pool_weights = 2*2*3
        pool_symbols = sympy.symbols(f'pool0:{num_pool_weights}')
        self.p = tf.Variable(initial_value=np.random.uniform(0, 1, (1, num_pool_weights)), dtype="float32", trainable=True, name="pool_weights")

        circuit += self.pool(source=self.qubits[0], sink=self.qubits[2], 
                            symbols=pool_symbols[:6])
        circuit += self.pool(source=self.qubits[1], sink=self.qubits[3], 
                            symbols=pool_symbols[6:])

        return circuit


class Full_Param_Cartpole_Model(Full_Param_VQC_Model):
    def __init__(self,  num_qubits, num_layers, activation='linear', scale=None):
        super(Full_Param_Cartpole_Model, self).__init__(num_qubits, num_layers, activation, scale)

    def build_readout_op(self):
        return [cirq.Z(self.qubits[i]) for i in (2,3)]

    def build_pooling_layer(self):
        circuit = cirq.Circuit()

        num_pool_weights = 2*2*3
        pool_symbols = sympy.symbols(f'pool0:{num_pool_weights}')
        self.p = tf.Variable(initial_value=np.random.uniform(0, 1, (1, num_pool_weights)), dtype="float32", trainable=True, name="pool_weights")

        circuit += self.pool(source=self.qubits[0], sink=self.qubits[2], 
                            symbols=pool_symbols[:6])
        circuit += self.pool(source=self.qubits[1], sink=self.qubits[3], 
                            symbols=pool_symbols[6:])

        return circuit

