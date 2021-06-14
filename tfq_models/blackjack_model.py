import cirq
import sympy
import tensorflow as tf
import numpy as np
from tfq_models.base_model import Small_VQC_Model, Full_Param_VQC_Model

class Small_Blackjack_Model(Small_VQC_Model):

    def __init__(self,  num_qubits, num_layers, activation='linear', scale=None):
        super(Small_Blackjack_Model, self).__init__(num_qubits, num_layers, activation, scale)

    def build_readout_op(self):
        return [cirq.Z(self.qubits[i]) for i in (1,2)]

    def build_pooling_layer(self):
        circuit = cirq.Circuit()

        num_pool_weights = 1*2*3
        pool_symbols = sympy.symbols(f'pool0:{num_pool_weights}')
        self.p = tf.Variable(initial_value=np.random.uniform(0, 1, (1, num_pool_weights)), dtype="float32", trainable=True, name="pool_weights")

        circuit += self.pool(source=self.qubits[0], sink=self.qubits[2], 
                            symbols=pool_symbols[:9])

        return circuit

class Full_Param_Blackjack_Model(Full_Param_VQC_Model):

    def __init__(self,  num_qubits, num_layers, activation='linear', scale=None):
        super(Full_Param_Blackjack_Model, self).__init__(num_qubits, num_layers, activation, scale)

    def build_readout_op(self):
        return [cirq.Z(self.qubits[i]) for i in (1,2)]

    def build_pooling_layer(self):
        circuit = cirq.Circuit()

        num_pool_weights = 1*2*3
        pool_symbols = sympy.symbols(f'pool0:{num_pool_weights}')
        self.p = tf.Variable(initial_value=np.random.uniform(0, 1, (1, num_pool_weights)), dtype="float32", trainable=True, name="pool_weights")

        circuit += self.pool(source=self.qubits[0], sink=self.qubits[2], 
                            symbols=pool_symbols[:9])

        return circuit
