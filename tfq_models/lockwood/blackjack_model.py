from abc import ABC
import cirq
import sympy
import tensorflow as tf
import numpy as np
from tfq_models.lockwood.base_model import Small_VQC_Model, Full_Param_VQC_Model, VQC_Model

class Blackjack_Model(VQC_Model, ABC):

    def build_readout_op(self):
        return [cirq.Z(self.qubits[i]) for i in (1,2)]

    def build_pooling_layer(self):
        circuit = cirq.Circuit()

        if self.pooling == 'v1':
            pool = self.pool_v1
            num_pool_weights = 6
        else:
            pool = self.pool_v2
            num_pool_weights = 9

        pool_symbols = sympy.symbols(f'pool0:{num_pool_weights}')

        circuit += pool(source=self.qubits[0], sink=self.qubits[2], 
                            symbols=pool_symbols)

        pooling_weights = np.zeros((1, num_pool_weights), dtype='float32')
        weight_values = tf.concat([pooling_weights, self.w], axis=1) 
        self.w = tf.Variable(initial_value=weight_values, 
                                dtype='float32', trainable=True, name='weights')
        
        return circuit

class Small_Blackjack_Model(Small_VQC_Model, Blackjack_Model):
    pass


class Full_Param_Blackjack_Model(Full_Param_VQC_Model, Blackjack_Model):
    pass

