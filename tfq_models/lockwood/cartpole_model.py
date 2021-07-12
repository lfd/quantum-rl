from abc import ABC
import numpy as np 
import tensorflow as tf
import cirq
import sympy
from math import exp
from tfq_models.lockwood.base_model import Small_VQC_Model, Full_Param_VQC_Model, VQC_Model

class Cartpole_Model(VQC_Model, ABC):

    def build_readout_op(self):
        return [cirq.Z(self.qubits[i]) for i in (2,3)]

    def build_pooling_layer(self):
        circuit = cirq.Circuit()

        if self.pooling == 'v1':
            pool = self.pool_v1
            num_pool_weights = 12
        else:
            pool = self.pool_v2
            num_pool_weights = 18

        pool_symbols = sympy.symbols(f'pool0:{num_pool_weights}')
        p = tf.Variable(initial_value=np.random.uniform(0, 1, (1, num_pool_weights)), dtype="float32", trainable=True, name="pool_weights")

        circuit += pool(source=self.qubits[0], sink=self.qubits[2], 
                            symbols=pool_symbols[:int(num_pool_weights*0.5)])
        circuit += pool(source=self.qubits[1], sink=self.qubits[3], 
                            symbols=pool_symbols[int(num_pool_weights*0.5):])

        self.w = tf.concat([p, self.w], axis=1)

        return circuit

class Small_Cartpole_Model(Small_VQC_Model, Cartpole_Model):
    pass


class Full_Param_Cartpole_Model(Full_Param_VQC_Model, Cartpole_Model):
    pass


