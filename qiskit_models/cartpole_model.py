from abc import ABC
import numpy as np
from qiskit_models.base_model import Small_VQC_Model, Full_Param_VQC_Model, VQC_Model

class Cartpole_Model(VQC_Model, ABC):

    def build_readout_op(self):
        self.circuit.measure(2,0)
        self.circuit.measure(3,1)

    def build_pooling_layer(self):
        if self.pooling == 'v1':
            pool = self.pool_v1
            self.num_pool_weights = 12
        else:
            pool = self.pool_v2
            self.num_pool_weights = 18

        self.w.resize(self.num_weights + self.num_pool_weights)

        pool(source=0, sink=2, 
                symbols=self.w[self.num_weights:self.num_weights+int(self.num_pool_weights*0.5)])
        pool(source=1, sink=3, 
                symbols=self.w[self.num_weights+int(self.num_pool_weights*0.5): ])


    # Limit Probabilities to measured qubits only and
    # Sum probablities for measuring 1 at qubit2 [index 0] 1 at qubit3 [index 1]
    def _interprete_probabilities(self, x):
        return np.transpose([x[:,2]+x[:,3], x[:,1]+x[:,3]])

class Small_Cartpole_Model(Small_VQC_Model, Cartpole_Model):
    pass


class Full_Param_Cartpole_Model(Full_Param_VQC_Model, Cartpole_Model):
    pass


