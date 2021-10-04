import numpy as np
from qiskit_models.base_model import VQC_Model

class Cartpole_Model(VQC_Model):

    def build_readout_op(self):
        self.circuit.measure(2,0)
        self.circuit.measure(3,1)

    def build_pooling_layer(self):
        self.num_pool_weights = 12

        self.w.resize(self.num_weights + self.num_pool_weights)

        self._pool(source=0, sink=2, 
                symbols=self.w[self.num_weights:self.num_weights+int(self.num_pool_weights*0.5)])
        self._pool(source=1, sink=3, 
                symbols=self.w[self.num_weights+int(self.num_pool_weights*0.5): ])

    # Limit Probabilities to measured qubits only and
    # calculate expectation value on each qubit 
    # Exp = p(0) - p(1)
    def _interprete_probabilities(self, x):
        exp_qubit2 = x[:,0]+x[:,1] - (x[:,2]+x[:,3])
        exp_qubit3 = x[:,0]+x[:,2] - (x[:,1]+x[:,3])
        return np.transpose([exp_qubit2, exp_qubit3])

