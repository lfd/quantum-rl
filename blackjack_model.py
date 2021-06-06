import cirq
import sympy
from tfq_model import QVC_Model, QVC_Model_full_parameterized

class VQC_Model_Blackjack(QVC_Model):

    def __init__(self,  num_qubits, num_layers, nonlinearity=None, scale=None):
        super(VQC_Model_Blackjack, self).__init__(num_qubits, num_layers, nonlinearity, scale)

    def build_readout_op(self):
        return [cirq.Z(self.qubits[i]) for i in (1,2)]

    def create_circuit(self, num_qubits, num_layers):
        circuit = cirq.Circuit()

        weight_symbols = sympy.symbols('weights0:' + str(num_qubits*num_layers))

        for idx in range(num_layers):
            circuit += self.vqc_layer(symbols=weight_symbols[idx*num_qubits : (idx+1)*num_qubits])

        pool_symbols = sympy.symbols('pool0:' + str(1*2*3))

        circuit += self.pool(source=self.qubits[0], sink=self.qubits[2], 
                            symbols=pool_symbols[:6])

        return circuit

class VQC_Model_Blackjack_full_param(QVC_Model_full_parameterized):

    def __init__(self,  num_qubits, num_layers, nonlinearity=None, scale=None):
        super(VQC_Model_Blackjack_full_param, self).__init__(num_qubits, num_layers, nonlinearity, scale)

    def build_readout_op(self):
        return [cirq.Z(self.qubits[i]) for i in (1,2)]

    def create_circuit(self, num_qubits, num_layers):
        circuit = cirq.Circuit()

        weight_symbols = sympy.symbols('weights0:' + str(num_qubits*num_layers * 3))

        for idx in range(num_layers):
            circuit += self.vqc_layer(symbols=weight_symbols[idx*num_qubits*3 : (idx+1)*num_qubits*3])

        pool_symbols = sympy.symbols('pool0:' + str(1*3*3))

        circuit += self.pool(source=self.qubits[0], sink=self.qubits[2], 
                            symbols=pool_symbols[:9])

        return circuit