from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
import tensorflow_quantum as tfq
import cirq
import numpy as np
import sympy

from tfq_models.vqc_model import VQC_Model_Base

# VQC as Keras Model
class VQC_Model(VQC_Model_Base, ABC):

    def __init__(self,  num_qubits, 
                        num_layers, 
                        activation='linear', 
                        scale=None, 
                        pooling='v1', 
                        hybrid=False):
        super(VQC_Model, self).__init__( num_qubits, 
                                            num_layers, 
                                            activation, 
                                            scale, 
                                            hybrid)
        if hybrid and pooling:
            raise ValueError("Hybrid netwok can not be initialized when pooling is given.")

        if pooling:
            self.pooling = pooling
            self.circuit += self.build_pooling_layer()
            self.vqc = tfq.layers.ControlledPQC(self.circuit, self.readout_op, 
                differentiator=tfq.differentiators.ParameterShift())

    def _encoding_ops(self, input, qubit):
        return [cirq.rx(input).on(qubit), cirq.rz(input).on(qubit)]
        
    # Base pooling
    def _pool(self, source, sink, symbols):
        circuit = cirq.Circuit()

        circuit.append([cirq.rx(symbols[0]).on(source), 
                        cirq.ry(symbols[1]).on(source), 
                        cirq.rz(symbols[2]).on(source)])

        circuit.append([cirq.rx(symbols[3]).on(sink), 
                        cirq.ry(symbols[4]).on(sink), 
                        cirq.rz(symbols[5]).on(sink)])

        circuit.append(cirq.CNOT(control=source, target=sink))

        return circuit

    # using pooling as in reference implementation
    # Uses 6 symbols
    def pool_v1(self, source, sink, symbols):
        circuit = cirq.Circuit()

        circuit += self._pool(source, sink, symbols)

        circuit.append([cirq.rx(-symbols[0]).on(sink), 
                        cirq.ry(-symbols[1]).on(sink), 
                        cirq.rz(-symbols[2]).on(sink)])

        return circuit

    # pooling approach using 9 parameters
    # Use different parameters for last sink rotation
    def pool_v2(self, source, sink, symbols):
        circuit = cirq.Circuit()

        circuit += self._pool(source, sink, symbols[:6])

        circuit.append([cirq.rx(-symbols[6]).on(sink), 
                        cirq.ry(-symbols[7]).on(sink), 
                        cirq.rz(-symbols[8]).on(sink)])

        return circuit

    @abstractmethod
    def build_readout_op(self):
        pass
        
    @abstractmethod
    def create_circuit(self):
        pass

    @abstractmethod
    def build_pooling_layer(self):
        pass

# VQC as described in Lockwood/Si Paper (20 Parameters)
class Small_VQC_Model(VQC_Model, ABC):

    def create_circuit(self):
        circuit = cirq.Circuit()

        num_weights = self.num_qubits*self.num_layers

        weight_symbols = sympy.symbols(f'weights0:{num_weights}')
        self.w = tf.Variable(initial_value=np.zeros((1, num_weights)), dtype="float32", trainable=True, name="weights")

        for idx in range(self.num_layers):
            circuit += self._vqc_layer(symbols=weight_symbols[idx*self.num_qubits : (idx+1)*self.num_qubits])

        return circuit

    def _vqc_layer(self, symbols):
        circuit = cirq.Circuit()

        # Create entanglement
        for idx in range(1, len(self.qubits)):
            circuit.append(cirq.CNOT(control=self.qubits[idx], target=self.qubits[idx-1]))

        # Apply qubit rotations
        for idx in range(len(self.qubits)):
            circuit.append([cirq.rx(symbols[idx]).on(self.qubits[idx]), 
                            cirq.ry(0.5 * np.pi).on(self.qubits[idx]), 
                            cirq.rz(np.pi).on(self.qubits[idx])])

        return circuit


# full parameterized VQC as in reference implementation (https://github.com/lockwo/quantum_computation)
class Full_Param_VQC_Model(VQC_Model, ABC):

    def create_circuit(self):
        circuit = cirq.Circuit()

        num_weights = self.num_qubits*self.num_layers*3

        weight_symbols = sympy.symbols(f'weights0:{num_weights}')
        self.w = tf.Variable(initial_value=np.zeros((1, num_weights)), dtype="float32", trainable=True, name="weights")

        for idx in range(self.num_layers):
            circuit += self._vqc_layer(symbols=weight_symbols[idx*self.num_qubits*3 : (idx+1)*self.num_qubits*3])

        return circuit

    def _vqc_layer(self, symbols):
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
