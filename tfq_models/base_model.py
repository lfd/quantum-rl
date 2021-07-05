from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
import tensorflow_quantum as tfq
import cirq
import numpy as np
import sympy

# VQC as Keras Model
class VQC_Model(keras.Model, ABC):

    def __init__(self,  num_qubits, num_layers, activation='linear', scale=None, pooling='v1'):
        super(VQC_Model, self).__init__()

        circuit = cirq.Circuit()

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.qubits = cirq.GridQubit.rect(1, self.num_qubits)
        self.activation=keras.layers.Activation(activation)

        self.pooling = pooling

        circuit += self.create_circuit()
        circuit += self.build_pooling_layer()

        readout_op = self.build_readout_op()

        self.vqc = tfq.layers.ControlledPQC(circuit, readout_op, 
            differentiator=tfq.differentiators.ParameterShift())

        self.scale = scale

    def call(self, inputs, trainig=False):
        x = [ self.encode_data(input, asTensor=True) for input in inputs ]
        weights = tf.concat([self.reparameterize(self.p), self.reparameterize(self.w)], axis=1)
        x = tf.concat([self.vqc([i, weights]) for i in x], axis=0)
        if(self.scale is not None):
            x = self.scale(x)
        return x

    def encode_data(self, input, asTensor=True):
        circuit = cirq.Circuit()
        for i, angle in enumerate(input):
            angle = angle.numpy()
            circuit.append(cirq.rx(angle).on(self.qubits[i]))
            circuit.append(cirq.rz(angle).on(self.qubits[i]))
        if asTensor:
            return tfq.convert_to_tensor([circuit])
        else:
            return circuit

    def reparameterize(self, weights):
        return self.activation(weights) * 2. * np.pi

    
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

    @abstractmethod
    def vqc_layer(self, symbols):
        pass


# VQC as described in Lockwood/Si Paper (20 Parameters)
class Small_VQC_Model(VQC_Model, ABC):
    def __init__(self, num_qubits, num_layers, activation='linear', scale=None, pooling='v1'):
        super(Small_VQC_Model, self).__init__(num_qubits, num_layers, activation, scale, pooling)

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
    def __init__(self, num_qubits, num_layers, activation='linear', scale=None, pooling='v1'):
        super(Full_Param_VQC_Model, self).__init__(num_qubits, num_layers, activation, scale, pooling)

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
