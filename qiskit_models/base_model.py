from abc import ABC, abstractmethod
from qiskit_machine_learning.neural_networks import CircuitQNN
import tensorflow as tf
from tensorflow import keras
import numpy as np
from qiskit import Aer
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.utils import QuantumInstance


# VQC as Keras Model
class VQC_Model(keras.Model, ABC):

    def __init__(self,  num_qubits, 
        num_layers, 
        activation='linear', 
        scale=None, 
        pooling='v1', 
        device=Aer.get_backend('qasm_simulator'),
        shots=10):

        super(VQC_Model, self).__init__()

        self.num_qubits = num_qubits
        self.num_layers = num_layers


        self.activation=activation 

        self.pooling = pooling

        self.circuit = QuantumCircuit(self.num_qubits, 2)
        self.encode_data()
        self.create_circuit()
        self.build_pooling_layer()
        self.build_readout_op()

        self.w_var = tf.Variable(
            initial_value=np.random.uniform(0, 1, (self.num_weights + self.num_pool_weights)), 
            dtype="float32", 
            trainable=True, 
            name="weights"
        )

        qi = QuantumInstance(device, shots=shots)

        self.qnn = CircuitQNN(circuit=self.circuit,
                    input_params=self.input_params,
                    weight_params=self.w,
                    quantum_instance=qi)

        self.scale = scale

    def call(self, inputs):
        x = self.qnn.forward(inputs, self.reparameterize(self.w_var))

        x = self._interprete_probabilities(x)

        if self.scale is not None:
            x = self.scale(x)
        return x

    # Calculates VQCs Gradients
    # TODO: check if there is a more efficient way
    def backward(self, inputs):
        _, grads = self.qnn.backward(inputs, self.derive_reparameterize(self.w_var))

        # Limit gradients to measured qubits only and sum up
        grads = np.sum(grads[:,:4], axis=0)

        # sum batch
        grads = np.sum(grads, axis=0)

        return grads

    def encode_data(self):
        self.input_params = ParameterVector('input', self.num_qubits)
        for i, angle in enumerate(self.input_params):
            self.circuit.rx(angle, i)
            self.circuit.rz(angle, i)

    def reparameterize(self, weights):
        return keras.activations.sigmoid(weights) * 2. * np.pi  if self.activation == 'sigmoid' else weights

    def derive_reparameterize(self, weights):
        return keras.activations.sigmoid(weights)*(1-keras.activations.sigmoid(weights)) * 2. * np.pi if self.activation == 'sigmoid' else weights

    # Base pooling
    def _pool(self, source, sink, symbols):
        self.circuit.rx(symbols[0], source)
        self.circuit.ry(symbols[1], source)
        self.circuit.rz(symbols[2], source)

        self.circuit.rx(symbols[3], sink)
        self.circuit.ry(symbols[4], sink)
        self.circuit.rz(symbols[5], sink)

        self.circuit.cx(source, sink)

    # using pooling as in reference implementation
    # Uses 6 symbols
    def pool_v1(self, source, sink, symbols):
        self._pool(source, sink, symbols)

        self.circuit.rx(-symbols[3], sink)
        self.circuit.ry(-symbols[4], sink)
        self.circuit.rz(-symbols[5], sink)


    # pooling approach using 9 parameters
    # Use different parameters for last sink rotation
    def pool_v2(self, source, sink, symbols):
        self._pool(source, sink, symbols)

        self.circuit.rx(-symbols[6], sink)
        self.circuit.ry(-symbols[7], sink)
        self.circuit.rz(-symbols[8], sink)


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

    # Interprete output of CircuitQNN
    @abstractmethod
    def _interprete_probabilities(self, x):
        pass


# VQC as described in Lockwood/Si Paper (20 Parameters)
class Small_VQC_Model(VQC_Model, ABC):

    def create_circuit(self):
        self.num_weights = self.num_qubits*self.num_layers

        self.w = ParameterVector('w', self.num_weights)
        
        for idx in range(self.num_layers):
            self.vqc_layer(symbols=self.w[idx*self.num_qubits : (idx+1)*self.num_qubits])


    def vqc_layer(self, symbols):
        # Create entanglement
        for idx in range(1, self.num_qubits):
            self.circuit.cx(idx, idx-1)

        # Apply qubit rotations
        for idx in range(self.num_qubits):
            self.circuit.rx(symbols[idx], idx)
            self.circuit.ry(0.5 * np.pi, idx)
            self.circuit.rz(np.pi, idx)


# full parameterized VQC as in reference implementation (https://github.com/lockwo/quantum_computation)
class Full_Param_VQC_Model(VQC_Model, ABC):

    def create_circuit(self):
        self.num_weights = self.num_qubits*self.num_layers*3

        self.w = ParameterVector('w', self.num_weights)

        for idx in range(self.num_layers):
            self.vqc_layer(symbols=self.w[idx*self.num_qubits*3 : (idx+1)*self.num_qubits*3])


    def vqc_layer(self, symbols):
        # Create entanglement0
        for idx in range(1, self.num_qubits):
            self.circuit.cx(idx, idx-1)

        # Apply qubit rotations
        for idx in range(self.num_qubits):
            weight_symbols = symbols[idx*3:(idx+1)*3]
            self.circuit.rx(weight_symbols[0], idx)
            self.circuit.ry(weight_symbols[1], idx)
            self.circuit.rz(weight_symbols[2], idx)

