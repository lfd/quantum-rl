from abc import ABC, abstractmethod
from qiskit_machine_learning.neural_networks import CircuitQNN
import tensorflow as tf
from tensorflow import keras
import numpy as np
from qiskit import Aer
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.utils import QuantumInstance


# full parameterized VQC as in Lockwoods reference implementation (https://github.com/lockwo/quantum_computation)
class VQC_Model(keras.Model, ABC):

    def __init__(self,  num_qubits, 
        num_layers, 
        activation='linear', 
        scale=None,
        backend=Aer.get_backend('qasm_simulator'),
        shots=1024):

        super(VQC_Model, self).__init__()

        self.num_qubits = num_qubits
        self.num_layers = num_layers

        self.activation=activation 

        self.circuit = QuantumCircuit(self.num_qubits, 2)
        self.encode_data()
        self.create_circuit()
        self.build_pooling_layer()
        self.build_readout_op()

        self.w_var = tf.Variable(
            initial_value=np.random.uniform(0, 1, (self.num_weights + self.num_pool_weights)), 
            dtype="float64", 
            trainable=True, 
            name="weights"
        )

        qi = QuantumInstance(backend, shots=shots)

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

    # using pooling as in reference implementation
    def _pool(self, source, sink, symbols):
        self.circuit.rx(symbols[0], source)
        self.circuit.ry(symbols[1], source)
        self.circuit.rz(symbols[2], source)

        self.circuit.rx(symbols[3], sink)
        self.circuit.ry(symbols[4], sink)
        self.circuit.rz(symbols[5], sink)

        self.circuit.cx(source, sink)

        self.circuit.rx(-symbols[3], sink)
        self.circuit.ry(-symbols[4], sink)
        self.circuit.rz(-symbols[5], sink)

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

    @abstractmethod
    def build_readout_op(self):
        pass

    @abstractmethod
    def build_pooling_layer(self):
        pass

    # Interprete output of CircuitQNN
    @abstractmethod
    def _interprete_probabilities(self, x):
        pass

