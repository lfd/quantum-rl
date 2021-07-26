from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
import tensorflow_quantum as tfq
import cirq
import numpy as np

class VQC_Model_Base(keras.Model, ABC):

    def __init__(self,  num_qubits, 
                        num_layers, 
                        activation='linear', 
                        scale=None,
                        hybrid=False):
        super(VQC_Model_Base, self).__init__()
        
        if hybrid and scale is not None:
            raise ValueError("Hybrid netwok can not be initialized when scale is given.")

        self.circuit = cirq.Circuit()

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.qubits = cirq.GridQubit.rect(1, self.num_qubits)
        self.activation=keras.layers.Activation(activation)

        # the vqc-weights. To be overritten in subclass
        self.w = None

        self.circuit += self.create_circuit()

        if hybrid:
            self.scale = keras.layers.Dense(2, activation=activation) 
            self.readout_op = [cirq.Z(self.qubits[i]) for i in range(num_qubits)]
        else:
            self.scale = scale
            self.readout_op = self.build_readout_op()

        self.vqc = tfq.layers.ControlledPQC(self.circuit, self.readout_op, 
            differentiator=tfq.differentiators.ParameterShift())

    def encode_data(self, input, asTensor=True):
        circuit = cirq.Circuit()
        for i, angle in enumerate(input):
            angle = angle.numpy()
            circuit.append(self._encoding_ops(angle, self.qubits[i]))
        if asTensor:
            return tfq.convert_to_tensor([circuit])
        else:
            return circuit

    def call(self, inputs, training=False):
        weights = self._reparameterize(self.w)

        # tile weights to batch size
        weights = tf.tile(weights, multiples=[inputs.shape[0], 1])

        x = tfq.convert_to_tensor([self.encode_data(input, asTensor=False) for input in inputs])
        
        x = self.vqc([x, weights])
        if(self.scale is not None):
            x = self.scale(x)
        return x

    def _reparameterize(self, weights):
        return self.activation(weights) * 2. * np.pi

    @abstractmethod
    def create_circuit(self):
        raise NotImplementedError

    @abstractmethod
    def _encoding_ops(self, input, qubit):
        raise NotImplementedError

    @abstractmethod
    def build_readout_op(self):
        raise NotImplementedError