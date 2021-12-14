import sympy
from TF.models_tfq.vqc_layers import Pooling_Layer, VQC_Layer_Lockwood
from TF.models_tfq.utils import encoding_ops_lockwood
import tensorflow as tf
from tensorflow import keras
import tensorflow_quantum as tfq
import cirq
import numpy as np
from random import sample

# TODO documentation
class VQC_Model(keras.Model):

    def __init__(self,  num_qubits, 
                        num_layers, 
                        activation='linear', 
                        scale=None,
                        hybrid=False,
                        encoding_ops = encoding_ops_lockwood,
                        readout_op = 'expval',
                        layertype = VQC_Layer_Lockwood):
        super(VQC_Model, self).__init__()
        
        if hybrid and not (scale is None and readout_op is None):
            raise ValueError("Hybrid netwok can not be initialized when readout_op or scale is given.")

        self.circuit = cirq.Circuit()

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.qubits = cirq.GridQubit.rect(1, self.num_qubits)
        self.activation=activation
        self.Layer = layertype
        self.vqc_layers = []
        self.pool_layers = []
        self.encoding_ops = encoding_ops
        inputs = sympy.symbols(f'inputs0:{num_qubits}')

        # input part
        self.circuit.append([encoding_ops(inputs[i], qubit) for i, qubit in enumerate(self.qubits)])

        self.circuit += self.create_circuit()

        if hybrid:
            self.scale = keras.layers.Dense(2, activation=activation) 
            self.readout_op = [cirq.Z(self.qubits[i]) for i in range(num_qubits)]
        else: 
            self.scale = scale
            if readout_op == 'expval':
                self.readout_op = [ cirq.PauliString(cirq.Z(qubit) for qubit in self.qubits[:2]),
                                    cirq.PauliString(cirq.Z(qubit) for qubit in self.qubits[2:]) ]
            elif readout_op == 'pooling':
                self.pool_layers.append(Pooling_Layer(0, self.activation, source = self.qubits[0], sink=self.qubits[2]))

                if num_qubits == 4:
                    self.pool_layers.append(Pooling_Layer(1, self.activation, source = self.qubits[1], sink=self.qubits[3]))

                self.circuit += [layer.circuit for layer in self.pool_layers]

                self.readout_op = self.build_readout_op()

        self.params = self._get_model_weights()

        self.vqc = tfq.layers.ControlledPQC(self.circuit, self.readout_op, 
            differentiator=tfq.differentiators.ParameterShift())

        symbols = [str(symb) for symb in np.concatenate([self._get_param_symbols(), inputs])]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])

    def _get_model_weights(self):
        weight_values= np.expand_dims(np.concatenate([layer.get_trainable_weights() for layer in [*self.vqc_layers, *self.pool_layers]]), axis=0)
        return tf.Variable(initial_value=weight_values, 
                                dtype='float32', trainable=True, name='weights')

    def _get_param_symbols(self):
        return np.concatenate([layer.get_params() for layer in [*self.vqc_layers, *self.pool_layers]])

    def call(self, inputs):
        batch_dim = tf.gather(tf.shape(inputs), 0)

        # tile to batch dimension
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_params = tf.tile(self.params, multiples=[batch_dim, 1])

        joined_vars = tf.concat([tiled_up_params, inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        x = self.vqc([tiled_up_circuits, joined_vars])

        if(self.scale is not None):
            x = self.scale(x)
        return x

    def _reparameterize(self, weights):
        return keras.activations.sigmoid(weights) * 2. * np.pi  if self.activation == 'sigmoid' else weights

    def create_circuit(self):
        circuit = cirq.Circuit()

        for idx in range(self.num_layers):
            layer = self.Layer(self.qubits, idx, self.activation)
            self.vqc_layers.append(layer)
            circuit += layer.circuit
        return circuit

    def build_readout_op(self):
        return [cirq.Z(self.qubits[i]) for i in range(self.num_qubits-2, self.num_qubits)]
        