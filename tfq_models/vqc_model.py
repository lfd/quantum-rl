from tfq_models.vqc_layers import VQC_Layer_Lockwood
from tfq_models.utils import encoding_ops_lockwood
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
                        readout_op = 'exp_val',
                        pooling_layertype=None,
                        initial_layers=3,
                        p=0,
                        q=3,
                        r=1,
                        update_rate=0,
                        layertype = VQC_Layer_Lockwood):
        super(VQC_Model, self).__init__()
        
        if hybrid and not (scale is None and readout_op is None):
            raise ValueError("Hybrid netwok can not be initialized when readout_op or scale is given.")

        self.circuit = cirq.Circuit()

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.qubits = cirq.GridQubit.rect(1, self.num_qubits)
        self.activation=keras.layers.Activation(activation)
        self.initial_layers=initial_layers
        self.p=p
        self.q=q
        self.r=r
        self.update_rate=update_rate
        self.phase=0
        self.Layer = layertype
        self.vqc_layers = []
        self.pool_layers = []
        self.encoding_ops = encoding_ops

        # the vqc-weights. To be overritten in subclass
        self.w = None

        self.circuit += self.create_circuit()

        if hybrid:
            self.scale = keras.layers.Dense(2, activation=activation) 
            self.readout_op = [cirq.Z(self.qubits[i]) for i in range(num_qubits)]
        else: 
            self.scale = scale
            if readout_op == 'expval':
                self.readout_op = [ cirq.PauliString(cirq.Z(qubit) for qubit in self.qubits[:2]),
                                    cirq.PauliString(cirq.Z(qubit) for qubit in self.qubits[2:]) ]
            else:
                if pooling_layertype is not None:
                    self.pool_layers.append(pooling_layertype(0, self.activation, source = self.qubits[0], sink=self.qubits[2]))

                    if num_qubits == 4:
                        self.pool_layers.append(pooling_layertype(1, self.activation, source = self.qubits[1], sink=self.qubits[3]))

                    self.circuit += [layer.circuit for layer in self.pool_layers]

                self.readout_op = self.build_readout_op()

        self._update_model_weights()

        self.vqc = tfq.layers.ControlledPQC(self.circuit, self.readout_op, 
            differentiator=tfq.differentiators.ParameterShift())

    def encode_data(self, input, asTensor=True):
        circuit = cirq.Circuit()
        for i, angle in enumerate(input):
            angle = angle.numpy()
            circuit.append(self.encoding_ops(angle, self.qubits[i]))
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

    def create_circuit(self):
        layer_circuit = self.add_layers(self.initial_layers)
        return layer_circuit

    def add_layers(self, num):
        circuit = cirq.Circuit()

        for idx in range(num):
            layer = self.Layer(self.qubits, len(self.vqc_layers), self.activation)
            self.vqc_layers.append(layer)
            circuit += layer.circuit
        return circuit
    
    def next_configuration(self):
        self.store_weights_in_layers()
        if len(self.vqc_layers) < self.num_layers:
            self.add_layers(self.p)
            self.freeze_layers()
            self.rebuild_vqc()
        else:
            self.phase=1
            self.freeze_random_layers()
            self.rebuild_vqc()
            if self.r < 1:
                self.r = self.r+self.update_rate

    def store_weights_in_layers(self):
        trainable_layers = [layer for layer in self.vqc_layers if layer.trainable]
        layer_weights = np.reshape(self.w[0], (-1, int(self.w.shape[1]/len(trainable_layers))))
        for i, layer in enumerate(trainable_layers):
            layer.update_weights(layer_weights[i])

    def _update_model_weights(self):
        weight_values= np.expand_dims(np.concatenate([layer.get_trainable_weights() for layer in [*self.vqc_layers, *self.pool_layers]]), axis=0)
        self.w = tf.Variable(initial_value=weight_values, 
                                dtype='float32', trainable=True, name='weights')

    def freeze_layers(self):
        for idx in range(len(self.vqc_layers)-self.q):
            self.vqc_layers[idx].freeze_weights()

    def freeze_random_layers(self):
        num_layers_to_freeze = int((1-self.r)*len(self.vqc_layers))
        freeze_layer_idx = sample(range(len(self.vqc_layers)), num_layers_to_freeze)
        for i,layer in enumerate(self.vqc_layers):
            if i in freeze_layer_idx:
                layer.freeze_weights()
            else:
                layer.unfreeze_weights()

    def rebuild_vqc(self):
        self.circuit = cirq.Circuit()
        self.circuit += [layer.circuit for layer in [*self.vqc_layers, *self.pool_layers]]
        self._update_model_weights()
        self.vqc = tfq.layers.ControlledPQC(self.circuit, self.readout_op, 
            differentiator=tfq.differentiators.ParameterShift())
    
    def copy_weights(self, other):
        other.store_weights_in_layers()
        for i,layer in enumerate([*self.vqc_layers, *self.pool_layers]):
            if i < len(self.vqc_layers):
                layer.update_weights(np.copy(other.vqc_layers[i].weights))
            else:
                layer.update_weights(np.copy(other.pool_layers[i-len(self.vqc_layers)].weights))
            layer.build()
        self.rebuild_vqc()

    def copy_layers(self, other):
        for i,layer in enumerate(self.vqc_layers):
            layer.gate_indices = np.copy(other.vqc_layers[i].gate_indices)
            layer.build()
        self.rebuild_vqc()

    def build_readout_op(self):
        return [cirq.Z(self.qubits[i]) for i in range(self.num_qubits-2, self.num_qubits)]