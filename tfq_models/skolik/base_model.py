from random import sample
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import numpy as np

from tfq_models.vqc_model import VQC_Model_Base
from tfq_models.skolik.vqc_layers import VQC_Layer

# TODO documentation
class VQC_Model(VQC_Model_Base):

    def __init__(self,  num_qubits, 
                        num_layers, 
                        activation='linear', 
                        scale=None,
                        hybrid=False,
                        initial_layers=2,
                        p=2,
                        q=4,
                        r=0.2,
                        layertype = VQC_Layer):
        self.initial_layers=initial_layers
        self.p=p
        self.q=q
        self.r=r
        self.phase=0
        self.Layer = layertype

        super(VQC_Model, self).__init__( num_qubits, 
                                            num_layers, 
                                            activation, 
                                            scale, 
                                            hybrid)

    def _encoding_ops(self, input, qubit):
        return cirq.rx(input).on(qubit)

    def build_readout_op(self):
        return [cirq.Z(self.qubits[i]) for i in range(self.num_qubits-2, self.num_qubits)]

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
                self.r = self.r+0.1

    def create_circuit(self):
        self.vqc_layers = []
        layer_circuit = self.add_layers(self.initial_layers)
        self._update_model_weights()
        return layer_circuit

    def store_weights_in_layers(self):
        trainable_layers = [layer for layer in self.vqc_layers if layer.trainable]
        layer_weights = np.reshape(self.w[0], (-1, int(self.w.shape[1]/len(trainable_layers))))
        for i, layer in enumerate(trainable_layers):
            layer.update_weights(layer_weights[i])

    def _update_model_weights(self):
        weight_values= np.expand_dims(np.concatenate([layer.get_trainable_weights() for layer in self.vqc_layers]), axis=0)
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
        self.circuit += [layer.circuit for layer in self.vqc_layers]
        self._update_model_weights()
        self.vqc = tfq.layers.ControlledPQC(self.circuit, self.readout_op, 
            differentiator=tfq.differentiators.ParameterShift())
    
    def copy_weights(self, other):
        for i,layer in enumerate(self.vqc_layers):
            layer.update_weights(np.copy(other.vqc_layers[i].weights))
            layer.build()
        self.rebuild_vqc()

    def copy_layers(self, other):
        for i,layer in enumerate(self.vqc_layers):
            layer.gate_indices = np.copy(other.vqc_layers[i].gate_indices)
            layer.build()
        self.rebuild_vqc()

    def add_layers(self, num):
        circuit = cirq.Circuit()

        for idx in range(num):
            layer = self.Layer(self.qubits, len(self.vqc_layers), self.activation)
            self.vqc_layers.append(layer)
            circuit += layer.circuit

        return circuit

