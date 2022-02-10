import numpy as np
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit_machine_learning.neural_networks import OpflowQNN, CircuitQNN
import torch
from qiskit.opflow.gradients import Gradient
from qiskit.opflow import StateFn, PauliSumOp, ListOp, AerPauliExpectation

from PT.models_qiskit.vqc_layers import Decode_Layer

# TODO: documentation
class VQC_Model(torch.nn.Module):
    """
    This class represents a trainable VQC using Qiskit.

    Attributes:

        num_qubits: Number of Qubits
        num_layers: Number of VQC-Layers
        in_scale: PyTorch Module representing trainable weights on the input
        out_scale: PyTorch Module representing trainable weights on the input
        encoding_ops: Rotation gates applied to inputs, 'skolik' or 'lockwood'
        readout_op: Extraction method, 'expval' or 'pooling'
        layertype: VQC-Layer architecture. 'skolik' or 'lockwood'
        shots: Number of shots in quantum instance
        nn_type: QNN-class that should be used, 'circuit_qnn' or 'opflow_qnn'
        q_backend: Qiskit's quantum backend
        device: torch device 
    """
    def __init__(self,  num_qubits, 
                        num_layers, 
                        in_scale:torch.nn.Module = None,
                        out_scale:torch.nn.Module = None,
                        encoding_ops = 'lockwood',
                        readout_op = 'expval',
                        layertype = 'lockwood',
                        shots:int =1024,
                        nn_type='circuit_qnn',
                        q_backend=Aer.get_backend('qasm_simulator'),
                        device = torch.device("cpu")):
        super(VQC_Model, self).__init__()
        self.num_qubits = num_qubits
        self.shots = shots
        self.nn_type = nn_type 
        self.in_scale = in_scale
        self.out_scale = out_scale
        
        input_params = ParameterVector('input', num_qubits)

        qi = QuantumInstance(q_backend, shots=shots)

        self.circuit = QuantumCircuit(self.num_qubits)

        if encoding_ops == 'skolik':
            for i, input in enumerate(input_params):
                self.circuit.rx(input, i)
        elif encoding_ops == 'lockwood':
            for i, input in enumerate(input_params):
                self.circuit.rx(input, i)
                self.circuit.rz(input, i)

        if layertype == 'skolik':
            num_gates = 2
            generate_layer = self.generate_layer_skolik
        else:
            num_gates = 3
            generate_layer=self.generate_layer_lockwood

        weight_params = ParameterVector('param', num_qubits*num_layers*num_gates)

        for i in range(num_layers):
            generate_layer(weight_params[i*num_qubits*num_gates : (i+1)*num_qubits*num_gates])

        if nn_type == 'circuit_qnn':
            qnn = CircuitQNN(circuit=self.circuit,
                    input_params=input_params,
                    weight_params=weight_params,
                    quantum_instance=qi,
                    gradient=Gradient(grad_method='param_shift'))

            self.decode_layer = Decode_Layer(num_qubits, shots, device)
        else:
            readout_op = ListOp([
                ~StateFn(PauliSumOp.from_list([('ZZII', 1.0)])) @ StateFn(self.circuit),
                ~StateFn(PauliSumOp.from_list([('IIZZ', 1.0)])) @ StateFn(self.circuit)])

            qnn = OpflowQNN(readout_op,
                        input_params=input_params,
                        weight_params=weight_params,
                        exp_val=AerPauliExpectation(),
                        quantum_instance=qi,
                        gradient=Gradient(grad_method='param_shift'))
        
        self.qnn = TorchConnector(qnn, initial_weights=torch.Tensor(np.zeros(num_qubits*num_layers*2)))

    def generate_layer_skolik(self, params):
        # variational part
        for i in range(self.num_qubits):
            self.circuit.ry(params[i*2], i)
            self.circuit.rz(params[i*2+1], i)

        # entangling part
        for i in range(self.num_qubits):
            self.circuit.cz(i, (i+1) % self.num_qubits)

    def generate_layer_lockwood(self, params):
        # entangling part
        for i in range(1,self.num_qubits):
            self.circuit.cnot(i, (i-1))

        # variational part
        for i in range(self.num_qubits):
            self.circuit.rx(params[i*2], i)
            self.circuit.ry(params[i*2+1], i)
            self.circuit.rz(params[i*2+2], i)

    def forward(self, inputs):
        if self.in_scale:
            self.in_scale(inputs)

        x = self.qnn(inputs)
        
        if self.nn_type == 'circuit_qnn':
            x = self.decode_layer(x)

        if self.out_scale:
            x = self.out_scale(x)
        return x

