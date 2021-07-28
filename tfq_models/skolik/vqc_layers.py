from abc import ABC, abstractmethod
from random import sample
import cirq
import numpy as np
import sympy

class Layer_Base(ABC):
    def __init__(self, qubits, idx, activation, trainable=True):
        self.trainable = trainable
        self.num_qubits = len(qubits)
        self.qubits=qubits
        self.layer_idx=idx
        self.activation=activation

    def _build(self, angles):
        self._circuit = cirq.Circuit()

        # Apply qubit rotations
        for idx in range(self.num_qubits):
            self._circuit.append(self._generate_gates(idx, angles))

        # Create entanglement
        self._add_entanglement_gates()

        return self._circuit

    def update_weights(self, weights):
        self._weights = weights

    def get_trainable_weights(self):
        return self.weights if self.trainable else [] 

    def freeze_weights(self):
        if self.trainable:
            self.trainable = False
            self.build()

    def unfreeze_weights(self):
        if not self.trainable:
            self.trainable = True
            self.build()

    # TODO multiple function definition -> refactor
    def _reparameterize(self, weights):
        return self.activation(weights) * 2. * np.pi

    @property
    def circuit(self):
        return self._circuit

    @property
    def weights(self):
        return self._weights

    @abstractmethod
    def _generate_gates(self, qubit_idx, symbols):
        pass

    @abstractmethod
    def _add_entanglement_gates(self):
        pass

    @abstractmethod
    def build(self):
        pass


class VQC_Layer(Layer_Base):
    def __init__(self, qubits, idx, activation, trainable=True):
        super(VQC_Layer, self).__init__(qubits, idx, activation, trainable)
        self.gate_indices=np.random.randint(0, 3, size=self.num_qubits)
        self._weights = np.zeros(self.num_qubits, dtype='float32')

        self.build()

    def build(self):
        if self.trainable:
            angles = sympy.symbols(f'weights_l{self.layer_idx}_0:{self.num_qubits}')
        else:
            angles = self._reparameterize(self._weights).numpy()

        return self._build(angles)

    def _generate_gates(self, qubit_idx, symbols):
        rotation_gates = [cirq.rx(symbols[qubit_idx]).on(self.qubits[qubit_idx]), 
                        cirq.ry(symbols[qubit_idx]).on(self.qubits[qubit_idx]),
                        cirq.rz(symbols[qubit_idx]).on(self.qubits[qubit_idx])]
        return rotation_gates[self.gate_indices[qubit_idx]]

    def _add_entanglement_gates(self):
        for i in range(self.num_qubits-1):
            for j in range(i+1, self.num_qubits):
                self._circuit.append(cirq.CZ.on(self.qubits[i], self.qubits[j]))


class VQC_Layer_V2(Layer_Base):

    def __init__(self, qubits, idx, activation, trainable=True):
        super(VQC_Layer_V2, self).__init__(qubits, idx, activation, trainable=trainable)
        self._weights = np.zeros(self.num_qubits*2, dtype='float32')

        self.build()

    def build(self):
        if self.trainable:
            angles = sympy.symbols(f'weights_l{self.layer_idx}_0:{self.num_qubits*2}')
        else:
            angles = self._reparameterize(self._weights).numpy()

        return self._build(angles)

    def _generate_gates(self, qubit_idx, symbols):
        return [cirq.ry(symbols[qubit_idx*2]).on(self.qubits[qubit_idx]),
                cirq.rz(symbols[qubit_idx*2+1]).on(self.qubits[qubit_idx])]

    def _add_entanglement_gates(self):
        for i in range(self.num_qubits):
            self._circuit.append(cirq.CZ.on(self.qubits[i], self.qubits[(i+1)%self.num_qubits]))