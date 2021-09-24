from abc import ABC, abstractmethod
import cirq
import numpy as np
import sympy

class Layer_Base(ABC):
    def __init__(self, idx, activation, trainable=True):
        self.trainable = trainable
        self.activation=activation
        self.layer_idx=idx
        self._circuit = cirq.Circuit()

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
    def build(self):
        pass

class Regular_Layer_Base(Layer_Base, ABC):
    def __init__(self, qubits, idx, activation, trainable=True):
        super(Regular_Layer_Base, self).__init__(idx, activation, trainable)
        self.num_qubits = len(qubits)
        self.qubits=qubits

    def _build(self, angles):
        self._circuit = cirq.Circuit()

        # Apply qubit rotations
        for idx in range(self.num_qubits):
            self._circuit.append(self._generate_gates(idx, angles))

        # Create entanglement
        self._add_entanglement_gates()

    @abstractmethod
    def _generate_gates(self, qubit_idx, symbols):
        pass

    @abstractmethod
    def _add_entanglement_gates(self):
        pass

    @abstractmethod
    def build(self):
        pass

class Pooling_Layer_Base(Layer_Base, ABC):
    def __init__(self, idx, activation, source, sink, trainable=True):
        super(Pooling_Layer_Base, self).__init__(idx, activation, trainable)
        self.source = source
        self.sink = sink

    # Base pooling
    def _build(self, angles):
        self._circuit = cirq.Circuit()

        self._circuit.append([cirq.rx(angles[0]).on(self.source), 
                            cirq.ry(angles[1]).on(self.source), 
                            cirq.rz(angles[2]).on(self.source)])

        self._circuit.append([cirq.rx(angles[3]).on(self.sink), 
                            cirq.ry(angles[4]).on(self.sink), 
                            cirq.rz(angles[5]).on(self.sink)])

        self._circuit.append(cirq.CNOT(control=self.source, target=self.sink))

        self._circuit.append([cirq.rx(-angles[6]).on(self.sink), 
                            cirq.ry(-angles[7]).on(self.sink), 
                            cirq.rz(-angles[8]).on(self.sink)])


class Pooling_Layer_V1(Pooling_Layer_Base):
    def __init__(self, idx, activation, source, sink, trainable=True):
        super(Pooling_Layer_V1, self).__init__(idx, activation, source, sink, trainable)
        self.num_weights=6
        self._weights=np.zeros(self.num_weights, dtype='float32')
        self.build()

    def build(self):
        if self.trainable:
            angles = sympy.symbols(f'pool{self.layer_idx}_0:{self.num_weights}')
        else:
            angles = self._reparameterize(self._weights).numpy()

        angles = angles + angles[3:]

        self._build(angles)

class Pooling_Layer_V2(Pooling_Layer_Base):
    def __init__(self, idx, activation, source, sink, trainable=True):
        super(Pooling_Layer_V2, self).__init__(idx, activation, source, sink, trainable)
        self.num_weights=9
        self._weights=np.zeros(self.num_weights, dtype='float32')
        self.build()

    def build(self):
        if self.trainable:
            angles = sympy.symbols(f'pool{self.layer_idx}_0:{self.num_weights}')
        else:
            angles = self._reparameterize(self._weights).numpy()

        self._build(angles)

class VQC_Layer_Skolik(Regular_Layer_Base):
    def __init__(self, qubits, idx, activation, trainable=True):
        super(VQC_Layer_Skolik, self).__init__(qubits, idx, activation, trainable)
        self.gate_indices=np.random.randint(0, 3, size=self.num_qubits)
        self._weights = np.zeros(self.num_qubits, dtype='float32')

        self.build()

    def build(self):
        if self.trainable:
            angles = sympy.symbols(f'weights_l{self.layer_idx}_0:{self.num_qubits}')
        else:
            angles = self._reparameterize(self._weights).numpy()

        self._build(angles)

    def _generate_gates(self, qubit_idx, symbols):
        rotation_gates = [cirq.rx(symbols[qubit_idx]).on(self.qubits[qubit_idx]), 
                        cirq.ry(symbols[qubit_idx]).on(self.qubits[qubit_idx]),
                        cirq.rz(symbols[qubit_idx]).on(self.qubits[qubit_idx])]
        return rotation_gates[self.gate_indices[qubit_idx]]

    def _add_entanglement_gates(self):
        for i in range(self.num_qubits-1):
            for j in range(i+1, self.num_qubits):
                self._circuit.append(cirq.CZ.on(self.qubits[i], self.qubits[j]))


class VQC_Layer_Skolik_V2(Regular_Layer_Base):

    def __init__(self, qubits, idx, activation, trainable=True):
        super(VQC_Layer_Skolik_V2, self).__init__(qubits, idx, activation, trainable=trainable)
        self._weights = np.zeros(self.num_qubits*2, dtype='float32')

        self.build()

    def build(self):
        if self.trainable:
            angles = sympy.symbols(f'weights_l{self.layer_idx}_0:{self.num_qubits*2}')
        else:
            angles = self._reparameterize(self._weights).numpy()

        self._build(angles)

    def _generate_gates(self, qubit_idx, symbols):
        return [cirq.ry(symbols[qubit_idx*2]).on(self.qubits[qubit_idx]),
                cirq.rz(symbols[qubit_idx*2+1]).on(self.qubits[qubit_idx])]

    def _add_entanglement_gates(self):
        for i in range(self.num_qubits):
            self._circuit.append(cirq.CZ.on(self.qubits[i], self.qubits[(i+1)%self.num_qubits]))

class VQC_Layer_Skolik_V3(VQC_Layer_Skolik_V2):
    def _add_entanglement_gates(self):
        for i in range(self.num_qubits-1):
            for j in range(i+1, self.num_qubits):
                self._circuit.append(cirq.CZ.on(self.qubits[i], self.qubits[j]))

class VQC_Layer_Lockwood(Regular_Layer_Base):

    def __init__(self, qubits, idx, activation, trainable=True):
        super(VQC_Layer_Lockwood, self).__init__(qubits, idx, activation, trainable=trainable)
        self._weights = np.zeros(self.num_qubits*3, dtype='float32')

        self.build()

    # In Lockwoods Paper entanglement Gates and Rotation Gates are switched
    def _build(self, angles):
        self._circuit = cirq.Circuit()

        # Create entanglement
        self._add_entanglement_gates()

        # Apply qubit rotations
        for idx in range(self.num_qubits):
            self._circuit.append(self._generate_gates(idx, angles))

    def build(self):
        if self.trainable:
            angles = sympy.symbols(f'weights_l{self.layer_idx}_0:{self.num_qubits*3}')
        else:
            angles = self._reparameterize(self._weights).numpy()

        self._build(angles)

    def _generate_gates(self, qubit_idx, symbols):
        return [cirq.rx(symbols[qubit_idx*3]).on(self.qubits[qubit_idx]),
                cirq.ry(symbols[qubit_idx*3+1]).on(self.qubits[qubit_idx]),
                cirq.rz(symbols[qubit_idx*3+2]).on(self.qubits[qubit_idx])]

    def _add_entanglement_gates(self):
        for i in range(1, self.num_qubits):
            self._circuit.append(cirq.CNOT(control=self.qubits[i], target=self.qubits[i-1]))

class VQC_Layer_Lockwood_V2(Regular_Layer_Base):

    def build(self):
        if self.trainable:
            angles = sympy.symbols(f'weights_l{self.layer_idx}_0:{self.num_qubits}')
        else:
            angles = self._reparameterize(self._weights).numpy()

        self._build(angles)

    def _generate_gates(self, qubit_idx, symbols):
        return [cirq.rx(symbols[qubit_idx*3]).on(self.qubits[qubit_idx]),
                cirq.ry(0.5 * np.pi).on(self.qubits[qubit_idx]), 
                cirq.rz(np.pi).on(self.qubits[qubit_idx])]