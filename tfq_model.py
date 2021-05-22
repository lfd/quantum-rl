import numpy as np 
import tensorflow as tf
from tensorflow import keras
import tensorflow_quantum as tfq
import cirq
import sympy


# VQC as Keras Model (with tfq)
class QVC_Model(keras.Model):

    def __init__(self,  num_qubits, num_layers, nonlinearity=None, scale=None):
        super(QVC_Model, self).__init__()

        self.qubits = cirq.GridQubit.rect(1, num_qubits)
        
        circuit = self.create_circuit(num_qubits, num_layers)
        readout_op = [cirq.Z(self.qubits[i]) for i in (2,3)]

        self.vqc = tfq.layers.PQC(circuit, readout_op)

        self.scale = scale

        
    def call(self, inputs, trainig=False):
        x = [ self.encode_data(input, asTensor=True) for input in inputs ]
        x = tf.concat([self.vqc(i) for i in x], axis=0)
        if(self.scale is not None):
            x = self.scale(x)
        return x
        

    def create_circuit(self, num_qubits, num_layers):
        circuit = cirq.Circuit()

        symbols = sympy.symbols('param0:' + str(num_qubits*num_layers + 2*2*3 )) # qubits * layers + 2 * 2 * 3 pooling

        for idx in range(num_layers):
            circuit += self.vqc_layer(symbols=symbols[idx*num_qubits : (idx+1)*num_qubits])

        circuit += self.pool(source=self.qubits[0], sink=self.qubits[2], 
                            symbols=symbols[num_qubits*num_layers : num_qubits*num_layers+6])
        circuit += self.pool(source=self.qubits[1], sink=self.qubits[3], 
                            symbols=symbols[num_qubits*num_layers+6 : ])

        return circuit


    def vqc_layer(self, symbols, nonlinearity=None):
        
        symbols = self.reparameterize(symbols, nonlinearity)

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


    # using pooling as in reference implementation
    def pool(self, source, sink, symbols, nonlinearity=None):
        symbols = self.reparameterize(symbols, nonlinearity)
        
        circuit = cirq.Circuit()

        circuit.append([cirq.rx(symbols[0]).on(source), 
                        cirq.ry(symbols[1]).on(source), 
                        cirq.rz(symbols[2]).on(source)])

        circuit.append([cirq.rx(symbols[3]).on(sink), 
                        cirq.ry(symbols[4]).on(sink), 
                        cirq.rz(symbols[5]).on(sink)])

        circuit.append(cirq.CNOT(control=source, target=sink))

        circuit.append([cirq.rx(-symbols[0]).on(sink), 
                        cirq.ry(-symbols[1]).on(sink), 
                        cirq.rz(-symbols[2]).on(sink)])
        return circuit

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

    def reparameterize(self, weight_symbols, nonlinearity=None):
        
        # TODO add nonlinearity
        
        if nonlinearity == None:
            weight_symbols = [symbol * 2 * np.pi for symbol in weight_symbols]

        return weight_symbols


# full parameterized QVC as in reference implementation (https://github.com/lockwo/quantum_computation)
class QVC_Model_full_parameterized(QVC_Model):
    def __init__(self,  num_qubits, num_layers, nonlinearity=None, scale=None):
        super(QVC_Model_full_parameterized, self).__init__(num_qubits, num_layers, nonlinearity, scale)


    def create_circuit(self, num_qubits, num_layers):
        circuit = cirq.Circuit()

        symbols = sympy.symbols('param0:' + str(num_qubits*num_layers*3 + 2*3*3)) # qubits * layers * 3 weights + 2 * 3 * 3 pooling

        for idx in range(num_layers):
            circuit += self.vqc_layer(symbols=symbols[idx*num_qubits*3 : (idx+1)*num_qubits*3])

        circuit += self.pool(source=self.qubits[0], sink=self.qubits[2], 
                            symbols=symbols[num_qubits*num_layers*3 : num_qubits*num_layers*3+9])
        circuit += self.pool(source=self.qubits[1], sink=self.qubits[3], 
                            symbols=symbols[num_qubits*num_layers*3+9 : ])
        return circuit


    def vqc_layer(self, symbols, nonlinearity=None):
        
        symbols = self.reparameterize(symbols, nonlinearity)

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


    def pool(self, source, sink, symbols, nonlinearity=None):
        symbols = self.reparameterize(symbols, nonlinearity)
        
        circuit = cirq.Circuit()
        
        circuit.append([cirq.rx(symbols[0]).on(source), 
                        cirq.ry(symbols[1]).on(source), 
                        cirq.rz(symbols[2]).on(source)])

        circuit.append([cirq.rx(symbols[3]).on(sink), 
                        cirq.ry(symbols[4]).on(sink), 
                        cirq.rz(symbols[5]).on(sink)])

        circuit.append(cirq.CNOT(control=source, target=sink))

        circuit.append([cirq.rx(-symbols[6]).on(sink), 
                        cirq.ry(-symbols[7]).on(sink), 
                        cirq.rz(-symbols[8]).on(sink)])
        return circuit

