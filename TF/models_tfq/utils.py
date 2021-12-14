import cirq

def encoding_ops_skolik(input, qubit):
    return cirq.rx(input).on(qubit)

def encoding_ops_lockwood(input, qubit):
    return [cirq.rx(input).on(qubit), cirq.rz(input).on(qubit)]
