import numpy as np
import torch

class Decode_Layer(torch.nn.Module):

    def __init__(self, num_qubits, shots = 1024, device=torch.device("cpu")):
        super(Decode_Layer, self).__init__()
        self.device=device
        self.num_qubits=num_qubits
        self.shots=shots
        basis_states = torch.tensor([val for val in range(2**self.num_qubits)], device=device, requires_grad=False)
        mask = 2 ** torch.arange(self.num_qubits - 1, -1, -1).to(device, basis_states.dtype)
        self.basis_states = basis_states.unsqueeze(-1).bitwise_and(mask).ne(0).float()
        
        self.high_index = [list(),list()]
        self.low_index = [list(), list()]
        for i in range(0,int(self.num_qubits/2)):
            ind = i*2
            for j, state in enumerate(self.basis_states):
                if state[ind] == 1 and state[ind + 1]==1 or state[ind] == 0 and state[ind + 1]==0:
                    self.high_index[i].append(j)
            self.low_index[i] = np.arange(self.num_qubits**2)
            self.low_index[i] = np.array(list(set(self.low_index[i])-set(self.high_index[i])))

        self.high_index = torch.as_tensor(self.high_index)
        self.low_index = torch.as_tensor(self.low_index)

    def forward(self, input):
        batch_size = input.shape[0]
        expectation_values = torch.zeros(int(self.num_qubits/2), batch_size)    
        for i in range(0,int(self.num_qubits/2)):
            high_x = torch.gather(input, index=torch.tile(self.high_index[i], (batch_size, 1)), dim=1)
            low_x = torch.gather(input, index=torch.tile(self.low_index[i], (batch_size, 1)), dim=1)
            expectation_values[i] = torch.sum(high_x, dim=1) - torch.sum(low_x, dim=1)
        return torch.transpose(expectation_values, 0, 1)

