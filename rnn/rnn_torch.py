import torch
from rnn.rnn_np import MultiLayerRNN
import numpy as np


class RNNTorchChecker(object):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, activation):
        self._input = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._bidirectional = bidirectional
        self._activation = activation
        self._num_directions = 2 if bidirectional else 1

        self.rnn_np = MultiLayerRNN(input_size, hidden_size, num_layers=num_layers,
                                    bidirectiaonl=bidirectional, activation=activation,
                                    has_out_layer=False, allocate=True)
        self.rnn_torch = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers,
                                      bias=True, batch_first=False, bidirectional=bidirectional,
                                      nonlinearity=activation)

    def pass_torch_parameters(self):
        rnn = self.rnn_np
        torch_weight = self.rnn_torch.all_weights
        for layer in range(self._num_layers):
            for direction in range(self._num_directions):
                rnn.set_weight_x(layer, direction,
                                 torch_weight[layer * self._num_directions + direction][0].detach().numpy())
                rnn.set_weight_h(layer, direction,
                                 torch_weight[layer * self._num_directions + direction][1].detach().numpy())
                rnn.set_bias_x(layer, direction,
                               torch_weight[layer * self._num_directions + direction][2].detach().numpy())
                rnn.set_bias_h(layer, direction,
                               torch_weight[layer * self._num_directions + direction][3].detach().numpy())

    def check(self, x, h0):
        with torch.no_grad():
            x = torch.as_tensor(x)
            h0 = torch.as_tensor(h0)
            y_torch, h_torch = self.rnn_torch(x, h0)
            y_np, h_np = self.rnn_np(x.detach().numpy(), h0.detach().numpy())
            # print(y_np, "\n", y_torch, "\n", y_np - y_torch.detach().numpy())
            assert (np.linalg.norm(y_np.flatten() - y_torch.detach().numpy().flatten(), ord=2) < 1e-4)
            assert (np.linalg.norm(h_np.flatten() - h_torch.detach().numpy().flatten(), ord=2) < 1e-4)
            print("check succeeds!\n")


if __name__ == "__main__":
    batch_size = 10
    input_size = 3
    hidden_size = 4
    num_layers = 3
    seq_len = 20
    bidirectional = True
    activation = "relu"
    num_directions = 2 if bidirectional else 1

    x = torch.randn(seq_len, batch_size, input_size)
    h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size) * 0

    rnn_checker = RNNTorchChecker(input_size, hidden_size, num_layers,
                                  bidirectional, activation)
    rnn_checker.pass_torch_parameters()
    rnn_checker.check(x, h0)
