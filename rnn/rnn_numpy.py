import numpy as np


MASK_MODE = ["none", "zero", "persistent"]


class RNN(object):
    """
    an implementation of rnn in numpy:
        h_t = tanh(w_x * x_t + b_x + w_h * h_t_1 + b_h)
        o_t = tanh(w_o * h_t + b_o)

    input:
        x: (seq_len, batch_size, input_size)
        h0: (batch_size, hidden_size)
        mask: (seq_len, batch_size)
        weight and bias in one layer of one direction:
            weight_x:
                first layer:
                    (hidden_size, input_size)
                other layers:
                    (hidden_size, num_directions * hidden_size)
            weight_h: (hidden_size, num_directions * hidden_size)
            weight_o: (hidden_size, hidden_size)

            bias_x: (hidden_size)
            bias_h: (hidden_size)
            bias_o: (hidden_size)

        all weights are concatenated into one weight:
            (num_layers * num_directions * (weight_x.size() + weight_h.size() + weight_o.size())
        all bias are concatenated into one bias:
            (num_layers * num_directions * (bias_x.size() + bias_h.size() + bias_o.size())

        h0 will be initialized to zero if it is None as input.
        weight_o and bias_o only works for has_out_layer = True.

    output:
            output: (seq_len, batch_size, num_directions * hidden_size)
            ht: (batch_size, hidden_size)
    """
    def __init__(self, input_size, hidden_size, activation="tanh", has_out_layer=False, allocate=True):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._has_out_layer = has_out_layer
        if allocate:
            self._weight_space = np.zeros((self.get_weight_space_size()))
            self._bias_space = np.zeros((self.get_bias_space_size()))
        else:
            self._weight_space = None
            self._bias_space = None
        if activation == "tanh":
            self._activation = np.tanh
        elif activation == "relu":
            self._activation = lambda x: np.maximum(x, 0)
        else:
            raise ValueError("activation should be tanh or relu")

    def get_weight_space_size(self):
        weight_space_size = 0
        weight_space_size += self._hidden_size * self._input_size
        weight_space_size += self._hidden_size * self._hidden_size
        if self._has_out_layer:
            weight_space_size += self._hidden_size * self._hidden_size
        return weight_space_size

    def get_bias_space_size(self):
        bias_space_size = 0
        bias_space_size += self._hidden_size
        bias_space_size += self._hidden_size
        if self._has_out_layer:
            bias_space_size += self._hidden_size
        return bias_space_size

    def get_weight_x(self):
        w_x = self._weight_space[:self._input_size * self._hidden_size]
        return w_x.reshape(self._hidden_size, self._input_size)

    def set_weight_x(self, w_x):
        self._weight_space[:self._input_size * self._hidden_size] = w_x.reshape(-1)

    def get_weight_h(self):
        w_h_begin = self._input_size * self._hidden_size
        w_h_size = self._hidden_size * self._hidden_size
        w_h = self._weight_space[w_h_begin : w_h_begin + w_h_size]
        return w_h.reshape(self._hidden_size, self._hidden_size)

    def set_weight_h(self, w_h):
        w_h_begin = self._input_size * self._hidden_size
        w_h_size = self._hidden_size * self._hidden_size
        self._weight_space[w_h_begin: w_h_begin + w_h_size] = w_h.reshape(-1)

    def get_weight_o(self):
        if not self._has_out_layer:
            return None
        w_o_begin = self._input_size * self._hidden_size + self._hidden_size * self._hidden_size
        w_o_size = self._hidden_size * self._hidden_size
        w_o = self._weight_space[w_o_begin : w_o_begin + w_o_size]
        return w_o.reshape(self._hidden_size, self._hidden_size)

    def set_weight_o(self, w_o):
        w_o_begin = self._input_size * self._hidden_size + self._hidden_size * self._hidden_size
        w_o_size = self._hidden_size * self._hidden_size
        self._weight_space[w_o_begin: w_o_begin + w_o_size] = w_o.reshape(-1)

    def get_bias_x(self):
        return self._bias_space[:self._hidden_size]

    def set_bias_x(self, b_x):
        self._bias_space[:self._hidden_size] = b_x.reshape(-1)

    def get_bias_h(self):
        return self._bias_space[self._hidden_size : 2 * self._hidden_size]

    def set_bias_h(self, b_h):
        self._bias_space[self._hidden_size: 2 * self._hidden_size] = b_h.reshape(-1)

    def get_bias_o(self):
        if not self._has_out_layer:
            return None
        return self._bias_space[2 * self._hidden_size : 3 * self._hidden_size]

    def set_bias_o(self, b_o):
        self._bias_space[2 * self._hidden_size: 3 * self._hidden_size] = b_o.reshape(-1)

    def __call__(self, x, h0=None, mask=None, mask_mode="none", inverse=False):
        assert(mask_mode in MASK_MODE)
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        input_size = x.shape[2]
        assert(input_size == self._input_size)
        if h0 is not None:
            assert(h0.shape[0] == batch_size)
            assert(h0.shape[1] == self._hidden_size)
        else:
            h0 = np.zeros((batch_size, self._hidden_size))

        if mask_mode != "none":
            assert(mask.shape[0] == seq_len)
            assert(mask.shape[1] == batch_size)

        w_x = self.get_weight_x()
        b_x = self.get_bias_x()
        w_h = self.get_weight_h()
        b_h = self.get_bias_h()
        w_o = self.get_weight_o()
        b_o = self.get_weight_o()
        x_all = np.matmul(x, w_x.transpose()) + b_x
        h = h0.copy()
        y = np.zeros((seq_len, batch_size, self._hidden_size))
        for seq in range(seq_len):
            actual_seq = seq if not inverse else seq_len - seq - 1
            if mask_mode == "zero":
                h *= mask[actual_seq]
            h = self._activation(np.matmul(h, w_h.transpose()) + b_h + x_all[actual_seq])
            if self._has_out_layer:
                y[actual_seq] = self._activation(np.matmul(w_o, h) + b_o)
            else:
                y[actual_seq] = h

            if mask_mode == "persistent":
                h = mask * h + (1 - mask) * h0
                h0[:] = h
        return y, h


class MultiLayerRNN(object):
    """
    input:
        x: (seq_len, batch_size, input_size)
        h0: (num_layers * num_directions, batch_size, hidden_size)
        mask: (seq_len, batch_size)
        h0 will be initialized to zero if it is None as input.

    output:
        output: (seq_len, batch_size, num_directions * hidden_size)
        ht: (num_layers * num_directions, batch_size, hidden_size)

    layout of weight_space_size:
                          ______________
        layer = 0        |   weight_x   |   hidden_size * input_size
        direction = 0    |______________|
                         |   weight_h   |   hidden_size * hidden_size
                         |______________|
                         |   weight_o   |   hidden_size * hidden_size
                         |______________|
        layer = 0        |   weight_x   |   hidden_size * input_size
        direction = 1    |______________|
                         |   weight_h   |   hidden_size * hidden_size
                         |______________|
                         |   weight_o   |   hidden_size * hidden_size
                         |______________|
        layer = 1        |   weight_x   |   hidden_size * directions * hidden_size
        direction = 0    |______________|
                         |   weight_h   |   hidden_size * hidden_size
                         |______________|
                         |   weight_o   |   hidden_size * hidden_size
                         |______________|
        layer = 1        |   weight_x   |   hidden_size * directions * hidden_size
        direction = 1    |______________|
                         |   weight_h   |   hidden_size * hidden_size
                         |______________|
                         |   weight_o   |   hidden_size * hidden_size
                         |______________|
        ...

        bias has the same order with weight.
        all weights are concatenated into one weight.
        all bias are concatenated into one bias.
        """
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bidirectiaonl=False,
                 activation="tanh", has_out_layer=False, allocate=True):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._num_directions = 2 if bidirectiaonl else 1
        self._has_out_layer = has_out_layer
        assert(activation in ["tanh", "relu"])
        self._activation = activation

        # used for inference size of weight and bias
        self._first_layer = RNN(self._input_size, self._hidden_size, activation=self._activation,
                                has_out_layer=self._has_out_layer, allocate=False)
        self._other_layer = RNN(self._num_directions * self._hidden_size, self._hidden_size,
                                activation=self._activation, has_out_layer=self._has_out_layer,
                                allocate=False)
        if allocate:
            self._weight_space = np.zeros((self.get_weight_space_size()))
            self._bias_space = np.zeros((self.get_bias_space_size()))
        else:
            self._weight_space = None
            self._bias_space = None

    def get_weight_space_size(self):
        weight_space_size = 0
        weight_space_size += self._num_directions * self._first_layer.get_weight_space_size()
        for layer in range(1, self._num_layers):
            weight_space_size += self._num_directions * self._other_layer.get_weight_space_size()
        return weight_space_size

    def get_bias_space_size(self):
        return self._num_layers * self._num_directions * self._first_layer.get_bias_space_size()

    def _get_weight_x_position(self, layer, direction):
        if layer == 0:
            size = self._hidden_size * self._input_size
            begin = direction * self._first_layer.get_weight_space_size()
        else:
            size = self._hidden_size * self._num_directions * self._hidden_size
            begin = self._num_directions * self._first_layer.get_weight_space_size()
            for _ in range(1, layer):
                begin += self._num_directions * self._other_layer.get_weight_space_size()
            begin += direction * self._other_layer.get_weight_space_size()
        return begin, size

    def get_weight_x(self, layer, direction):
        begin, size = self._get_weight_x_position(layer, direction)
        return self._weight_space[begin : begin + size].reshape(self._hidden_size, -1)

    def set_weight_x(self, layer, direction, w_x):
        begin, size = self._get_weight_x_position(layer, direction)
        self._weight_space[begin: begin + size] = w_x.reshape(-1)

    def _get_weight_h_position(self, layer, direction):
        begin_x, size_x = self._get_weight_x_position(layer, direction)
        begin = begin_x + size_x
        size = self._hidden_size * self._hidden_size
        return begin, size

    def get_weight_h(self, layer, direction):
        begin, size = self._get_weight_h_position(layer, direction)
        return self._weight_space[begin: begin + size].reshape(self._hidden_size, self._hidden_size)

    def set_weight_h(self, layer, direction, w_h):
        begin, size = self._get_weight_h_position(layer, direction)
        self._weight_space[begin: begin + size] = w_h.reshape(-1)

    def _get_weight_o_position(self, layer, direction):
        begin_h, size_h = self._get_weight_h_position(layer, direction)
        begin = begin_h + size_h
        size = self._hidden_size * self._hidden_size
        return begin, size

    def get_weight_o(self, layer, direction):
        if not self._has_out_layer:
            return None
        begin, size = self._get_weight_o_position(layer, direction)
        return self._weight_space[begin: begin + size].reshape(self._hidden_size, self._hidden_size)

    def set_weight_o(self, layer, direction, w_o):
        begin, size = self._get_weight_o_position(layer, direction)
        self._weight_space[begin: begin + size] = w_o.reshape(-1)

    def _get_bias_x_position(self, layer, direction):
        begin = layer * self._num_directions * self._first_layer.get_bias_space_size()
        begin += direction * self._first_layer.get_bias_space_size()
        return begin, self._hidden_size

    def get_bias_x(self, layer, direction):
        begin, size = self._get_bias_x_position(layer, direction)
        return self._bias_space[begin: begin + size].reshape(-1)

    def set_bias_x(self, layer, direction, b_x):
        begin, size = self._get_bias_x_position(layer, direction)
        self._bias_space[begin: begin + size] = b_x.reshape(-1)

    def _get_bias_h_position(self, layer, direction):
        begin_x, size_x = self._get_bias_x_position(layer, direction)
        return begin_x + size_x, self._hidden_size

    def get_bias_h(self, layer, direction):
        begin, size = self._get_bias_h_position(layer, direction)
        return self._bias_space[begin: begin + size].reshape(-1)

    def set_bias_h(self, layer, direction, b_h):
        begin, size = self._get_bias_h_position(layer, direction)
        self._bias_space[begin: begin + size] = b_h.reshape(-1)

    def _get_bias_o_position(self, layer, direction):
        begin_h, size_h = self._get_bias_h_position(layer, direction)
        return begin_h + size_h, self._hidden_size

    def get_bias_o(self, layer, direction):
        if not self._has_out_layer:
            return None
        begin, size = self._get_bias_o_position(layer, direction)
        return self._bias_space[begin: begin + size].reshape(-1)

    def set_bias_o(self, layer, direction, b_o):
        begin, size = self._get_bias_o_position(layer, direction)
        self._bias_space[begin: begin + size] = b_o.reshape(-1)

    def __call__(self, x, h0=None, mask=None, mask_mode="none"):
        assert (mask_mode in MASK_MODE)
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        input_size = x.shape[2]
        assert (input_size == self._input_size)
        if h0 is not None:
            assert (h0.shape[0] == self._num_layers * self._num_directions)
            assert (h0.shape[1] == batch_size)
            assert (h0.shape[2] == self._hidden_size)
        else:
            h0 = np.zeros((self._num_layers * self._num_directions, batch_size, self._hidden_size))

        if mask_mode != "none":
            assert (mask.shape[0] == seq_len)
            assert (mask.shape[1] == batch_size)

        x_input = x.copy()
        y = np.zeros((seq_len, batch_size, self._num_directions * self._hidden_size))
        h = h0.copy()
        for layer in range(self._num_layers):
            rnn = RNN(x_input.shape[-1], self._hidden_size,
                      activation=self._activation, has_out_layer=self._has_out_layer)
            for direction in range(self._num_directions):
                h_input = h0[layer * self._num_directions + direction]
                rnn.set_weight_x(self.get_weight_x(layer, direction))
                rnn.set_weight_h(self.get_weight_h(layer, direction))
                rnn.set_bias_x(self.get_bias_x(layer, direction))
                rnn.set_bias_h(self.get_bias_h(layer, direction))
                if self._has_out_layer:
                    rnn.set_weight_o(self.get_weight_o(layer, direction))
                    rnn.set_bias_o(self.get_bias_o(layer, direction))
                y_out, h_out = rnn(x_input, h_input, mask, mask_mode, inverse=(direction == 1))
                y[:, :, direction * self._hidden_size: (direction + 1) * self._hidden_size] = y_out
                h[layer * self._num_directions + direction] = h_out
            x_input = y.copy()
        return y, h
