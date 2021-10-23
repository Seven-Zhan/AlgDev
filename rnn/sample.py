import caffe
caffe.set_mode_cpu()
import numpy as np


# inputs
input_x = np.random.randn(2, 1, 10)
np.save('input_x.npy', input_x)
input_cont = np.random.randn(2, 1)
np.save('input_cont.npy', input_cont)
input_h = np.random.randn(1, 1, 15)
np.save('input_h.npy', input_h)

# model
model_caffe = caffe.SGDSolver('solver.prototxt')
model_caffe.net.blobs['input_x'].data[...] = input_x
model_caffe.net.blobs['input_cont'].data[...] = input_cont
model_caffe.net.blobs['input_h'].data[...] = input_h

model_caffe.net.forward()

# outputs
output_y = model_caffe.net.blobs['output_y'].data[...]
np.save('output_y.npy', output_y)
output_h = model_caffe.net.blobs['output_h'].data[...]
np.save('output_h.npy', output_h)

# weights and bias
params = model_caffe.net.params['rnn1']
for i, p in enumerate(params):
    np.save('{}_{}.npy'.format(i, p.data.shape), p.data)



# validate caffe rnn with numpy
numpy_x = np.load('input_x.npy')
numpy_cont = np.load('input_cont.npy')
numpy_h = np.load('input_h.npy')

w_xh, b_xh = np.load('0_(15, 10).npy'), np.load('1_(15,).npy')
w_hh = np.load('2_(15, 15).npy')
w_ho, b_ho = np.load('3_(15, 15).npy'), np.load('4_(15,).npy')

# input_cont -> input_cont1, input_cont2
cont1, cont2 = np.split(numpy_cont, 2, axis=0)

# input_x -> w_xh_x -> w_xh_x1, w_xh_x2
w_xh_x = np.dot(numpy_x, w_xh.T) + b_xh
w_xh_x1, w_xh_x2 = np.split(w_xh_x, 2, axis=0)

# h0 -> h_conted0 -> w_hh_h0
# w_hh_h0, w_xh_h1 -> h_neuron_input1 -> h1
# h1 -> w_ho_h1 -> o1
h_conted0 = numpy_h * cont1
w_hh_h0 = np.dot(h_conted0, w_hh)
h1 = np.tanh(w_hh_h0 + w_xh_x1)
o1 = np.tanh(np.dot(h1, w_ho.T) + b_ho)

# cont2, h1 -> h_conted1 -> w_hh_h1
# w_hh_h1, w_xh_x2 -> h_neuron_input2 -> h2
# h2 -> w_ho_h2 -> o2
h_conted1 = h1 * cont2
w_hh_h1 = np.dot(h_conted1, w_hh.T)
numpy_output_h = np.tanh(w_hh_h1 + w_xh_x2)
o2 = np.tanh(np.dot(numpy_output_h, w_ho.T) + b_ho)

# o1, o2 -> o
numpy_output_y = np.concatenate([o1, o2], axis=0)

# check the diff
diff_h = np.abs(output_h - numpy_output_h)
diff_o = np.abs(output_y - numpy_output_y)
print(diff_h.shape, '\n', diff_h.min(), diff_h.max())
print(diff_o.shape, '\n', diff_o.min(), diff_o.max())

# clean stuff
import subprocess
subprocess.run('rm *.npy', shell=True)