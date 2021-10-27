import caffe
caffe.set_mode_cpu()
import numpy as np


# inputs
input_x = np.random.randn(3, 1, 10)
input_cont = np.random.randn(3, 1)
input_h = np.random.randn(1, 1, 15)

# model
model_caffe = caffe.SGDSolver('solver.prototxt')
model_caffe.net.blobs['input_x'].data[...] = input_x
model_caffe.net.blobs['input_cont'].data[...] = input_cont
model_caffe.net.blobs['input_h'].data[...] = input_h

model_caffe.net.forward()

# outputs
caffe_output_y = model_caffe.net.blobs['output_y'].data[...]
caffe_output_h = model_caffe.net.blobs['output_h'].data[...]
print(caffe_output_h.shape)
print(caffe_output_y.shape)

# parameters: weights and bias
params = model_caffe.net.params['rnn1']



##### validation: caffe <-> numpy #####
w_xh, b_xh = params[0].data[...], params[1].data[...]
w_hh = params[2].data[...]
w_ho, b_ho = params[3].data[...], params[4].data[...]

# w_xh_x = fc(w_xh*x + b_xh)
# h_cont = h * cont
# w_hh_h = fc(w_hh*h_cont)
# h = tanh(w_xh_x + w_hh_h)
# y = tanh(fc(w_ho*h + b_ho))
numpy_output_h = input_h
numpy_output_ys = []
for i in range(input_x.shape[0]):
    h_cont = numpy_output_h * input_cont[i]
    w_xh_x = np.dot(input_x[i], w_xh.T) + b_xh
    w_hh_h = np.dot(h_cont, w_hh.T)
    numpy_output_h = np.tanh(w_xh_x + w_hh_h)
    numpy_output_y = np.tanh(np.dot(numpy_output_h, w_ho.T) + b_ho)
    numpy_output_ys.append(numpy_output_y)
numpy_output_y = np.concatenate(numpy_output_ys, axis=0)

# check the diff
diff_h = np.abs(caffe_output_h - numpy_output_h)
diff_o = np.abs(caffe_output_y - numpy_output_y)
print(diff_h.shape, diff_h.min(), diff_h.max())
print(diff_o.shape, diff_o.min(), diff_o.max())
