Convert Keras to Pytorch
-------------------------

python3 keras2torch.py -i input.h5 -o output.pth

Convert Pytorch to Keras
-------------------------

python3 torch2keras.py -i input.pth -o output.h5 N1 N2 N3 ...

where N1 N2 N3 are the number of neurons per layer, including input and output layer.

For example, for a feed forward NN with input dimension of 1024, first hidden neuron dimension 2048, second hidden neuron dimension 512,
output neuron dimension 10, you will need:

python3 torch2keras.py -i input.pth -o output.h5 1024 2048 512 10

For Image dataset:

python3 torch2keras.py -i input.pth -o output.h5 --flatten --image_size 28 --image_channel 1 N1 N2 N3 ...

If the input is a 2-D image, you need the --flatten flag and also specify --image_size and --image_channel

