""" $lic$
Copyright (C) 2016-2020 by Tsinghua University and The Board of Trustees of
Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, \
        PoolingLayer, EltwiseLayer

'''
ResNet-34

He, Zhang, Ren, and Sun, 2015
'''

NN = Network('ResNet')

NN.set_input_layer(InputLayer(3, 224))

NN.add('conv1', ConvLayer(3, 64, 112, 7, 2))
NN.add('pool1', PoolingLayer(64, 56, 3, 2))

RES_PREV = 'pool1'

for i in range(3):
    NN.add('conv2_{}_a'.format(i), ConvLayer(64 if i == 0 else 128, 64, 56, 3))
    NN.add('conv2_{}_b'.format(i), ConvLayer(64, 64, 56, 3))

    # With residual shortcut.
    if i == 0:
        NN.add('conv2_br', ConvLayer(64, 128, 56, 3), prevs=(RES_PREV,))
        RES_PREV = 'conv2_br'
    NN.add('conv2_{}_res'.format(i), EltwiseLayer(128, 56, 2),
           prevs=(RES_PREV, 'conv2_{}_b'.format(i)))
    RES_PREV = 'conv2_{}_res'.format(i)

for i in range(4):
    NN.add('conv3_{}_a'.format(i), ConvLayer(128 if i == 0 else 256, 128, 56, 3))
    NN.add('conv3_{}_b'.format(i), ConvLayer(128, 128, 56, 3))

    # With residual shortcut.
    if i == 0:
        NN.add('conv3_br', ConvLayer(128, 256, 56, 3), prevs=(RES_PREV,))
        RES_PREV = 'conv3_br'
    NN.add('conv3_{}_res'.format(i), EltwiseLayer(256, 56, 2),
           prevs=(RES_PREV, 'conv3_{}_b'.format(i)))
    RES_PREV = 'conv3_{}_res'.format(i)

for i in range(6):
    NN.add('conv4_{}_a'.format(i), ConvLayer(256 if i == 0 else 512, 256, 56, 3))
    NN.add('conv4_{}_b'.format(i), ConvLayer(256, 256, 56, 3))

    # With residual shortcut.
    if i == 0:
        NN.add('conv4_br', ConvLayer(256, 512, 56, 3), prevs=(RES_PREV,))
        RES_PREV = 'conv4_br'
    NN.add('conv4_{}_res'.format(i), EltwiseLayer(512, 56, 2),
           prevs=(RES_PREV, 'conv4_{}_b'.format(i)))
    RES_PREV = 'conv4_{}_res'.format(i)
    
for i in range(3):
    NN.add('conv5_{}_a'.format(i), ConvLayer(512 if i == 0 else 1024, 512, 56, 3))
    NN.add('conv5_{}_b'.format(i), ConvLayer(512, 512, 56, 3))

    # With residual shortcut.
    if i == 0:
        NN.add('conv5_br', ConvLayer(512, 1024, 56, 3), prevs=(RES_PREV,))
        RES_PREV = 'conv5_br'
    NN.add('conv5_{}_res'.format(i), EltwiseLayer(1024, 56, 2),
           prevs=(RES_PREV, 'conv5_{}_b'.format(i)))
    RES_PREV = 'conv5_{}_res'.format(i)

NN.add('pool5', PoolingLayer(1024, 1, 7))

NN.add('fc', FCLayer(1024, 1000))