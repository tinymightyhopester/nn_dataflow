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
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, PoolingLayer

'''
InceptionV

ILSVRC 2014
'''

NN = Network('Inception')

NN.set_input_layer(InputLayer(3, 299))

'''this section represents the stem block'''

NN.add('conv1a', ConvLayer(3, 64, 192, 3, 2))
NN.add('conv1b', ConvLayer(3, 64, 192, 3))
NN.add('conv1c', ConvLayer(3, 64, 192, 3))
NN.add('pool1', PoolingLayer(64, 56, 3, strd=2))

NN.add('conv2_3x3a', ConvLayer(64, 64, 56, 3))
NN.add('conv2_3x3b', ConvLayer(64, 192, 56, 3))

NN.add('pool2', PoolingLayer(192, 28, 3, strd=2))


def add_inception(network, incp_id, sfmap, nfmaps_in, nfmaps_1, nfmaps_3r,
                  nfmaps_3, nfmaps_5r, nfmaps_5, nfmaps_pool, prevs):
    ''' Add an inception module to the network. '''
    pfx = 'inception_{}_'.format(incp_id)
    # 1x1 branch.
    network.add(pfx + '1x1', ConvLayer(nfmaps_in, nfmaps_1, sfmap, 1),
                prevs=prevs)
    # 3x3 branch.
    network.add(pfx + '3x3_reduce_a', ConvLayer(nfmaps_in, nfmaps_3r, sfmap, 1),
                prevs=prevs)
    network.add(pfx + '3x3a', ConvLayer(nfmaps_3r, nfmaps_3, sfmap, 3))
    # 3x3 branch.
    network.add(pfx + '3x3_reduce_b', ConvLayer(nfmaps_in, nfmaps_3r, sfmap, 1),
                prevs=prevs)
    network.add(pfx + '3x3b', ConvLayer(nfmaps_3r, nfmaps_3, sfmap, 3))
    # Pooling branch.
    network.add(pfx + 'pool_proj', ConvLayer(nfmaps_in, nfmaps_pool, sfmap, 1),
                prevs=prevs)
    # Merge branches.
    return (pfx + '1x1', pfx + '3x3a', pfx + '3x3b', pfx + 'pool_proj')

def add_inception_big(network, incp_id, sfmap, nfmaps_in, nfmaps_1, nfmaps_3r,
                  nfmaps_3, nfmaps_5r, nfmaps_5, nfmaps_pool, prevs):
    ''' Add an inception module to the network. '''
    pfx = 'inception_{}_'.format(incp_id)
    # 1x1 branch.
    network.add(pfx + '1x1', ConvLayer(nfmaps_in, nfmaps_1, sfmap, 1),
                prevs=prevs)
    # 3x3 branch.
    network.add(pfx + '3x3_reduce_a1', ConvLayer(nfmaps_in, nfmaps_3r, sfmap, 1),
                prevs=prevs)
    network.add(pfx + '3x3a1', ConvLayer(nfmaps_3r, nfmaps_3, sfmap, 3))
    # 3x3 branch.
    network.add(pfx + '3x3_reduce_a2', ConvLayer(nfmaps_in, nfmaps_3r, sfmap, 1),
                prevs=prevs)
    network.add(pfx + '3x3a2', ConvLayer(nfmaps_3r, nfmaps_3, sfmap, 3))
    
    # 3x3 branch.
    network.add(pfx + '3x3_reduce_b1', ConvLayer(nfmaps_in, nfmaps_3r, sfmap, 1),
                prevs=prevs)
    network.add(pfx + '3x3b1', ConvLayer(nfmaps_3r, nfmaps_3, sfmap, 3))
    # 3x3 branch.
    network.add(pfx + '3x3_reduce_b1', ConvLayer(nfmaps_in, nfmaps_3r, sfmap, 1),
                prevs=prevs)
    network.add(pfx + '3x3b1', ConvLayer(nfmaps_3r, nfmaps_3, sfmap, 3))

    # Pooling branch.
    network.add(pfx + 'pool_proj', ConvLayer(nfmaps_in, nfmaps_pool, sfmap, 1),
                prevs=prevs)
    # Merge branches.
    return (pfx + '1x1', pfx + '3x3a1', pfx + '3x3a2', pfx + '3x3b1', pfx + '3x3b2', pfx + 'pool_proj')

def add_inception_intermediate(network, incp_id, sfmap, nfmaps_in, nfmaps_1, nfmaps_3r,
                  nfmaps_3, nfmaps_5r, nfmaps_5, nfmaps_pool, prevs):
    ''' Add an inception module to the network. '''
    pfx = 'inception_{}_'.format(incp_id)
    # 1x1 branch.
    network.add(pfx + '1x1', ConvLayer(nfmaps_in, nfmaps_1, sfmap, 1),
                prevs=prevs)
    # 3x3 branch.
    network.add(pfx + '3x3_reduce_a', ConvLayer(nfmaps_in, nfmaps_3r, sfmap, 1),
                prevs=prevs)
    network.add(pfx + '3x3', ConvLayer(nfmaps_3r, nfmaps_3, sfmap, 3))
    # Pooling branch.
    network.add(pfx + 'pool_proj', ConvLayer(nfmaps_in, nfmaps_pool, sfmap, 1),
                prevs=prevs)
    # Merge branches.
    return (pfx + '1x1', pfx + '3x3', pfx + 'pool_proj')

    


_PREVS = ('pool2',)

# First "block"
_PREVS = add_inception(NN, '3a', 28, 192, 64, 96, 128, 16, 32, 32,
                       prevs=_PREVS)
_PREVS = add_inception(NN, '3b', 28, 192, 64, 96, 128, 16, 32, 32,
                       prevs=_PREVS)
_PREVS = add_inception(NN, '3c', 28, 256, 128, 128, 192, 32, 96, 64,
                       prevs=_PREVS)

NN.add('pool3', PoolingLayer(480, 14, 3, strd=2), prevs=_PREVS)
_PREVS = ('pool3',)

# Intermediate
_PREVS = add_inception_intermediate(NN, '3-4', 28, 256, 128, 128, 192, 32, 96, 64,
                       prevs=_PREVS)
NN.add('pool3-4', PoolingLayer(480, 14, 3, strd=2), prevs=_PREVS)
_PREVS = ('pool3-4',)

# Second "Block"
_PREVS = add_inception(NN, '4a', 14, 480, 192, 96, 208, 16, 48, 64,
                       prevs=_PREVS)
_PREVS = add_inception(NN, '4b', 14, 512, 160, 112, 224, 24, 64, 64,
                       prevs=_PREVS)
_PREVS = add_inception(NN, '4c', 14, 512, 128, 128, 256, 24, 64, 64,
                       prevs=_PREVS)

NN.add('pool4', PoolingLayer(832, 7, 3, strd=2), prevs=_PREVS)
_PREVS = ('pool4',)

# Intermediate
_PREVS = add_inception_intermediate(NN, '4-5', 28, 256, 128, 128, 192, 32, 96, 64,
                       prevs=_PREVS)
NN.add('pool4-5', PoolingLayer(480, 14, 3, strd=2), prevs=_PREVS)
_PREVS = ('pool4-5',)

# third "block"
_PREVS = add_inception_big(NN, '5a', 7, 832, 256, 160, 320, 32, 128, 128,
                       prevs=_PREVS)
_PREVS = add_inception_big(NN, '5b', 7, 832, 384, 192, 384, 48, 128, 128,
                       prevs=_PREVS)

NN.add('pool5', PoolingLayer(1024, 1, 7), prevs=_PREVS)

NN.add('fc', FCLayer(1024, 1000))
