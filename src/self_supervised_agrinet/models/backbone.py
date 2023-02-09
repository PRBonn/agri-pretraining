# credits: https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
import time

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

# function for saving the main things of the backbone
class BackboneConfig():
  def __init__(self, name, h, w, d, dropout):
    self.name = name  # backbone name
    self.h = h  # height of image
    self.w = w  # width of image
    self.d = d  # number of channels of image
    self.dropout = dropout


# 3x3-kernel convolution with padding
# dilation = spacing between kernel points for atrous convolution. If =1, no spacing (dense kernel)
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
  padding = int((3 + ((dilation - 1) * 2) - 1) / 2)
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=padding, dilation=dilation, bias=False)



def conv3x1(in_planes, out_planes, stride=1, dilation=1):
  #padding = int((3 + ((dilation - 1) * 2) - 1) / 2)
  return nn.Conv2d(in_planes, out_planes, kernel_size=(3,1), stride=stride,
                   padding=(1,0), dilation=dilation, bias=False)


def conv1x3(in_planes, out_planes, stride=1, dilation=1):
  #padding = int((3 + ((dilation - 1) * 2) - 1) / 2)
  return nn.Conv2d(in_planes, out_planes, kernel_size=(1,3), stride=stride,
                   padding=(0,1), dilation=dilation, bias=False)



# basic ResNet block for ResNet18 and ResNet34: two convolutions with shortcut connection skipping the two layers
# the output is y = relu[w2 * relu(w1 * x) + x]
class BasicBlock(nn.Module):
  # this parameter here is not needed. It's been added only to have a parallel with the bottleneck layer that uses it
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_d=0.1):
    super(BasicBlock, self).__init__()
    # inplanes = num. channels of input, planes = num channels of output, bn_d = momentum for batchnorm
    self.conv1 = conv3x3(inplanes, planes, stride, dilation)
    self.bn1 = nn.BatchNorm2d(planes, momentum=bn_d)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes, momentum=bn_d)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    # relu(w1 * x)
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    # w2 * relu(w1 * x)
    out = self.conv2(out)
    out = self.bn2(out)

    # if a downsampling happened, the shortcut connection must be downsampled as well to not have dimensionality problems
    if self.downsample is not None:
      residual = self.downsample(x)

    # w2 * relu(w1 * x) + x
    out += residual

    # relu[w2 * relu(w1 * x) + x]
    out = self.relu(out)

    return out



class BasicBlockModified(nn.Module):
  # this parameter here is not needed. It's been added only to have a parallel with the bottleneck layer that uses it
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_d=0.1):
    super(BasicBlockModified, self).__init__()
    # inplanes = num. channels of input, planes = num channels of output, bn_d = momentum for batchnorm
    self.conv1 = conv3x1(inplanes, planes, stride, dilation)
    self.conv1_1 = conv1x3(planes, planes, dilation)
    self.bn1 = nn.BatchNorm2d(planes, momentum=bn_d)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x1(planes, planes)
    self.conv2_2 = conv1x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes, momentum=bn_d)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    # relu(w1 * x)
    out = self.conv1(x)
    out = self.conv1_1(out)
    out = self.bn1(out)
    out = self.relu(out)

    # w2 * relu(w1 * x)
    out = self.conv2(out)
    out = self.conv2_2(out)
    out = self.bn2(out)

    # if a downsampling happened, the shortcut connection must be downsampled as well to not have dimensionality problems
    if self.downsample is not None:
        residual = self.downsample(x)

    # w2 * relu(w1 * x) + x
    out += residual

    # relu[w2 * relu(w1 * x) + x]
    out = self.relu(out)

    return out



# bottleneck layer for ResNet50, 101, 152: it consists of 1x1 convolution, 3x3 convolution, 1x1 convolution with
# shortcut connection at the end of the three. the first 1x1 convolution reduces the number of channels by 4, the
# last one increases it back by a factor of 4!
class Bottleneck(nn.Module):
  expansion = 4   # parameter necessary for the increasing of the number of channels at the end

  def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_d=0.1):
    super(Bottleneck, self).__init__()
    self.padding = int((3 + ((dilation - 1) * 2) - 1) / 2)
    # conv1: from num_channels = inplanes to num_channels = planes
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes, momentum=bn_d)
    # conv2: same number of num_channels = planes from input to output
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=self.padding, dilation=dilation, bias=False)
    self.bn2 = nn.BatchNorm2d(planes, momentum=bn_d)
    # conv3: from num_channels = planes to num_channels = planes*expansion as explained before
    self.conv3 = nn.Conv2d(
        planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_d)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    # if downsampling is necessary, then the input must be processed in order to augment the number of channels
    # the downsample layer is added to the structure in the make_layer func
    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


# ******************************************************************************
# weight files for the pretrained resnets
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet34-modified': 'suca',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# number of layers per model: a layer is either a basicblock or a bottleneckblock
# to retrieve the number of the name: sum(layers) * K + 2, K = 2 for resnet18-34, K = 3 for resnet50-101-152
# the additional 2 stands for the initial 7x7 convolution + 3x3 max pooling with stride 2
model_layers = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet34-modified' : [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3],
}

# constitutional block of each kind of resnet for the construction
model_block = {
    'resnet18': BasicBlock,
    'resnet34': BasicBlock,
    'resnet34-modified' : BasicBlockModified,
    'resnet50': Bottleneck,
    'resnet101': Bottleneck,
    'resnet152': Bottleneck,
}


class Backbone(nn.Module):
  def __init__(self, input_size=None, OS=32, dropout=0.0, bn_d=0.1, model='resnet18'):
    super(Backbone, self).__init__()
    if input_size is None:
      input_size = [640, 640, 3]  # basic dimensions for COCO dataset
    self.inplanes = 64  # number of channels to be obtained after the initial 7x7 convolution
    self.dropout_r = dropout
    self.bn_d = bn_d
    self.resnet = model
    self.OS = OS  # output stride, i.e. how much will the input be downsampled at the end of the encoder process

    # check that resnet exists
    assert self.resnet in model_layers.keys()

    # generate layers depending on resnet type
    self.layers = model_layers[self.resnet]
    self.block = model_block[self.resnet]
    self.url = model_urls[self.resnet]
    self.strides = [2, 2, 1, 2, 2, 2]     # stride of each layer: 2 for 7x7, 2 for maxpool, 1 for 3x3-64, 2 for the others
    self.dilations = [1, 1, 1, 1, 1, 1]   # no atrous convolution!
    self.last_depth = input_size[2]
    self.last_channel_depth = 512 * self.block.expansion

    # check current stride is aligned with output stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Original OS: ", current_os)

    if OS > current_os:
      print("Can't do OS, ", OS, " because it is bigger than original ", current_os)
    else:
      # redo strides according to needed stride
      current_dil = int(current_os / OS)
      for i, stride in enumerate(reversed(self.strides), 0):
        self.dilations[-1 - i] *= int(current_dil)
        if int(current_os) != OS: # this should hopefully fail, this means that the cumulative stride sums up to OS
          if stride == 2:
            current_os /= 2
            current_dil /= 2
            self.strides[-1 - i] = 1
          if int(current_os) == OS:
            break
      print("New OS: ", int(current_os))
      print("Strides: ", self.strides)
      print("Dilations: ", self.dilations)

    # check input sizes to see if strides will be valid (0=h, 1=w)
    assert input_size[0] % OS == 0 and input_size[1] % OS == 0

    # input block: 7x7 convolution with out_channels = 64, followed by batchnorm, relu
    padding = int((7 + ((self.dilations[0] - 1) * 2) - 1) / 2)
    self.conv1 = nn.Conv2d(self.last_depth, 64, kernel_size=7,
                           stride=self.strides[0], padding=padding, bias=False)
    self.bn1 = nn.BatchNorm2d(64, momentum=self.bn_d)
    self.relu = nn.ReLU(inplace=True)

    # second block: 3x3 maxpooling with stride=2
    self.maxpool = nn.MaxPool2d(kernel_size=3,
                                stride=self.strides[1],
                                padding=1)

    # layers: notice that the output channels increase by a factor of 2 each time, strides are read from the above
    # vector and similar for dilations (even if they should be always 1). at the end, good-old dropout
    # block 1
    self.layer1 = self._make_layer(self.block, 64, self.layers[0], stride=self.strides[2],
                                   dilation=self.dilations[2], bn_d=self.bn_d)

    # block 2
    self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=self.strides[3],
                                   dilation=self.dilations[3], bn_d=self.bn_d)

    # block 3
    self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=self.strides[4],
                                   dilation=self.dilations[4], bn_d=self.bn_d)

    # block 4
    self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=self.strides[5],
                                   dilation=self.dilations[5], bn_d=self.bn_d)

    self.dropout = nn.Dropout2d(self.dropout_r)

    ## UNCOMMENT TO LOAD PRE TRAINED WEIGHTS from urls above 
    # load weights from internet using the provided urls
    # strict needs to be false because we don't have fc layer in backbone
    #print("Check for loading pretrained weights.")
    #if input_size[2] == 3:
    #  # only load if images are RGB
    #  print("Loading pretrained weights from internet for backbone...")
    #  self.load_state_dict(model_zoo.load_url(self.url,
    #                                          map_location=lambda storage,
    #                                          loc: storage),
    #                       strict=False)
    #  print("Done!")

  # make layer useful function
  def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_d=0.1):
    downsample = None

    # if stride > 1 or there is an increasing in the output channels, then we need to downsample at the beginning
    # downsampling layer that will process x and generate the residual with more channels (old channels * expansion)
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(planes * block.expansion, momentum=bn_d),
      )

    # append the first layer + the number of layers required ("blocks" is the number of layers defined above!)
    layers = []
    layers.append(block(self.inplanes, planes,
                        stride, dilation, downsample, bn_d))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, bn_d=bn_d))

    return nn.Sequential(*layers)

  def get_last_depth(self):
    return self.last_channel_depth

  def run_layer(self, x, layer, skips, os):
    y = layer(x)
    # if the input decreases in dimension, then we save the skip connection to be handed over to the decoder
    # remember that the input is in form (N, C, H, W) so we check entry 2 and 3 (namely H and W)
    if y.shape[2] < x.shape[2] and y.shape[3] < x.shape[3]:
      skips[os] = x.data
      os *= 2
    x = y
    return x, skips, os

  def forward(self, x):
    # store for skip connections
    skips = {}
    os = 1
    # run cnn in a sequential fashion
    x, skips, os = self.run_layer(x, self.conv1, skips, os)
    x, skips, os = self.run_layer(x, self.bn1, skips, os)
    x, skips, os = self.run_layer(x, self.relu, skips, os)
    x, skips, os = self.run_layer(x, self.maxpool, skips, os)
    x, skips, os = self.run_layer(x, self.layer1, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.layer2, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.layer3, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.layer4, skips, os)
    # for key in skips.keys():
    #   print(key, skips[key].shape)

    return x, skips


