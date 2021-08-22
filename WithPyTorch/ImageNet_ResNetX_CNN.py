################################################################################
#
# FILE
#
#    ImageNet_ResNetX_CNN.py
#
# DESCRIPTION
#
#    Creates a convolutional neural network model in PyTorch designed for
#    ImageNet modified to 100 classes and downsampled to 3x56x56 sized images
#    via resizing and cropping. The model is based on the RegNetX image
#    classifier and modified slightly to fit the modified data.
#
#    Two python classes are defined in this file:
#       1. XBlock: The building block used in the model. At stride=1, this block
#          is a standard building block. At stride=2, this block is a
#          downsampling building block.
#       2. Model: Creates the convolutional neural network model using the
#          building block class XBlock.
#
#    There are six distinct layers in the model: the stem, encoder level 0,
#    encoder level 1, encoder level 2, encoder level 3, and the decoder. The
#    layer details are shown below:
#       1. Stem:
#               Conv(3x3,s=1)
#
#       2. Encoder level 0:
#               XBlock(s=1)
#
#       3. Encoder level 1:
#               XBlock(s=2)
#
#       4. Encoder level 2:
#               XBlock(s=2)
#               XBlock(s=1)
#               XBlock(s=1)
#               XBlock(s=1)
#
#       5. Encoder level 3:
#               XBlock(s=2)
#               XBlock(s=1)
#               XBlock(s=1)
#               XBlock(s=1)
#               XBlock(s=1)
#               XBlock(s=1)
#               XBlock(s=1)
#
#       6. Decoder:
#               AvgPool
#               Flatten
#               Linear
#
#    After being trained for 100 epochs with Adam as the optimizer and a
#    learning rate schedule of linear warmup followed by cosine decay, the final
#    accuracy achieved is 70.93%. (Note: training code is not provided in this
#    file).
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

# torch
import torch
import torch.nn as nn

################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_NUM_CHANNELS = 3
DATA_NUM_CLASSES  = 100

# model
MODEL_LEVEL_0_BLOCKS            = 1
MODEL_LEVEL_1_BLOCKS            = 1
MODEL_LEVEL_2_BLOCKS            = 4
MODEL_LEVEL_3_BLOCKS            = 7
MODEL_STEM_END_CHANNELS         = 24
MODEL_LEVEL_0_IDENTITY_CHANNELS = 24
MODEL_LEVEL_1_IDENTITY_CHANNELS = 56
MODEL_LEVEL_2_IDENTITY_CHANNELS = 152
MODEL_LEVEL_3_IDENTITY_CHANNELS = 368

# training
TRAINING_DISPLAY         = False

################################################################################
#
# NETWORK BUILDING BLOCK
#
################################################################################

# X block
class XBlock(nn.Module):

    # initialization
    def __init__(self, Ni, No, Fr=3, Fc=3, Sr=1, Sc=1, G=8):

        # parent initialization
        super(XBlock, self).__init__()

        # identity
        if ((Ni != No) or (Sr > 1) or (Sc > 1)):
            self.conv0_present = True
            self.conv0         = nn.Conv2d(Ni, No, (1, 1), stride=(Sr, Sc), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        else:
            self.conv0_present = False

        # residual
        self.bn1   = nn.BatchNorm2d(Ni, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(Ni, No, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')

        self.bn2   = nn.BatchNorm2d(No, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(No, No, (Fr, Fc), stride=(Sr, Sc), padding=(1, 1), dilation=(1, 1), groups=G, bias=False, padding_mode='zeros')

        self.bn3   = nn.BatchNorm2d(No, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(No, No, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')


    # forward path
    def forward(self, x):

        #if TRAINING_DISPLAY: print("\tXBLOCK_INPUT: ", x.shape)

        # residual
        res = self.bn1(x)
        res = self.relu1(res)
        res = self.conv1(res)
        #if TRAINING_DISPLAY: print("\tXBLOCK_CONV1: ", res.shape)

        res = self.bn2(res)
        res = self.relu2(res)
        res = self.conv2(res)
        #if TRAINING_DISPLAY: print("\tXBLOCK_CONV2: ", res.shape)

        res = self.bn3(res)
        res = self.relu3(res)
        res = self.conv3(res)
        #if TRAINING_DISPLAY: print("\tXBLOCK_CONV3: ", res.shape)

        # identity
        if (self.conv0_present == True):
            x = self.conv0(x)
            #if TRAINING_DISPLAY: print("\tXBLOCK_CONV0: ", x.shape)

        # summation
        x = x + res

        # return
        return x

################################################################################
#
# NETWORK
#
################################################################################

# define
class Model(nn.Module):

    # initialization
    def __init__(self,
                 data_num_channels,
                 data_num_classes,
                 model_level_0_blocks,
                 model_level_1_blocks,
                 model_level_2_blocks,
                 model_level_3_blocks,
                 model_stem_end_channels,
                 model_level_0_identity_channels,
                 model_level_1_identity_channels,
                 model_level_2_identity_channels,
                 model_level_3_identity_channels):

        # parent initialization
        super(Model, self).__init__()

        # encoder stem
        self.enc_stem = nn.ModuleList()
        self.enc_stem.append(nn.Conv2d(data_num_channels, model_stem_end_channels, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))

        # encoder level 0
        self.enc_0 = nn.ModuleList()
        self.enc_0.append(XBlock(model_stem_end_channels, model_level_0_identity_channels))
        for n in range(model_level_0_blocks - 1):
            self.enc_0.append(XBlock(model_level_0_identity_channels, model_level_0_identity_channels))

        # encoder level 1
        self.enc_1 = nn.ModuleList()
        self.enc_1.append(XBlock(model_level_0_identity_channels, model_level_1_identity_channels, Sr=2, Sc=2))
        for n in range(model_level_1_blocks - 1):
            self.enc_1.append(XBlock(model_level_1_identity_channels, model_level_1_identity_channels))

        # encoder level 2
        self.enc_2 = nn.ModuleList()
        self.enc_2.append(XBlock(model_level_1_identity_channels, model_level_2_identity_channels, Sr=2, Sc=2))
        for n in range(model_level_2_blocks - 1):
            self.enc_2.append(XBlock(model_level_2_identity_channels, model_level_2_identity_channels))

        # encoder level 3
        self.enc_3 = nn.ModuleList()
        self.enc_3.append(XBlock(model_level_2_identity_channels, model_level_3_identity_channels, Sr=2, Sc=2))
        for n in range(model_level_3_blocks - 1):
            self.enc_3.append(XBlock(model_level_3_identity_channels, model_level_3_identity_channels))

        # encoder level 3 complete the bn - relu pattern
        self.enc_3.append(nn.BatchNorm2d(model_level_3_identity_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_3.append(nn.ReLU())

        # decoder
        self.dec = nn.ModuleList()
        self.dec.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.dec.append(nn.Flatten())
        self.dec.append(nn.Linear(model_level_3_identity_channels, data_num_classes, bias=True))

    # forward path
    def forward(self, x):

        #if TRAINING_DISPLAY: print("MODEL_INPUT: ", x.shape)

        # encoder stem
        for layer in self.enc_stem:
            #if TRAINING_DISPLAY: print("MODEL_ENC_STEM: ", x.shape)
            x = layer(x)

        # encoder level 0
        for layer in self.enc_0:
            #if TRAINING_DISPLAY: print("MODEL_ENC0: ", x.shape)
            x = layer(x)

        # encoder level 1
        for layer in self.enc_1:
            #if TRAINING_DISPLAY: print("MODEL_ENC1: ", x.shape)
            x = layer(x)

        # encoder level 2
        for layer in self.enc_2:
            #if TRAINING_DISPLAY: print("MODEL_ENC2: ", x.shape)
            x = layer(x)

        # encoder level 3
        for layer in self.enc_3:
            #if TRAINING_DISPLAY: print("MODEL_ENC3: ", x.shape)
            x = layer(x)

        # decoder
        for layer in self.dec:
            #if TRAINING_DISPLAY: print("MODEL_DEC: ", x.shape)
            x = layer(x)

        # return
        return x

# create
model = Model(DATA_NUM_CHANNELS,
              DATA_NUM_CLASSES,
              MODEL_LEVEL_0_BLOCKS,
              MODEL_LEVEL_1_BLOCKS,
              MODEL_LEVEL_2_BLOCKS,
              MODEL_LEVEL_3_BLOCKS,
              MODEL_STEM_END_CHANNELS,
              MODEL_LEVEL_0_IDENTITY_CHANNELS,
              MODEL_LEVEL_1_IDENTITY_CHANNELS,
              MODEL_LEVEL_2_IDENTITY_CHANNELS,
              MODEL_LEVEL_3_IDENTITY_CHANNELS)