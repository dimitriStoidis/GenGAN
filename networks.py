import torchvision.models as models
import torch.nn as nn
from torch.nn.functional import log_softmax
from torchvision.models.alexnet import AlexNet
import argparse
import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import librosa
import soundfile as sf

from torch.nn.functional import relu, max_pool1d, sigmoid, log_softmax


def double_conv(channels_in, channels_out, kernel_size):
    return nn.Sequential(
        nn.utils.weight_norm(nn.Conv2d(channels_in, channels_out, kernel_size, padding=1)),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(),
        nn.utils.weight_norm(nn.Conv2d(channels_out, channels_out, kernel_size, padding=1)),
        nn.BatchNorm2d(channels_out),
        nn.ReLU()
    )


class UNet_GenGAN(nn.Module):
    def __init__(self, channels_in, channels_out, chs=[8, 16, 32, 64, 128], kernel_size = 3, image_width=64, image_height=64, noise_dim=65, activation='sigmoid', nb_classes=2, embedding_dim=16, use_cond=True):
        super().__init__()
        self.use_cond = use_cond
        self.width = image_width
        self.height = image_height
        self.activation = activation

        # modified noise projection layer
        self.project_noise = nn.Linear(noise_dim, noise_dim)

        # condition projection layer
        self.project_cond = nn.Linear(embedding_dim, image_width//16 * image_height//16)

        self.dconv_down1 = double_conv(channels_in, chs[0], kernel_size)
        self.pool_down1 = nn.MaxPool2d(2, stride=2)

        self.dconv_down2 = double_conv(chs[0], chs[1], kernel_size)
        self.pool_down2 = nn.MaxPool2d(2, stride=2)

        self.dconv_down3 = double_conv(chs[1], chs[2], kernel_size)
        self.pool_down3 = nn.MaxPool2d(2, stride=2)

        self.dconv_down4 = double_conv(chs[2], chs[3], kernel_size)

        self.pool_down4 = nn.MaxPool2d(2, stride=2)

        self.dconv_down5 = double_conv(chs[3], chs[4], kernel_size)

        # modified deconvolution to insert noise
        self.dconv_up5 = double_conv(chs[4]+chs[3]+1, chs[3], kernel_size)
        self.dconv_up4 = double_conv(chs[3]+chs[2], chs[2], kernel_size)
        self.dconv_up3 = double_conv(chs[2]+chs[1], chs[1], kernel_size)
        self.dconv_up2 = double_conv(chs[1]+chs[0], chs[0], kernel_size)
        self.dconv_up1 = nn.Conv2d(chs[0], channels_out, kernel_size=1)

        self.pad = nn.ConstantPad2d((1,0,0,0),0)

    def forward(self, x, z, cond):

        conv1_down = self.dconv_down1(x)
        pool1 = self.pool_down1(conv1_down)

        conv2_down = self.dconv_down2(pool1)
        pool2 = self.pool_down2(conv2_down)

        conv3_down = self.dconv_down3(pool2)
        pool3 = self.pool_down3(conv3_down)

        conv4_down = self.dconv_down4(pool3)
        pool4 = self.pool_down4(conv4_down)

        conv5_down = self.dconv_down5(pool4)
        z = z.reshape(x.shape[0], 1, conv5_down.shape[-2], conv5_down.shape[-1])
        noise = self.project_noise(z)

        conv5_down = torch.cat((conv5_down, noise), dim=1)

        conv5_up = F.interpolate(conv5_down, scale_factor=2, mode='nearest')

        conv5_up = torch.cat((conv4_down, conv5_up), dim=1)

        conv5_up = self.dconv_up5(conv5_up)

        conv4_up = F.interpolate(conv5_up, scale_factor=2, mode='nearest')
        conv4_up = torch.cat((conv3_down, conv4_up), dim=1)
        conv4_up = self.dconv_up4(conv4_up)

        conv3_up = F.interpolate(conv4_up, scale_factor=2, mode='nearest')
        conv3_up = self.pad(conv3_up)

        conv3_up = torch.cat((conv2_down, conv3_up), dim=1)
        conv3_up = self.dconv_up3(conv3_up)

        conv2_up = F.interpolate(conv3_up, scale_factor=2, mode='nearest')
        conv2_up = self.pad(conv2_up)
        conv2_up = torch.cat((conv1_down, conv2_up), dim=1)
        conv2_up = self.dconv_up2(conv2_up)

        conv1_up = self.dconv_up1(conv2_up)

        out = torch.tanh(conv1_up)
        # breakpoint()
        return out


def AlexNet_Discriminator(num_classes):
    model = models.AlexNet(num_classes = num_classes)

    # Make single input channel
    model.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
    # Change number of output classes to num_classes
    model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    return model
