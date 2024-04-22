import os
from argparse import ArgumentParser

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset
import torch.nn as nn

from DeDoDe.train import train_k_steps
from DeDoDe.datasets.megadepth import MegadepthBuilder
from DeDoDe.descriptors.descriptor_loss import DescriptorLoss
from DeDoDe.checkpoint import CheckPoint
from DeDoDe.descriptors.dedode_descriptor import DeDoDeDescriptor
from DeDoDe.encoder import VGG
from DeDoDe.decoder import ConvRefiner, Decoder
from DeDoDe import dedode_detector_S, dedode_descriptor_B
from DeDoDe.benchmarks import MegaDepthPoseMNNBenchmark
#from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
#from DeDoDe.matchers.soft_dual_softmax_matcher import SoftDualSoftMaxMatcher


from DeDoDe.utils import *
from PIL import Image
import cv2
import numpy as np


if __name__ == "__main__":
    device = get_best_device()
    detector = dedode_detector_S(weights = torch.load("dedode_detector_S_v2.pth", map_location = device))
    descriptor = dedode_descriptor_B(weights = torch.load("dedode_descriptor_B.pth", map_location = device))
    matcher = DualSoftMaxMatcher()

    mega_1500 = MegaDepthPoseMNNBenchmark()
    mega_1500.benchmark(
        detector_model = detector,
        descriptor_model = descriptor,
        matcher_model = matcher)