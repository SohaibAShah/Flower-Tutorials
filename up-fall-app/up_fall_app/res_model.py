from turtle import pd
import torch, random, numpy as np
import torch.nn as nn, os, csv
from utils import CustomDataset, display_result, scaled_data
from fl_simple import Server, Client
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm # 1. Import tqdm


class ModelCSVIMG(nn.Module):
    def __init__(self, num_csv_features, img_shape1, img_shape2):
        super(ModelCSVIMG, self).__init__()

        # --- Branch 1: For processing numerical CSV data ---
        self.csv_fc_1 = nn.Linear(num_csv_features, 2000)
        self.csv_bn_1 = nn.BatchNorm1d(2000)
        self.csv_fc_2 = nn.Linear(2000, 600)
        self.csv_bn_2 = nn.BatchNorm1d(600)
        self.csv_dropout = nn.Dropout(0.2)

        # --- Branch 2: For processing images from Camera 1 (CNN) ---
        self.img1_conv_1 = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.img1_batch_norm = nn.BatchNorm2d(18)
        self.img1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Flattened features from the CNN go into a fully connected layer
        self.img1_fc1 = nn.Linear(18 * 16 * 16, 100)
        self.img1_dropout = nn.Dropout(0.2)

        # --- Branch 3: For processing images from Camera 2 (identical to Branch 2) ---
        self.img2_conv = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.img2_batch_norm = nn.BatchNorm2d(18)
        self.img2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img2_fc1 = nn.Linear(18 * 16 * 16, 100)
        self.img2_dropout = nn.Dropout(0.2)

        # --- Fusion and Final Classification Layers ---
        # The input size is 600 (from CSV) + 100 (from Image 1) + 100 (from Image 2) = 800
        self.fc1 = nn.Linear(800, 1200)
        self.dr1 = nn.Dropout(0.2)
        # A residual connection is used here: input to fc2 is the original 800 + output of fc1 (1200) = 2000
        self.fc2 = nn.Linear(2000, 12) # 12 output classes

    def forward(self, x_csv, x_img1, x_img2):
        # --- Process CSV data ---
        x_csv = F.relu(self.csv_bn_1(self.csv_fc_1(x_csv)))
        x_csv = F.relu(self.csv_bn_2(self.csv_fc_2(x_csv)))
        x_csv = self.csv_dropout(x_csv)

        # --- Process Image 1 data ---
        # Reshape image from (batch, height, width, channels) to (batch, channels, height, width)
        x_img1 = x_img1.permute(0, 3, 1, 2)
        x_img1 = F.relu(self.img1_conv_1(x_img1))
        x_img1 = self.img1_batch_norm(x_img1)
        x_img1 = self.img1_pool(x_img1)
        x_img1 = x_img1.contiguous().view(x_img1.size(0), -1) # Flatten
        x_img1 = F.relu(self.img1_fc1(x_img1))
        x_img1 = self.img1_dropout(x_img1)

        # --- Process Image 2 data ---
        x_img2 = x_img2.permute(0, 3, 1, 2)
        x_img2 = F.relu(self.img2_conv(x_img2))
        x_img2 = self.img2_batch_norm(x_img2)
        x_img2 = self.img2_pool(x_img2)
        x_img2 = x_img2.contiguous().view(x_img2.size(0), -1) # Flatten
        x_img2 = F.relu(self.img2_fc1(x_img2))
        x_img2 = self.img2_dropout(x_img2)

        # --- Fusion ---
        x = torch.cat((x_csv, x_img1, x_img2), dim=1)
        residual = x # Keep a copy for the residual connection
        
        # --- Final layers ---
        x = F.relu(self.fc1(x))
        x = self.dr1(x)
        # Concatenate the residual connection
        x = torch.cat((residual, x), dim=1)
        # Final output with softmax for classification
        x = F.softmax(self.fc2(x), dim=1)

        return x