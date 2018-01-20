from models import LeNet
from loader import DataLoader
import argparse


DataLoader.load_scaled_data_with_labels('dataset/training_data/', 50, 50)
