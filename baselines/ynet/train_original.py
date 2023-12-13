import pandas as pd
import yaml
import argparse
import torch
from original.model import YNet
import pickle5 as pickle

CONFIG_FILE_PATH = 'original/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = 'sdd_trajnet'  # arbitrary name for this experiment
DATASET_NAME = 'sdd'

TRAIN_DATA_PATH = 'data/SDD/train_trajnet.pkl'
TRAIN_IMAGE_PATH = 'data/SDD/train'
VAL_DATA_PATH = 'data/SDD/test_trajnet.pkl'
VAL_IMAGE_PATH = 'data/SDD/test'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

BATCH_SIZE = 4

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('original/')[1]

df_train = pickle.load(open(TRAIN_DATA_PATH, "rb"))#pd.read_pickle(TRAIN_DATA_PATH)
df_val = pickle.load(open(VAL_DATA_PATH, "rb"))

print(df_train.head())

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)


model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ,
            device=None, dataset_name=DATASET_NAME)