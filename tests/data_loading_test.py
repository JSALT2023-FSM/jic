import jax
import jax.numpy as jnp
import numpy.testing as npt

from jic import models
from jic.data_loading import create_label_encoders, create_dict 
train_csvs_dir  = "/gpfsstore/rech/nou/uzn19yk/JSALT/data/slurp_csvs/"
encoder_dir = "/gpfsscratch/rech/nou/uzn19yk/encoder_outputs/"

train_csv = f"{train_csvs_dir}/train_real-type=direct.csv"
train_synthetic= f"{train_csvs_dir}/train_synthetic-type=direct.csv"
test_csv= f"{train_csvs_dir}/test-type=direct.csv"
dev_csv= f"{train_csvs_dir}/devel-type=direct.csv"


