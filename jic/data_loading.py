import functools
import flax.linen as nn
import jax
import sys
import jax.numpy as jnp
import last
import optax
import tensorflow as tf
import os
import tensorflow_datasets as tfds
import numpy as np
import pickle
import orbax.checkpoint
import pandas as pd
from jic.models import RecognitionLatticeConfig
from tqdm import tqdm
from utils import get_scenario_action
from jic import models

import dataclasses
# Disallow TensorFlow from using GPU so that it won't interfere with JAX.
train_csvs_dir  = "/gpfsstore/rech/nou/uzn19yk/JSALT/data/slurp_csvs/"
encoder_dir = "/gpfsscratch/rech/nou/uzn19yk/encoder_outputs/"

train_csv = f"{train_csvs_dir}/train_real-type=direct.csv"
train_synthetic= f"{train_csvs_dir}/train_synthetic-type=direct.csv"
test_csv= f"{train_csvs_dir}/test-type=direct.csv"
dev_csv= f"{train_csvs_dir}/devel-type=direct.csv"

def create_label_encoders(train_csvs) :
  actions_dict = {}
  scen_dict ={}
  intent_dict ={}
  intent_count = 0
  act_count = 0
  scen_count = 0
  tables = []
  for train_csv in train_csvs : 
    tables.append (pd.read_csv(train_csv))
  table = pd.concat(tables)
  semantics = list(table["semantics"])
  actions = []
  scenarios = []
  intents = []
  for sem in semantics : 
      scenario, action, intent = get_scenario_action(sem)
      intents.append(intent)
      actions.append(action)
      scenarios.append(scenario)
  unique_actions = set(actions)
  unique_scenarios = set(scenarios)
  unique_intents = set(intents)
  for act in unique_actions : 
      actions_dict[act] = act_count 
      act_count+=1
  for scen in unique_scenarios : 
      scen_dict[scen] = scen_count 
      scen_count+=1
  for intent in unique_intents : 
      intent_dict[intent] = intent_count
      intent_count +=1
      
  return scen_dict,actions_dict, scen_count, act_count, intent_dict, intent_count


scen_dict, actions_dict, scen_count,act_count, intent_dict, intent_count =create_label_encoders([train_csv, train_synthetic]) 
with open(os.path.join(checkpoint_dir, "intent_dict.pickle"), "wb") as pickle_out : 
    pickle.dump(intent_dict, pickle_out)
print(f" total number of scenarios : {scen_count}")
print(f" total number of actions : {act_count}")
print(f" total number of intents : {intent_count}")


def preprocess_example_tf(wav, scenario, action, intent, ids):
    b= tf.numpy_function( numpy_preprocess, [wav, scenario, action, intent, ids], [tf.float32, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64])
    return {"encoder_frames": b[0], "scenario": b[1], "action" : b[2],"intent": b[3], 'num_frames': b[4], "ids" : b[5]}

def numpy_preprocess(wav, scenario, action, intent, ids):
  encoder_frames = np.squeeze(np.load(os.path.join(encoder_dir, wav.decode("utf-8")+".npy")))
  action_label, scenario_label, intent_label = actions_dict[action.decode("utf-8")], scen_dict[scenario.decode("utf-8")], intent_dict[intent.decode("utf-8")]
  return encoder_frames,scenario_label, action_label, intent_label, encoder_frames.shape[0], ids
sizes =[150,250,400,600]

def shorten_batch(batch, sizes=sizes): 
    num_frames = tf.math.reduce_max(batch["num_frames"])
    sizes = tf.constant(sizes, dtype=tf.int64)
    valid_sizes = sizes >= num_frames
    min_valid_size = tf.math.reduce_min(tf.where(valid_sizes, sizes, tf.math.reduce_max(sizes)))
    new_batch = dict(batch)
    new_batch["encoder_frames"] = batch["encoder_frames"][:,:min_valid_size]
    return new_batch
def preprocess(
    dataset: tf.data.Dataset,
    is_train: bool = True,
    batch_size: int = 6,
    max_num_frames: int = 600
) -> tf.data.Dataset:
  """Applies data preprocessing for training and evaluation."""
  # Preprocess individual examples.
  if is_train:
    dataset = dataset.shuffle(buffer_size=1000).repeat()
  dataset = dataset.map(preprocess_example_tf, num_parallel_calls=tf.data.AUTOTUNE)
  # Shuffle and repeat data for training.
  # Pad and batch examples.
  dataset = dataset.padded_batch(
      batch_size,
      {
      'encoder_frames' : [max_num_frames, None],
      'action' : [],
      'scenario': [],
      "intent": [],
      'num_frames': [],
      'ids' : []
      },
  )
  dataset = dataset.map(shorten_batch, num_parallel_calls=tf.data.AUTOTUNE)
  return dataset

def create_dict(train_csvs, test= False): 
    tables = []
    for train_csv in train_csvs : 
        tables.append (pd.read_csv(train_csv))
    table = pd.concat(tables)
    if test : 
        table = table[0:13074]
    semantics = table["semantics"]
    ids = table["ID"]
    file_ids = [x.split("/")[-1] for x in list(table["wav"])]
    actions = []
    scenarios = []
    intents = []
    for sem in semantics : 
        scenario, action, intent = get_scenario_action(sem)
        actions.append(action)
        scenarios.append(scenario)
        intents.append(intent)
    return  file_ids, scenarios, actions, intents, ids


