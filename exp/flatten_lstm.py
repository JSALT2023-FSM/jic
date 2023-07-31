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
from jic import full_lstm, models

import dataclasses
# Disallow TensorFlow from using GPU so that it won't interfere with JAX.
tf.config.set_visible_devices([], 'GPU')
if not jax.devices('gpu'):
  raise RuntimeError('We recommend using a GPU to run this notebook')

checkpoint_dir = sys.argv[1]
if not os.path.exists(checkpoint_dir) : 
    os.makedirs(checkpoint_dir)

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
sizes =[100,150,250,400,600]

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
    is_train: bool ,
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
    else :
        table = table.sample(frac = 1, random_state= 22)
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

train_all = create_dict([train_csv, train_synthetic])
test_all = create_dict([test_csv], test=True)
#test_all = create_dict([train_csv, train_synthetic])
dev_all = create_dict([dev_csv], test=True)
train_dataset = tf.data.Dataset.from_tensor_slices(train_all)
test_dataset = tf.data.Dataset.from_tensor_slices(test_all)
dev_dataset = tf.data.Dataset.from_tensor_slices(dev_all)


TEST_BATCH_SPLIT = 2
# A single test batch.
INIT_BATCH = next(
    test_dataset.take(TEST_BATCH_SPLIT)
    .apply(functools.partial(preprocess, batch_size=TEST_BATCH_SPLIT, is_train=False))
    .as_numpy_iterator()
)
dev_dataset = (
    dev_dataset.apply(functools.partial(preprocess, is_train=False))
    .prefetch(tf.data.AUTOTUNE)
)
test_dataset = (
    test_dataset.apply(functools.partial(preprocess, is_train=False))
    .prefetch(tf.data.AUTOTUNE)
)

# An iterator of training batches.

train_dataset = (
    train_dataset.apply(functools.partial(preprocess, is_train=True))
    .prefetch(tf.data.AUTOTUNE)
)
TRAIN_BATCHES = train_dataset.as_numpy_iterator()

options = orbax.checkpoint.CheckpointManagerOptions(save_interval_steps=2, max_to_keep=10)
mngr = orbax.checkpoint.CheckpointManager(
    checkpoint_dir, orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), options)

with open('/gpfsscratch/rech/nou/uzn19yk/jax_intent_classification/test_out.pickle', 'rb') as f:
  lattice_config, lattice_params = pickle.load(f)

lattice = lattice_config.build()

def count_number_params(params):
    return sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x : x.size,params)))

def compute_accuracies(intents, batch, loss):
    intents_results = intents == batch["intent"]
    return {"intents" : intents_results, "loss": loss }
decoder_params = lattice_params["params"]
scheduler = optax.exponential_decay(
    init_value=2e-4,
    transition_steps=1000,
    decay_rate=0.99)

def train_and_eval(
    INIT_BATCH,
    test_dataset,
    dev_dataset,
    train_batches,
    model,
    step,
    optimizer=optax.chain(
    optax.clip_by_global_norm(5.0),
    optax.scale_by_schedule(scheduler),
    optax.adam(2e-4)) , 
    num_steps=140000,
    num_steps_per_eval=10000,
):
  # Initialize the model parameters using a fixed RNG seed. Flax linen Modules
  # need to know the shape and dtype of its input to initialize the parameters,
  # we thus pass it the test batch.
  train_rng = jax.random.PRNGKey(22)

  if step is None : 
      params = jax.jit(model.init)(jax.random.PRNGKey(0), INIT_BATCH["encoder_frames"], INIT_BATCH["num_frames"]).unfreeze()
      import pprint
      print(f" number of params total : {count_number_params(params) - count_number_params(lattice_params)}")
      params["params"]["lattice"] = lattice_params["params"]
      num_done_steps = 0
      opt_state = optimizer.init(params)
  else : 
      print(f"loading step {step}")
      num_done_steps = step
      params = jax.jit(model.init)(jax.random.PRNGKey(0), INIT_BATCH["encoder_frames"], INIT_BATCH["num_frames"])
      #params = model.init(jax.random.PRNGKey(0), INIT_BATCH["encoder_frames"], INIT_BATCH["num_frames"])
      opt_state = optimizer.init(params)
      params,opt_state = mngr.restore(step, items = [params, opt_state])
  # jax.jit compiles a JAX function to speed up execution.
  # `donate_argnums=(0, 1)` means we know that the input `params` and
  # `opt_state` won't be needed after calling `train_step`, so we donate them
  # and allow JAX to use their memory for storing the output.
  @functools.partial(jax.jit, donate_argnums=(0, 1))
  def train_step(params, opt_state,rng, batch):
    # Compute the loss value and the gradients.
    print(f" batch is {batch['encoder_frames'].shape}", flush =True)
    print(batch)
    def loss_fn(params, rng) : 
        intent_logits = model.apply(params,batch["encoder_frames"], batch["num_frames"], is_test=False, rngs={"dropout": rng}) 
        loss_intent = optax.softmax_cross_entropy_with_integer_labels(intent_logits, batch["intent"])
        return jnp.mean(loss_intent)
    next_rng,rng = jax.random.split(rng)
    loss, grads = jax.value_and_grad(loss_fn)(params, rng)

    # Compute the actual updates based on the optimizer state and the gradients.
    updates, opt_state = optimizer.update(grads, opt_state, params)
    # Apply the updates.
    params = optax.apply_updates(params, updates)
    #Uncomment the next line if you want to freeze the lattice decoder
    #params['params']['lattice'] = decoder_params
    return params, opt_state, next_rng, {'loss': loss, "grads": optax.global_norm(grads)}

  # We are not passing additional arguments to jax.jit, so it can be used
  # directly as a function decorator.
  @jax.jit
  def eval_step(params, batch):
    intents_logits= model.apply(params, batch["encoder_frames"], batch["num_frames"], is_test=True)
    test_loss = optax.softmax_cross_entropy_with_integer_labels(intents_logits, batch["intent"])
    # Test accuracy.
    intents = jnp.argmax(intents_logits, axis = 1)
    #Compute accuracies 
    return compute_accuracies(intents, batch, test_loss)  

  while num_done_steps < num_steps:
    for step in tqdm(range(num_steps_per_eval), ascii=True):
      params, opt_state,train_rng, train_metrics = train_step(
          params, opt_state, train_rng, next(train_batches)
      )

    eval_metrics = { "intents" :[], "loss": [] }
    print("running the validation")
    for test_batch in tqdm(dev_dataset.as_numpy_iterator()) : 
        eval_metrics_step = eval_step(params, test_batch)
        for i in eval_metrics : 
            eval_metrics[i].append(eval_metrics_step[i])
       

    num_done_steps += num_steps_per_eval

    mngr.save(num_done_steps,[params, opt_state])
    print(f'step {num_done_steps}\ttrain {train_metrics}')
    log_file_path = os.path.join(checkpoint_dir, "log_file.txt")
    with open(log_file_path, "a") as log_file : 

        log_file.write(f"step {num_done_steps}\ttrain {train_metrics} \t eval loss : {jnp.mean(jnp.concatenate(eval_metrics['loss']))} \t eval_accuracy {jnp.mean(jnp.concatenate(eval_metrics['intents']))}")
        log_file.write("\n")
    for i in eval_metrics : 
        print(f" {i} : {jnp.mean(jnp.concatenate(eval_metrics[i]))}")
  
  eval_metrics = { "intents" :[], "loss": [] }
  print("running the test inferences")
  for test_batch in tqdm(test_dataset.as_numpy_iterator()) : 
      eval_metrics_step = eval_step(params, test_batch)
      for i in eval_metrics : 
         eval_metrics[i].append(eval_metrics_step[i])
       

  log_file_path_test = os.path.join(checkpoint_dir, "test_log_file.txt")
  with open(log_file_path_test, "a") as log_file : 
      log_file.write(f"step {num_done_steps}\t test loss : {jnp.mean(jnp.concatenate(eval_metrics['loss']))} \t eval_accuracy {jnp.mean(jnp.concatenate(eval_metrics['intents']))}")
      log_file.write("\n")
  for i in eval_metrics : 
      print(f" {i} : {jnp.mean(jnp.array(eval_metrics[i]))}")
 

model = full_lstm.Model(lattice=lattice, classifier = models.IntentClassifier())
step = mngr.latest_step()
#import pdb; pdb.set_trace()
train_and_eval(INIT_BATCH, test_dataset,dev_dataset, TRAIN_BATCHES, model, step)
