import math
import argparse
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

import jax
import flax
import optax
import models
import orbax.checkpoint as ocp
import time

import pickle

import jax.numpy as jnp
import numpy as np

from tqdm import tqdm
from flax.training import train_state  # Useful dataclass to keep train state
from typing import Any


# ===========================================================
# HERE THERE'S THE PARSER

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log_prova', help='Log dir [default: log_prova]')
parser.add_argument('--model', default='pointnet_basic', help='Model name: pointnet_basic or pointnet [default: pointnet_basic]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]') # default = 1024
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during evaluation [default: 4]')
parser.add_argument('--dump_dir', default='dump_prova', help='dump folder path [default: dump_prova]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--checkpoint', default='250', help='step of the orbax-checkpoint to restore (333:best eval loss model - 777:best eval acc model) [default: 250]')
FLAGS = parser.parse_args()

# ===========================================================

# HERE WE SET THE FLAGS

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
OPTIMIZER = FLAGS.optimizer
MODEL = FLAGS.model
CHECKPOINT = FLAGS.checkpoint

NUM_CLASSES = 40

SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 


# ===========================================================
#  THIS WILL BE USED TO STORE THE RESULT AND TO LOAD THE CHOSEN MODEL

LOG_DIR = FLAGS.log_dir

DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# ===========================================================

# ModelNet40 official train/test split
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

# ===================================================================

# Here you can find functions for saving and loading the model

def save_model(params, model_path):
    """
    Saves the model parameters to a specified path using pickle.

    This function creates the directory at `model_path` if it does not already exist
    and saves the model parameters as a pickle file.

    Args:
        params (Any): The model parameters to be saved. These are typically the weights and biases of the trained model.
        model_path (str): The directory where the model parameters should be saved. If the directory does not exist, it will be created.

    Returns:
        None: The function does not return any value, it just saves the parameters to the specified path.
    """
    os.makedirs(model_path, exist_ok=True)
    model_file = _get_model_file(model_path, 'Flax_Accuracy')

    # You can also use flax's checkpoint package. To show an alternative,
    # you can instead save the parameters simply in a pickle file.
    with open(model_file, 'wb') as f:
        pickle.dump(params, f)



def _get_model_file(model_path, model_name):
    """
    Constructs the full file path for saving the model file.

    This function takes the model path and model name, and returns the full 
    path to the model file with a `.tar` extension.

    Args:
        model_path (str): The directory path where the model file will be saved.
        model_name (str): The name of the model file (without extension).

    Returns:
        str: The full file path, including the directory and the `.tar` file extension.
    """
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")


# ===================================================================
# Here you will find log_string function and schedulers functions

def log_string(out_str):
    """
    Logs a string to both a file and the console.

    Args:
        out_str (str): The message to log.
    """
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

# ===================================================================
# Here there will be the classes used for the architecture.

class TrainState(train_state.TrainState):
    """
    A custom class that extends `TrainState` to store additional attributes.

    This class extends the `TrainState` class to include additional attributes 
    such as `batch_stats` and `key`, which are necessary for tracking the 
    training state during model training.

    Attributes:
        batch_stats (Any): Stores the batch statistics, typically used in batch normalization.
        key (jax.Array): The random key used for dropout and other stochastic operations.
    """
    batch_stats: Any
    key: jax.Array



def create_train_state(module, dict_state, dropout_key, optimizer):
    """
    Creates and returns a custom `TrainState` for model training.

    This function initializes a `TrainState` object with model parameters, batch 
    statistics, optimizer, and other necessary training state attributes. It 
    supports different optimizers like 'momentum' and 'adam' to choose from.

    Args:
        module (flax.linen.Module): The model module that contains the model's parameters.
        dict_state (dict): A dictionary containing the model parameters (`params`) 
                           and batch statistics (`batch_stats`).
        dropout_key (jax.Array): The random key used for dropout during training.
        optimizer (str): The optimizer to use for training. It can be 'momentum' or 'adam'.

    Returns:
        TrainState: A `TrainState` object with the initialized parameters, batch stats, 
                    and optimizer.
    """
    
    model_params = dict_state['params']
    batch_stats = dict_state['batch_stats']
    
    if optimizer == 'momentum':
        tx = optax.inject_hyperparams(optax.sgd)(learning_rate=lr, momentum=bn)
    elif optimizer == 'adam':
        tx = optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=bn)  
    
    return TrainState.create(
        apply_fn=module.apply, params=model_params,
        tx=tx,
        batch_stats=batch_stats,
        key=dropout_key)

# ====================================================================

def EvalModelInBatches(X, Y, state, total_correct, total_seen, total_correct_class, total_seen_class):
    """
    Evaluates the model on batches of data.

    This function processes the input data in batches, applies the model for evaluation, 
    and computes various metrics such as loss, accuracy, and class-wise statistics. 
    It also saves the predicted labels to a file.

    Args:
        X (jax.Array): The input data, typically a batch of point clouds or other features.
        Y (jax.Array): The true labels corresponding to the input data.
        state (train_state.TrainState): The current state of the model, including parameters and batch statistics.
        total_correct (int): The running total of correctly classified samples.
        total_seen (int): The running total of processed samples.
        total_correct_class (np.ndarray): An array tracking correct predictions for each class.
        total_seen_class (np.ndarray): An array tracking the number of samples seen for each class.

    Returns:
        tuple: A tuple containing:
            - loss_sum (float): The total loss computed across all batches.
            - total_correct (int): The total number of correct predictions.
            - total_seen (int): The total number of processed samples.
            - total_correct_class (np.ndarray): Updated array of correct predictions per class.
            - total_seen_class (np.ndarray): Updated array of samples seen per class.
    """
    
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    
    num_votes = 1
    num_batches = math.floor(X.shape[0]//BATCH_SIZE)
    batches = jnp.arange(num_batches) ### Batch Indices

    print(X.shape, X.dtype)
    print(Y.shape, Y.dtype)
    print('Batches: %d' % num_batches)

    loss_sum = 0
    
    for batch in tqdm(batches):
        if batch != batches[-1]:
            start, end = int(batch*BATCH_SIZE), int(batch*BATCH_SIZE+BATCH_SIZE)
        else:
            start, end = int(batch*BATCH_SIZE), X.shape[0]
        
        X_batch, Y_batch = X[start:end], Y[start:end] ## Single batch of data

        cur_batch_size = end - start

        batch_loss_sum = 0 # sum of losses for the batch
        batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
        batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes

        for vote_idx in range(num_votes):
            rotated_data = provider.rotate_point_cloud_by_angle(X_batch,
                                                  vote_idx/float(num_votes) * np.pi * 2)


            loss, correct, class_predicted = eval_step(state, rotated_data, Y_batch, bn)

            batch_pred_sum += loss
            batch_pred_val = class_predicted
            for el_idx in range(cur_batch_size):
                batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += (loss * cur_batch_size / float(num_votes))

        total_correct += correct
        total_seen += cur_batch_size
        loss_sum += batch_loss_sum

        for i in range(cur_batch_size):
            l = Y_batch[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (class_predicted[i] == l)
            fout.write('%d, %d\n' % (class_predicted[i], l))

    return loss_sum, total_correct, total_seen, total_correct_class, total_seen_class


@jax.jit
def eval_step(state: train_state.TrainState, X_batch, Y_batch, bn, reg_weights = 0.001):
    """
    Evaluates the model's performance on a batch of data.

    This function calculates the loss and accuracy for a given batch, 
    along with an optional regularization term based on matrix differential loss.

    Args:
        state (train_state.TrainState): The state of the model, including parameters and batch statistics.
        X_batch (jax.Array): A batch of input data (e.g., point clouds or feature vectors).
        Y_batch (jax.Array): A batch of true labels corresponding to the input data.
        bn (float): The batch normalization decay rate used during inference.
        reg_weights (float, optional): A regularization weight for matrix differential loss. Defaults to 0.001.

    Returns:
        tuple: A tuple containing:
            - loss (float): The total loss, which includes the cross-entropy loss and regularization loss.
            - correct (int): The number of correctly classified samples in the batch.
            - pred_choice (jax.Array): The predicted class indices for each sample in the batch.
    """
    
    mat_diff_loss = 0

    # One-hot encode the labels
    c_one_hot = jax.nn.one_hot(Y_batch, NUM_CLASSES)

    # Apply the model and compute logits and other intermediate variables
    logits, end_points = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        X_batch, training=False, bn_decay=bn)
    
    # Compute softmax predictions and determine the predicted class labels
    preds = jax.nn.softmax(logits, axis=1)
    pred_choice = jnp.argmax(preds, axis=1)
    
    # Count correct predictions
    correct = jnp.sum(pred_choice == Y_batch)

    # Compute the softmax cross-entropy loss
    loss = optax.losses.softmax_cross_entropy(logits=logits, labels=c_one_hot).mean(axis=0)

    # Optionally apply matrix differential loss if available in the end_points
    if bool(end_points):
        transform = end_points['transform']
        k = transform.shape[1]
        
        # Compute the identity matrix
        I = jnp.eye(k)[None, :, :]
        
        # Compute the matrix differential loss
        mat_diff_loss = jnp.mean(jnp.linalg.norm(jnp.matmul(transform, jnp.transpose(transform, (0, 2, 1))) - I, axis=(1, 2)))
    
    # Return total loss (cross-entropy + regularization) and other outputs
    return loss + mat_diff_loss * reg_weights, correct, pred_choice

# ===================================================================

# MAIN

if __name__ == "__main__":
    print("Hi :) !\nLet's start evaluating!\n")

    print("JAX Version : {}".format(jax.__version__))
    print("Flax Version : {}".format(flax.__version__))
    print("Optax Version : {}".format(optax.__version__))
    print("Orbax-Checkpoint Version : {}".format(ocp.__version__))

    seed = 0
    
    if MODEL == 'pointnet_basic':
        model = models.PointNetBasic()
        model_name = 'PointNetBasic'
    elif MODEL == 'pointnet':
        model = models.PointNet()
        model_name = 'PointNet'

    # These are used to save checkpoints
    
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(os.getcwd()+'/'+ LOG_DIR, options = options)

    restored_checkpoint = checkpoint_manager.restore(CHECKPOINT, args = ocp.args.PyTreeRestore())
    
    dict_state = restored_checkpoint['model']
    bn = restored_checkpoint['bn_story']
    lr = restored_checkpoint['lr_story']

    seed_key = jax.random.PRNGKey(0)
    _, dropout_key = jax.random.split(seed_key, num = 2)


    state = create_train_state(model,dict_state,dropout_key,OPTIMIZER)
 
    beginning = time.time()

    log_string('-------'*3)

    total_correct = 0
    total_seen = 0
    loss_val = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    for fn in range(len(TEST_FILES)):
        log_string("===FILE TEST: %d ===" %fn)

        current_data_test, current_label_test = provider.loadDataFile(TEST_FILES[fn])
        current_data_test = current_data_test[:,0:NUM_POINT,:]
        current_label_test = np.squeeze(current_label_test)
        
        loss_aux, total_correct, total_seen, total_correct_class, total_seen_class  = EvalModelInBatches(current_data_test, current_label_test, state, total_correct, total_seen, total_correct_class, total_seen_class)
        loss_val += loss_aux

    loss = loss_val / float(total_seen)
    accuracy = (total_correct / float(total_seen))

    log_string('eval mean loss: {:.6f}'.format(loss))
    log_string('eval accuracy: {:.3f}'.format(accuracy*100))
    log_string('eval avg class acc: {:.3f}\n'.format((np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float32))) * 100))

    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))

    log_string('-------'*3)
    log_string('Time for Evaluating: %f'% (time.time() - beginning))

    save_model(class_accuracies,DUMP_DIR)
    
    log_string('Loaded Model ' + str(CHECKPOINT))

    LOG_FOUT.close()

    sys.modules[__name__].__dict__.clear()
