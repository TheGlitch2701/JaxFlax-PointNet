import math
import argparse
import sys
import os

import orbax.checkpoint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider


# ===========================================================
# HERE THERE'S THE PARSER

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log_prova', help='Log dir [default: log_Prova]')
parser.add_argument('--model', default='pointnet_basic', help='Model name: pointnet_basic or pointnet [default: pointnet_basic]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]') # default = 1024
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]') # default = 250
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

# ===========================================================

# HERE WE SET THE FLAGS

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MODEL = FLAGS.model

NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# ===========================================================
#  THIS WILL BE USED TO STORE THE RESULT AND TO LOAD THE CHOSEN MODEL

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# ===========================================================

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

# ======================================
# THIS ARE THE TEMPORARY IMPORTS

import jax
import flax
import optax
import orbax.checkpoint as ocp
import time

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from flax.training import train_state  # Useful dataclass to keep train state
from typing import Any

import models

# ===================================================================
# Here you will find log_string function and schedulers functions

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def create_momentum_fn():
    """Creates momentum schedule."""
        
    exp_fn = optax.exponential_decay(
        init_value=BN_INIT_DECAY, # 0.5 
        decay_rate= BN_DECAY_DECAY_RATE, # 0.5
        staircase=True,
        transition_steps= BN_DECAY_DECAY_STEP)# 200000
        
    
    return exp_fn


def create_learning_rate_fn():
    """Creates learning rate schedule."""
        
    exp_fn = optax.exponential_decay(
        init_value=BASE_LEARNING_RATE,
        decay_rate=DECAY_RATE,
        staircase=True,
        transition_steps= DECAY_STEP,
        end_value=0.00001)
    
    return exp_fn

# ===================================================================
# Here there will be the classes used for the architecture.

class TrainState(train_state.TrainState):
    batch_stats: Any
    key: jax.Array


def create_train_state(module, rng, dropout_key, optimizer):
    """Creates an initial `TrainState`."""
    variables = module.init(rng, jnp.ones([1,NUM_POINT,3]), training = False, bn_decay = BN_INIT_DECAY) # initialize parameters by passing a template image
    
    model_params = variables['params']
    batch_stats = variables['batch_stats']
    
    if optimizer == 'momentum':
        tx = optax.inject_hyperparams(optax.sgd)(learning_rate = BASE_LEARNING_RATE, momentum = BN_INIT_DECAY)
    elif optimizer == 'adam':
        tx = optax.inject_hyperparams(optax.adamw)(learning_rate = BASE_LEARNING_RATE, weight_decay = BN_INIT_DECAY)  
    
    return TrainState.create(
        apply_fn=module.apply, params=model_params,
        tx=tx,
        batch_stats = batch_stats,
        key = dropout_key)


@jax.jit
def train_step(state: train_state.TrainState, X_batch, Y_batch, bn):
    
    dropout_train_key = jax.random.fold_in(key=state.key, data=state.step)

    def loss_fn(params, x, c, bn, reg_weights = 1e-3):
        
        mat_diff_loss = 0
        c_one_hot = jax.nn.one_hot(c, NUM_CLASSES) # one hot encode the class index

        (final_layer, end_points), updates = state.apply_fn({'params': params, 'batch_stats':state.batch_stats}, 
                                                  x, training = True, bn_decay = bn , mutable = ['batch_stats'],
                                                  rngs = {'dropout': dropout_train_key})
            

        preds = jax.nn.softmax(final_layer, axis = 1) # (32, 40)
        preds = jnp.argmax(preds, axis = 1) # preds = (32, 1)
        
        correct = jnp.sum(preds == c)
        loss = optax.losses.softmax_cross_entropy(logits=final_layer, labels=c_one_hot).mean(axis = 0) # scalar


        if(bool(end_points)):
            
            transform = end_points['transform']
            k = transform.shape[1]

            I = jnp.eye(k)[None, :, :]
            mat_diff_loss = jnp.mean(jnp.linalg.norm(jnp.matmul(transform, jnp.transpose(transform, (0, 2, 1))) - I, axis=(1, 2)))

        return loss + mat_diff_loss * reg_weights, (correct, updates)


    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (correct, updates)), grads = gradient_fn(state.params, X_batch, Y_batch, bn)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])

    return state, loss, correct

@jax.jit
def eval_step(state: train_state.TrainState, X_batch, Y_batch, bn, reg_weights = 1e-3):
    """Validation for a single step."""

    mat_diff_loss = 0

    c_one_hot = jax.nn.one_hot(Y_batch, NUM_CLASSES) # one hot encode the class index

    logits, end_points = state.apply_fn(
        {'params': state.params, 'batch_stats':state.batch_stats},
        X_batch, training = False, bn_decay = bn)
    
    preds = jax.nn.softmax(logits, axis = 1) # (32, 40)
    pred_choice = jnp.argmax(preds, axis = 1) # preds = (32, 1)
    
    correct = jnp.sum(pred_choice == Y_batch)
    loss = optax.losses.softmax_cross_entropy(logits=logits, labels=c_one_hot).mean(axis = 0) # scalar

    if(bool(end_points)):
        
        transform = end_points['transform']
        k = transform.shape[1]

        I = jnp.eye(k)[None, :, :]
        mat_diff_loss = jnp.mean(jnp.linalg.norm(jnp.matmul(transform, jnp.transpose(transform, (0, 2, 1))) - I, axis=(1, 2)))
            
    return loss + mat_diff_loss * reg_weights, correct, pred_choice


# ====================================================================

# Here you can find usefull functions for computing the loss function and
# for training and validating


# Training Loop across each batch (Computes Loss & Accuracy of each TRAIN FILE | Updates optimizer state and weights) 
def TrainModelInBatches(X, Y, state, learning_rate_fn, momentum_fn):
    
    num_batches = math.floor(X.shape[0]//BATCH_SIZE)
    batches = jnp.arange(num_batches) ### Batch Indices

    print(X.shape,X.dtype)
    print(Y.shape,Y.dtype)
    print('Batches: %d' % num_batches)

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    losses = [] ## Record loss of each batch
    for batch in tqdm(batches): # batches
        if batch != batches[-1]:
            start, end = int(batch*BATCH_SIZE), int(batch*BATCH_SIZE+BATCH_SIZE)
        else:
            start, end = int(batch*BATCH_SIZE), X.shape[0] 


        X_batch, Y_batch = X[start:end], Y[start:end] ## Single batch of data
        X_batch = provider.rotate_point_cloud(X_batch)
        X_batch = provider.jitter_point_cloud(X_batch)
        
        bn = min(BN_DECAY_CLIP, 1 - momentum_fn(BATCH_SIZE * state.step))

        state, loss, correct = train_step(state, X_batch, Y_batch, bn)

        lr = learning_rate_fn(BATCH_SIZE * state.step)
        bn = min(BN_DECAY_CLIP, 1 - momentum_fn(BATCH_SIZE * state.step))
        
        state.opt_state.hyperparams['learning_rate'] = lr
        if OPTIMIZER == 'adam':
            state.opt_state.hyperparams['weight_decay'] = bn
        elif OPTIMIZER == 'momentum':
            state.opt_state.hyperparams['momentum'] =  bn

        total_correct += correct
        total_seen += (end - start)
        loss_sum += loss

        losses.append(loss) ## Record Loss

        # Updating each story for each batch of objects
        
        lr_story.append(lr)
        bn_story.append(bn)
    
    train_loss = np.array(losses).mean()
    train_acc = (total_correct / float(total_seen))

    log_string("CrossEntropyLoss : {:.6f}".format(train_loss))
    log_string('mean loss: {:.6f}'.format(loss_sum / float(num_batches)))
    log_string('accuracy: {:.3f} %\n'.format(train_acc*100))
    
    return state, train_loss, train_acc


# Validating Loop across each batch (Computes Loss & Accuracy of each TEST FILE | NO update of optimizer state and weights)
def EvalModelInBatches(X, Y, state, total_correct, total_seen, total_correct_class, total_seen_class):
    
    num_batches = math.floor(X.shape[0]//BATCH_SIZE)
    batches = jnp.arange(num_batches) ### Batch Indices

    print(X.shape,X.dtype)
    print(Y.shape,Y.dtype)
    print('Batches: %d' % num_batches)

    loss_sum = 0
    

    for batch in tqdm(batches):
        if batch != batches[-1]:
            start, end = int(batch*BATCH_SIZE), int(batch*BATCH_SIZE+BATCH_SIZE)
        else:
            start, end = int(batch*BATCH_SIZE), X.shape[0]
        
        X_batch, Y_batch = X[start:end], Y[start:end] ## Single batch of data

        if OPTIMIZER == 'adam':
            bn = state.opt_state.hyperparams['weight_decay']
        elif OPTIMIZER == 'momentum':
            bn = state.opt_state.hyperparams['momentum']

        loss, correct, class_predicted = eval_step(state, X_batch, Y_batch, bn)


        total_correct += correct
        total_seen += (end - start)
        loss_sum += loss * (end - start)

        
        for i in range(end - start):
            l = Y_batch[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (class_predicted[i] == l)
    

    return loss_sum, total_correct, total_seen, total_correct_class, total_seen_class



# Simple function which save some usefull figures for comparison and visualisation
def figures(train_loss, eval_loss, train_accuracy, eval_accuracy, lr_story, bn_story):

    # Train/Validation Losses & Accuracies
    fig, axs = plt.subplots(2,2)

    axs[0,0].plot(range(1,len(train_loss) + 1), train_loss, 'b-*')
    axs[0,0].set_title('Loss')
    axs[0,0].set_ylabel('Train')
    axs[0,0].grid(visible = True)

    axs[1,0].plot(range(1,len(eval_loss)+ 1), eval_loss, 'r-*')
    axs[1,0].set_ylabel('Validation')
    axs[1,0].grid(visible = True)

    axs[0,1].plot(range(1,len(train_accuracy)+ 1), train_accuracy, 'b-*')
    axs[0,1].set_title('Accuracy')
    axs[0,1].grid(visible = True)

    axs[1,1].plot(range(1,len(eval_accuracy) + 1), eval_accuracy, 'r-*')
    axs[1,1].grid(visible = True)
    plt.savefig(LOG_DIR + '/Loss & Accuracy')
    # plt.show()

    # # Learning Rate History
    # fig, ax = plt.subplots()
    # ax.plot(range(1,len(lr_story) + 1), lr_story, marker = '*', linestyle = '')
    # ax.set_title('Learning Rate History')
    # ax.set_xlabel('Steps')
    # ax.grid(visible = True)
    # plt.savefig(LOG_DIR + '/Learning Rate History')
    # # plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(range(1,len(bn_story) + 1), bn_story, marker = '*', linestyle = '')
    # ax.set_title('BN_decay History')
    # ax.set_xlabel('Steps')
    # ax.grid(visible = True)
    # plt.savefig(LOG_DIR + '/BN_decay History')
    # # plt.show()

# ===================================================================

# MAIN

if __name__ == "__main__":
    print("Hi :) !\nLet's start working!\n")

    print("JAX Version : {}".format(jax.__version__))
    print("Flax Version : {}".format(flax.__version__))
    print("Optax Version : {}".format(optax.__version__))
    print("Orbax-Checkpoint Version : {}".format(ocp.__version__))

    # These are used to save checkpoints
    
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(os.getcwd()+'/'+ LOG_DIR, options = options)

    options_best_acc = ocp.CheckpointManagerOptions(max_to_keep = 1, create = True)
    best_checkpoint_acc = ocp.CheckpointManager(os.getcwd()+'/'+ LOG_DIR, options = options_best_acc)
    
    # ============================================= SETUP ======================================================
    seed = 0
    step = 0
    partial_time = 0
    aux_opt = OPTIMIZER
    seed_key = jax.random.PRNGKey(seed)
    param_key, dropout_key = jax.random.split(seed_key, num = 2)


    if MODEL == 'pointnet_basic':
        model = models.PointNetBasic()
        model_name = 'PointNetBasic'
    elif MODEL == 'pointnet':
        model = models.PointNet()
        model_name = 'PointNet'
    
    # This function create a train_step where inside opt_state we have the optimizer state (adam weights and bias)
    # in weights we have the actual weights and bias of the structure
    # and inside train_step we have a lambda function which call the computation done during training time

    state = create_train_state(model,param_key,dropout_key,aux_opt)


    # =========================================== TRAINING & VALIDATION LOOPS ===================================== 

    beginning = time.time()

    aux_loss = []
    aux_acc = []

    lr_story = []
    bn_story = []

    train_loss = []
    train_accuracy = []
    eval_loss = []
    eval_accuracy = []
    best_loss_eval = 100.
    best_eval_acc = 0.

    for epoch in range(0, MAX_EPOCH + 1):

        start = time.time()

        train_file_idxs = np.arange(0, len(TRAIN_FILES))
        np.random.shuffle(train_file_idxs)

        log_string("===EPOCH %d===\n" %epoch)
    
        for fn in range(len(TRAIN_FILES)):
            log_string("===FILE USED: N° %d==="% train_file_idxs[fn])

            current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
            current_data = current_data[:,0:NUM_POINT,:]
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
            current_label = np.squeeze(current_label)

            state, loss, accuracy = TrainModelInBatches(current_data, current_label, state, create_learning_rate_fn(), create_momentum_fn())
            aux_loss.append(loss)
            aux_acc.append(accuracy)
        
        train_loss.append(np.mean(aux_loss))
        train_accuracy.append(np.mean(aux_acc))
        aux_loss.clear()
        aux_loss.clear()

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

        eval_accuracy.append(accuracy)
        eval_loss.append(loss)

        log_string('-------'*3)
        

        partial_time += time.time() - start

        # Here there is the implementation of the checkpoint saver  
        if (eval_accuracy[-1] > best_eval_acc):
            ckpt = {'model': state,
                    'train_acc':[train_accuracy], 'train_loss':[train_loss],
                    'eval_acc':[eval_accuracy], 'eval_loss':[eval_loss],
                    'lr_story':lr_story[-1], 'bn_story':bn_story[-1],
                    'partial_time': partial_time}
            
            best_checkpoint_acc.save(epoch, args=ocp.args.PyTreeSave(ckpt))
            best_checkpoint_acc.wait_until_finished()
            log_string('\nBEST ACCURACY MODEL SAVED (' + str(epoch) + ') !\n')
            best_eval_acc = accuracy
        # if (epoch==250): # (epoch) % 10 == 0 or
        #     ckpt = {'model': state,
        #             'train_acc':[train_accuracy], 'train_loss':[train_loss],
        #             'eval_acc':[eval_accuracy], 'eval_loss':[eval_loss],
        #             'lr_story':lr_story[-1], 'bn_story':bn_story[-1],
        #             'partial_time': partial_time}
            
        #     checkpoint_manager.save(epoch, args=ocp.args.PyTreeSave(ckpt))
        #     checkpoint_manager.wait_until_finished()
        #     log_string('\nMODEL SAVED!\n')

        log_string('Time Spent for epoch %d:\t%f' % (epoch, (time.time() - start)))
    

    log_string('Maximum Training Accuracy Reached: {0:.3f}\tEpoch: {1:d}'.format(np.max(np.array(train_accuracy) * 100), np.argmax(np.array(train_accuracy) * 100)))
    log_string('Maximum Validation Accuracy Reached: {0:.3f}\tEpoch: {1:d}'.format(np.max(np.array(eval_accuracy) * 100), np.argmax(np.array(eval_accuracy) * 100)))
    log_string('Minimum Training Loss Reached: {0:.6f}\tEpoch: {1:d}'.format(np.min(np.array(train_loss)), np.argmin(np.array(train_loss))))
    log_string('Minimum Validation Loss Reached: {0:.6f}\tEpoch: {1:d}'.format(np.min(np.array(eval_loss)), np.argmin(np.array(eval_loss))))

    log_string('Time for training: %f'% (time.time() - beginning))
    

    # Plotting (saving in the directory) some figures

    figures(train_loss,eval_loss,train_accuracy,eval_accuracy,lr_story,bn_story)

    LOG_FOUT.close()

    sys.modules[__name__].__dict__.clear()
