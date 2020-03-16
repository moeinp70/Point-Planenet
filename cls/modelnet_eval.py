import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import modelnet40
import preprocess

import time

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='planenet_cls', help='Model name [default: hirerical_planenet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=300, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true',default=False, help='Whether to use normal information')
parser.add_argument('--transform', type=bool, default=True, help='Whether to use normal information')
best_model = 'log/best/model.ckpt'
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NORMAL=FLAGS.normal
TRANSFORM=FLAGS.transform

if NORMAL:
    num_channle=6
else:
    num_channle=3
NUM_CLASSES = 40

MODEL = importlib.import_module(FLAGS.model)
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')

LOG_DIR = FLAGS.log_dir
best_model_dir = 'log/best'
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if not os.path.exists(best_model_dir): os.mkdir(best_model_dir)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

ROOT2_DIR = os.path.dirname(ROOT_DIR)
DATA_PATH = os.path.join(ROOT2_DIR, 'data/modelnet40_normal_resampled')
TRAIN_DATASET = modelnet40.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT,batch_size=BATCH_SIZE, split='train', normal_channel=FLAGS.normal)
TEST_DATASET = modelnet40.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT,batch_size=BATCH_SIZE, split='test', normal_channel=FLAGS.normal)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.000005) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl,is_training_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,num_channle)

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            pred, forloss = MODEL.get_model(pointclouds_pl, is_training_pl,TRANSFORM, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, forloss)
            tf.summary.scalar('total loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            print( "--- Get Training operator ---")

            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)

            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            saver = tf.train.Saver()

            # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        saver.restore(sess,best_model)
        log_string("model restored.")
        # Init variables



        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'end_points': forloss}




        eval_test(sess, ops,12)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def vis(xyz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb=['r','b','g','c','y','k','m','orange','gold','brown','silver','pink','gray','Indigo','Wheat','BurlyWood']
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=rgb[0], marker='o') #np.random.randint(0,15)
    plt.show()

def eval_test(sess, ops,num_votes=12):
    """ ops: dict mapping from string to tf ops """

    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    #log_string(str(datetime.now()))
    #log_string('---- EPOCH EVALUATION ----' )

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]

        batch_pred_sum = np.zeros((bsize, NUM_CLASSES)) # score for classes
        for vote_idx in range(num_votes):
            # Shuffle point order to achieve different farthest samplings
            shuffled_indices = np.arange(NUM_POINT)
            np.random.shuffle(shuffled_indices)
            if FLAGS.normal:
                rotated_data = preprocess.rotate_point_cloud_by_angle_with_normal(batch_data[:, shuffled_indices, :],
                    vote_idx/float(num_votes) * np.pi * 2)
            else:
                rotated_data = preprocess.rotate_point_cloud_by_angle(batch_data[:, shuffled_indices, :],
                    vote_idx/float(num_votes) * np.pi * 2)


            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: batch_label,
                         ops['is_training_pl']: is_training}

            loss_val, pred_val,tnet = sess.run([ops['loss'],ops['pred'],ops['end_points']], feed_dict=feed_dict)
            b=rotated_data[0]
            c=np.matmul(b , tnet[0])
            vis(b)
            vis(c)
            batch_pred_sum += pred_val

        pred_val = np.argmax(batch_pred_sum, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += (loss_val * float(bsize/BATCH_SIZE))
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    ov_acc=total_correct / float(total_seen)
    avg_acc=np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
    log_string('eval loss: %f   - eval accuracy: %f  - eval avg-class acc: %f' % (
        loss_sum / float(batch_idx),
        ov_acc,
        avg_acc))
    #log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    #log_string('eval avg class acc: %f' % (
    #    np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    TEST_DATASET.reset()
    return


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    #p = Pool()
    train()
    LOG_FOUT.close()