import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
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
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true',default=False, help='Whether to use normal information')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--model', default='modelnet_cls', help='Model name [default: hirerical_planenet_cls]')
parser.add_argument('--transform', type=bool, default=True, help='Whether to use normal information')

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

#ROOT2_DIR = os.path.dirname(ROOT_DIR)
ROOT2_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT2_DIR, 'data/modelnet40_normal_resampled')
TRAIN_DATASET = modelnet40.ModelNetDataset(root=DATA_PATH,shuffle=True, npoints=NUM_POINT,batch_size=BATCH_SIZE, split='train', normal_channel=FLAGS.normal)
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
    learning_rate = tf.maximum(learning_rate, 0.00005) # CLIP THE LEARNING RATE!
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

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            pointclouds, labels,is_training = MODEL.placeholder_inputs(NUM_POINT,num_channle)
            pred, forloss = MODEL.get_model(pointclouds, is_training,TRANSFORM, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels, forloss)
            tf.summary.scalar('Total loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('Accuracy', accuracy)

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
        merged = tf.summary.merge_all()


        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        log_string(str(datetime.now()))
        log_string("--- Numbers of parameters :  %d " % total_parameters)

        ops = {'pointclouds': pointclouds,
               'labels': labels,
               'is_training': is_training,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'forloss': forloss}


        start1 = time.time()

        best_rotate_acc = -1
        best_ov_acc = -1
        best_epoch=0
        best_rotate_epoch=-1
        for epoch in range(MAX_EPOCH):

            start = time.time()
            log_string('\n**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_epoch(sess, ops, train_writer,epoch)

            # evaluation on initial test data
            avg_acc, ov_acc = eval_epoch(sess, ops, test_writer,num_votes=1)
            if (avg_acc > best_ov_acc):
                best_ov_acc = avg_acc
                best_epoch = epoch
                saver.save(sess, os.path.join(best_model_dir, "model.ckpt"))

            # evaluation on 12 rotate of test data
            # if avg_acc>0.91:
            #     log_string('eval with rotation:')
            #     rotate_acc,ov_acc = eval_epoch(sess, ops, test_writer,num_votes=12)
            #     if rotate_acc > best_rotate_acc:
            #         saver.save(sess, os.path.join(best_model_dir, "model.ckpt"))
            #         best_rotate_acc = rotate_acc
            #         best_rotate_epoch = epoch
            #

            if epoch % 10 == 0:
                saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))

            end = time.time()

            log_string('time: %f -best accracy : %f  in epoch: %d -best rotate accracy : %f  in epoch: %d\n' % (end - start,best_ov_acc,best_epoch,best_rotate_acc,best_rotate_epoch))

        end1 = time.time()
        log_string('300 epochs finished in : %f\n' % (end1 - start1))
        log_string('best accuracy : %f in epoch: %d ' % (best_ov_acc,best_epoch))
        log_string('best avg accuracy : %f in epoch: %d' % (best_rotate_acc,best_rotate_epoch))



def train_epoch(sess, ops, train_writer, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0

    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)
        bsize = batch_data.shape[0]

        feed_dict = {ops['pointclouds']: batch_data,
                     ops['labels']: batch_label,
                     ops['is_training']: is_training}

        summary, step,_,  loss_val, pred_val = sess.run([ops['merged'], ops['step'],ops['train_op'],
                                                         ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])

        total_correct += correct
        total_seen += bsize
        loss_sum += (loss_val*float(bsize/BATCH_SIZE))
        batch_idx += 1
        if epoch == 0:
            if (batch_idx + 1) % 50 == 0:
                log_string(' -- %03d -- ' % (batch_idx + 1))
                log_string('train loss: %f' % (loss_sum / 50))
                log_string('train accuracy: %f' % (total_correct / float(total_seen)))
                total_correct = 0
                total_seen = 0
                loss_sum = 0

    if epoch != 0:
        log_string('train loss : %f  - train Accuracy: %f' % (loss_sum / batch_idx, total_correct / float(total_seen)))

    TRAIN_DATASET.reset()


def eval_epoch(sess, ops, test_writer,num_votes=12):
    """ ops: dict mapping from string to tf ops """

    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]


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


            feed_dict = {ops['pointclouds']: rotated_data,
                         ops['labels']: batch_label,
                         ops['is_training']: is_training}

            summary, loss_val, pred_val = sess.run([ops['merged'],ops['loss'],
                                                          ops['pred']], feed_dict=feed_dict)


            batch_pred_sum += pred_val

        pred_val = np.argmax(batch_pred_sum, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += (loss_val * float(bsize / BATCH_SIZE))
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

    TEST_DATASET.reset()
    return ov_acc,avg_acc


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()