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

import time
#from multiprocessing import Pool
#from sampling import voxelize
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='mnist_cls', help='Model name [default: hirerical_planenet_cls]')
parser.add_argument('--log_dir', default='log_mnist', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=512, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=400000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true',default=False, help='Whether to use normal information')
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
    num_channle=5
else:
    num_channle=2
NUM_CLASSES = 10

MODEL = importlib.import_module(FLAGS.model)
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')

LOG_DIR = FLAGS.log_dir
best_model_dir = 'log_mnist/best'
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if not os.path.exists(best_model_dir): os.mkdir(best_model_dir)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

ROOT2_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT2_DIR, 'data/mnist2')

def loadmnist(root):
    data=np.load(root + '/pc.npy')
    label = np.load(root + '/label.npy')
    return data,label

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

DATA_TRAIN = os.path.join(DATA_PATH, 'train')
DATA_TEST = os.path.join(DATA_PATH, 'test')

data_train,labels_train=loadmnist(DATA_TRAIN)
data_test,labels_test=loadmnist(DATA_TEST)

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
            #print(shape)
            #print(len(shape))
            variable_parameters = 1
            for dim in shape:
                #print(dim)
                variable_parameters *= dim.value
            #print(variable_parameters)
            total_parameters += variable_parameters
        log_string("--- Numbers of parameters :  %d " % total_parameters)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': forloss}


        start1 = time.time()

        best_rotate_acc = -1
        best_ov_acc = -1
        best_epoch=0
        best_rotate_epoch=-1
        for epoch in range(MAX_EPOCH):

            start = time.time()
            log_string('\n**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_epoch(sess, ops, train_writer,data_train,labels_train,epoch)
            avg_acc, ov_acc = test_epoch(sess, ops, test_writer,data_test,labels_test)
            if (avg_acc > best_ov_acc):
                best_ov_acc = avg_acc
                best_epoch = epoch
                saver.save(sess, os.path.join(best_model_dir, "model.ckpt"))

            # if avg_acc>0.903:
            #     log_string('eval with rotation:')
            #     rotate_acc,ov_acc = eval_epoch(sess, ops, test_writer,num_votes=12)
            #     if rotate_acc > best_rotate_acc:
            #         saver.save(sess, os.path.join(best_model_dir, "model.ckpt"))
            #         best_rotate_acc = rotate_acc
            #         best_rotate_epoch = epoch


            if epoch % 50 == 0:
                saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))

            end = time.time()

            log_string('time: %f -best accracy : %f  in epoch: %d -best rotate accracy : %f  in epoch: %d\n' % (end - start,best_ov_acc,best_epoch,best_rotate_acc,best_rotate_epoch))

        end1 = time.time()
        log_string('300 epochs finished in : %f\n' % (end1 - start1))
        log_string('best accuracy : %f' % (best_ov_acc))
        log_string('best avg accuracy : %f' % (best_rotate_acc))



def shuffle_data(data, labels):
    idx = np.arange(len(labels))
    idx2 = np.arange(NUM_POINT)

    np.random.shuffle(idx)
    np.random.shuffle(idx2)
    return data[idx][:,idx2, ...], labels[idx], idx




def train_epoch(sess, ops, train_writer,data_train,labels_train, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    current_data, current_label,_ = shuffle_data(data_train, np.squeeze(labels_train))
    #current_label = np.squeeze(current_label)

    file_size = current_data.shape[0]
    num_batches = int(np.ceil(file_size / BATCH_SIZE))

    total_correct = 0
    total_seen = 0
    loss_sum = 0


    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        if batch_idx == num_batches - 1:
            end_idx = len(current_label)
        batch_data = current_data[start_idx:end_idx, :, :]
        batch_labels = current_label[start_idx:end_idx]
        bsize = len(batch_labels)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_labels,
                     ops['is_training_pl']: is_training, }

        summary, step,_,  loss_val, pred_val = sess.run([ops['merged'], ops['step'],ops['train_op'],
                                                         ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_labels[0:bsize])

        total_correct += correct
        total_seen += bsize
        loss_sum += (loss_val * float(bsize / BATCH_SIZE))
        batch_idx += 1
        if epoch == 0:
            if (batch_idx + 1) % 100 == 0:
                log_string(' -- %03d -- ' % (batch_idx + 1))
                log_string('train loss: %f' % (loss_sum / 100))
                log_string('train accuracy: %f' % (total_correct / float(total_seen)))
                total_correct = 0
                total_seen = 0
                loss_sum = 0

    if epoch != 0:
        log_string('train loss : %f  - train Accuracy: %f' % (loss_sum / batch_idx, total_correct / float(total_seen)))




def test_epoch(sess, ops, test_writer,data_test,labels_test):
    """ ops: dict mapping from string to tf ops """

    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]


    # Shuffle train samples
    current_data, current_label = data_test, np.squeeze(labels_test)
    file_size = current_data.shape[0]
    num_batches = int(np.ceil(file_size / BATCH_SIZE))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        if batch_idx == num_batches - 1:
            end_idx = len(current_label)
        batch_data = current_data[start_idx:end_idx, :, :]
        batch_labels = current_label[start_idx:end_idx]
        bsize = len(batch_labels)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_labels,
                     ops['is_training_pl']: is_training}

        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],ops['loss'],
                                                      ops['pred']], feed_dict=feed_dict)


        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_labels[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += (loss_val * float(bsize / BATCH_SIZE))
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_labels[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    ov_acc = total_correct / float(total_seen)
    avg_acc = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
    log_string('eval loss: %f   - eval accuracy: %f  - eval avg-class acc: %f' % (
        loss_sum / float(batch_idx),
        ov_acc,
        avg_acc))
    return ov_acc,avg_acc




def eval_epoch(sess, ops, test_writer,num_votes=12):
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

            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],ops['loss'],
                                                          ops['pred']], feed_dict=feed_dict)


            test_writer.add_summary(summary, step)
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

    ov_acc = total_correct / float(total_seen)
    avg_acc = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
    log_string('eval loss: %f   - eval accuracy: %f  - eval avg-class acc: %f' % (
        loss_sum / float(batch_idx),
        ov_acc,
        avg_acc))

    TEST_DATASET.reset()
    return ov_acc, avg_acc


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    #p = Pool()
    train()
    LOG_FOUT.close()