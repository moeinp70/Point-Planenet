import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import util
from planenet_util import plane_module,conv_module,transform_moudule

def placeholder_inputs(batch_size, num_point,num_channle):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(None, num_point, num_channle))
    labels_pl = tf.placeholder(tf.int32, shape=(None))
    is_training = tf.placeholder(tf.bool, shape=())

    return pointclouds_pl, labels_pl,is_training

def get_model(point_cloud, is_training,Transform=False, bn_decay=None):
    forloss = []
    idx=None
    if point_cloud.get_shape()[-1].value>2:
        l0_xyz = point_cloud[:,:,0:2]
        l0_points = point_cloud [:,:,2:]
    else:
        l0_xyz = point_cloud
        l0_points = None


    if Transform:
        l0_xyz, tnet,l0_points= transform_moudule(l0_xyz,l0_points, is_training,idx, [64,128,1024],[[1, 1], [1, 1],[1, 1]], [512,256],num_channle=32, nsample=32, scope='Tnet1', bn_decay=bn_decay, bn=True, K=2)
        #forloss=tnet
    l0_points, plane_feature, _ = plane_module(l0_xyz, l0_points, idx, centralize=True, num_channle=32, npoint=512,
                                               have_sample=False,
                                               nsample=32, is_training=is_training, scope='plane', pool='avg',
                                               use_xyz=True, bn=True, bn_decay=bn_decay, weight_decay=0.0)

    new_points,_ = conv_module(l0_points,plane_feature,mlp=[64,128,128,1024], is_training=is_training, scope='conv', bn=True, bn_decay=bn_decay)
    # Fully connected layers
    net = util.fully_connected(new_points, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = util.fully_connected(net, 10, activation_fn=None, scope='fc3')


    return net, forloss


def get_loss(pred, label, forloss):

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', loss)
    if forloss!=[]:
        K = forloss.get_shape()[2].value
        transform_diff = tf.matmul(forloss, tf.transpose(forloss, perm=[0, 2, 1]))
        transform_diff -= tf.constant(np.eye(K), dtype=tf.float32)
        transform_loss = tf.nn.l2_loss(transform_diff)
        tf.summary.scalar('transform loss', transform_loss)
        loss = (transform_loss * 0.01) + loss
    return loss