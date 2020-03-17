import os
import sys
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
import util


def angletodcm(transform,mod='XYZ'):
    rotation_matrix = tf.stack([tf.multiply(tf.cos(transform[:,1]),tf.cos(transform[:,2]))
                                   ,-tf.multiply(tf.cos(transform[:,1]),tf.sin(transform[:,2]))
                                   ,tf.sin(transform[:,1])
                                   ,tf.add(tf.multiply(tf.multiply(tf.sin(transform[:,0]),tf.sin(transform[:,1])),tf.cos(transform[:,2])),tf.multiply(tf.cos(transform[:,0]),tf.sin(transform[:,2])))
                                   ,tf.add(-tf.multiply(tf.multiply(tf.sin(transform[:,0]),tf.sin(transform[:,1])),tf.sin(transform[:,2])),tf.multiply(tf.cos(transform[:,0]),tf.cos(transform[:,2])))
                                   ,-tf.multiply(tf.sin(transform[:,0]),tf.cos(transform[:,1]))
                                   ,tf.add(-tf.multiply(tf.multiply(tf.cos(transform[:,0]),tf.sin(transform[:,1])),tf.cos(transform[:,2])),tf.multiply(tf.sin(transform[:,0]),tf.sin(transform[:,2])))
                                   ,tf.add(tf.multiply(tf.multiply(tf.cos(transform[:,0]),tf.sin(transform[:,1])),tf.sin(transform[:,2])),tf.multiply(tf.sin(transform[:,0]),tf.cos(transform[:,2])))
                                   ,tf.multiply(tf.cos(transform[:,0]),tf.cos(transform[:,1]))] ,axis=1)

    rotation_matrix=tf.reshape(rotation_matrix,[-1,3,3])
    return rotation_matrix



def activation_function(input):
    with tf.variable_scope('activation_f') as sc:
        down = tf.add(1.0,tf.square(input))
        output = tf.divide(1.0, down)
    return output

def activation_function2(input):
    with tf.variable_scope('activation_f') as sc:
        output = tf.divide(1, (1+tf.exp(input)))
    return output

def activation_function3(input):
    with tf.variable_scope('activation_f') as sc:
        output = tf.exp(-input)

    return output

def group_point(xyz,idx):
    with tf.variable_scope('grouping_moudule') as sc:
        batch_size = tf.shape(xyz)[0]
        num_point = idx.get_shape()[1].value
        numdim = idx.get_shape()[2].value

        count = tf.range(batch_size)
        count=tf.expand_dims(count, 1)
        count = tf.expand_dims(count, 2)
        count = tf.tile(count, [1,num_point,numdim])
        count = tf.cast(count, tf.int32)
        idxs=tf.stack([count, idx], axis=-1)

        result=tf.gather_nd(xyz, idxs)
    return result

def cen(xyz,knn_xyz,nsample):
    with tf.variable_scope('Centeralize') as sc:
        knn_xyz = tf.subtract(knn_xyz,tf.expand_dims(xyz, 2))
    return knn_xyz

def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    with tf.variable_scope('knn_moudule') as sc:
        b = tf.shape(xyz1)[0]
        n,c = xyz1.get_shape()[1].value,xyz1.get_shape()[2].value
        m = n
        print (b, n, c, m)
        print (xyz1, (b,1,n,c))
        xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
        xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
        dist = tf.reduce_sum(tf.square(xyz1-xyz2), -1)
        dist=-dist
        _, idx = tf.nn.top_k(dist, k=k)

    return idx


def knn_point2(k, xyz,xyz1):
    with tf.variable_scope('knn_moudule') as sc:
        transpose_xyz = tf.transpose(xyz,[0,2,1])
        Gram_matrix = tf.matmul(xyz,transpose_xyz)
        eye = tf.eye(1024,1024)
        g = tf.reduce_sum(tf.multiply(eye,Gram_matrix),-1,keep_dims=True)
        transpose_g = tf.transpose(g,[0,2,1])
        one= tf.ones_like(g)
        transpose_one=tf.transpose(one,[0,2,1])
        distance= tf.matmul(g,transpose_one) + tf.matmul(one,transpose_g) + (-2 * Gram_matrix)
        neg_distance = -distance
        _, idx = tf.nn.top_k(neg_distance, k=k)
    return idx

def knn_point3(k, xyz,xyz1):
    with tf.variable_scope('knn_moudule') as sc:
        r= tf.reduce_sum(xyz * xyz,axis=2,keep_dims=True)
        m= tf.matmul(xyz,tf.transpose(xyz,[0,2,1]))
        distance= r - (2 * m) + tf.transpose(r,[0,2,1])
        neg_distance = -distance
        _, idx = tf.nn.top_k(neg_distance, k=k)
    return idx



def conv_plane(inputs,
               num_output_channels,
               kernel_size,
               scope,
               pool,
               use_xavier=True,
               stddev=1e-3,
               weight_decay=0.0,
               activation_fn=tf.nn.sigmoid,
               bn=False,
               bn_decay=None,
               is_training=None):
    with tf.variable_scope(scope) as sc:

        kernel_h, kernel_w = kernel_size
        kernel_shape = [kernel_h, kernel_w,
                        num_output_channels]
        kernel = util._variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)


        num_output_channels = kernel.get_shape()[-1].value

        input_re = tf.expand_dims(inputs, 1)

        kernel_re = tf.transpose(kernel, (0, 2, 1))
        kernel_re = tf.expand_dims(kernel_re, 2)
        kernel_re = tf.expand_dims(kernel_re, 3)

        outputs = tf.reduce_sum(tf.multiply(input_re, kernel_re), -1)
        outputs = tf.transpose(outputs, (0, 2, 3, 1))
        d = util._variable_on_cpu('d', [num_output_channels],
                             tf.constant_initializer(0.0))
        outputs = tf.add(outputs, d)
        outputs = tf.abs(outputs)
        #outputs = tf.divide(outputs, tf.norm(kernel, axis=1, keep_dims=True)) #axis=????????/

        if pool == 'max':
            outputs = tf.reduce_max(outputs, 2,keep_dims=True)
        if pool == 'sum':
            outputs = tf.reduce_sum(outputs, 2,keep_dims=True)
        elif pool=='avg':
            outputs = tf.reduce_sum(outputs, 2,keep_dims=True)
            nsample = inputs.get_shape()[2].value
            outputs = tf.divide(outputs, nsample)
        elif pool == 'minmax':
            max = tf.reduce_max(outputs, 2,keep_dims=True)
            min = tf.reduce_min(outputs, 2,keep_dims=True)
            outputs = tf.subtract(max,min)

        outputs = tf.negative(outputs)


        if bn:
            outputs = util.batch_norm_for_conv2d(outputs, is_training,
                                          bn_decay=bn_decay, scope='bn-plane')


        if activation_fn is not None:
            outputs = tf.nn.sigmoid(outputs)

        return outputs


def conv_plane2d(inputs,
               num_output_channels,
               kernel_size,
               scope,
               pool,
               use_xavier=True,
               stddev=1e-3,
               weight_decay=0.0,
               activation_fn=tf.nn.sigmoid,
               bn=False,
               bn_decay=None,
               is_training=None):
    with tf.variable_scope(scope) as sc:

        kernel_h, kernel_w = kernel_size

        outputs = util.conv2d(inputs, 32, [1, 1],
                                  padding='VALID', stride=[1, 1],
                                  bn=bn, is_training=is_training,
                                  scope='conv2d%d' % (64), bn_decay=bn_decay)

        outputs = tf.reduce_max(outputs, 2,keep_dims=True)

        return outputs

def transform_moudule_seg(xyz,points, is_training,idx, num_out_conv, size_conv, num_out_fc,num_channle,nsample, scope, bn_decay, bn=True,K=3):
    with tf.variable_scope(scope) as sc:
        new_points, net ,idx = plane_module(xyz, points, idx,centralize=True, num_channle=num_channle, npoint=1024,
                                                          have_sample=False, nsample=nsample,
                                                          is_training=is_training, scope='Tnet-plane', pool='avg',
                                                          use_xyz=True, bn=bn, bn_decay=bn_decay,weight_decay=0.0)


        batch_size = tf.shape(xyz)[0]
        for i in range(len(num_out_conv)):
            net = util.conv2d(net, num_out_conv[i], size_conv[i], scope='tconv' + str(i + 1), stride=[1, 1], bn=True,
                         bn_decay=bn_decay, is_training=is_training, padding='VALID')
        net = tf.squeeze(net, 2)
        net = tf.reduce_max(net, axis=1)

        for i in range(len(num_out_fc)):
            net = util.fully_connected(net, num_out_fc[i], scope='tfc' + str(i + 1), bn_decay=bn_decay, bn=True,
                                  is_training=is_training)
        with tf.variable_scope(scope) as sc:
            weights = tf.get_variable('weights', [256, K * K],
                                      initializer=tf.constant_initializer(0.0),
                                      dtype=tf.float32)
            biases = tf.get_variable('biases', [K * K],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
            transform = tf.matmul(net, weights)
            transform = tf.nn.bias_add(transform, biases)

        transform = tf.reshape(transform, [batch_size, K, K])

        xyz = tf.matmul(xyz, transform)
        if new_points is not None:
            new = tf.matmul(new_points[:,:,3:6], transform)
            new_points=tf.concat([new_points[:, :, 0:3],new],-1)
    return xyz,transform,new_points


def conv_plane2(inputs,
               num_output_channels,
               kernel_size,
               scope,
               pool,
               use_xavier=True,
               stddev=1e-3,
               weight_decay=0.0,
               activation_fn=tf.nn.sigmoid,
               bn=False,
               bn_decay=None,
               is_training=None):
    with tf.variable_scope(scope) as sc:

        kernel_h, kernel_w = kernel_size
        kernel_shape = [kernel_h, kernel_w,
                        num_output_channels]


        inputs = tf.expand_dims(inputs, -1)
        outputs = util.conv3d(inputs, 32, [1,1,3],
                                    padding='VALID', stride=[1, 1,1],
                                    bn=bn, is_training=is_training,activation_fn=tf.nn.sigmoid,
                                    scope='conv2d%d' % (32), bn_decay=bn_decay)

        outputs = tf.reduce_sum(outputs, 2)
        nsample = inputs.get_shape()[2].value
        outputs = tf.divide(outputs, nsample)


        return outputs



# def transform_moudule(xyz,is_training,num_out_conv,size_conv,num_out_fc,nsample,scope, bn_decay,  bn=True,K=3):
#     with tf.variable_scope('Transform-net') as sc:
#         _, idx = knn_point(nsample, xyz, xyz)
#         knn_xyz = group_point(xyz, idx)
#         knn_xyz = cen(xyz, knn_xyz, nsample)24
#         tnet=util.plane_tnet(xyz,knn_xyz, is_training, num_out_conv,size_conv, num_out_fc, scope, bn_decay=bn_decay, bn=True,K=K)
#     return tnet,idx




def transform_moudule2(xyz,points, is_training,idx, num_out_conv, size_conv, num_out_fc,num_channle,nsample, scope, bn_decay, bn=True,K=3):
    with tf.variable_scope(scope) as sc:
        new_points, net ,idx = plane_module(xyz, points, idx,centralize=True, num_channle=num_channle, npoint=1024,
                                                          nsample=nsample,
                                                          is_training=is_training, scope='Tnet-plane', pool='avg',
                                                          use_xyz=True, bn=bn, bn_decay=bn_decay,weight_decay=0.0)
        if new_points is not None:
            net = tf.concat([net, new_points], axis=-1)

        batch_size = tf.shape(xyz)[0]
        for i in range(len(num_out_conv)):
            net = util.conv2d(net, num_out_conv[i], size_conv[i], scope='tconv' + str(i + 1), stride=[1, 1], bn=True,
                         bn_decay=bn_decay, is_training=is_training, padding='VALID')
        net = tf.squeeze(net, 2)
        net = tf.reduce_max(net, axis=1)

        for i in range(len(num_out_fc)):
            net = util.fully_connected(net, num_out_fc[i], scope='tfc' + str(i + 1), bn_decay=bn_decay, bn=True,
                                  is_training=is_training)
        with tf.variable_scope(scope) as sc:
            weights = tf.get_variable('weights', [256, K],
                                      initializer=tf.constant_initializer(1.0),
                                      dtype=tf.float32)
            biases = tf.get_variable('biases', [K],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            transform = tf.matmul(net, weights)
            transform = tf.nn.bias_add(transform, biases)
            #transform = tf.sigmoid(transform)
            pi = tf.constant(2 * np.pi, name="2pi")
            transform = tf.multiply(transform, pi)
            transform=angletodcm(transform)

        xyz = tf.matmul(xyz, transform)
        if new_points is not None:
            new_points = tf.matmul(new_points, transform)
    return xyz,transform,new_points



def transform_moudule(xyz,points, is_training,idx, num_out_conv, size_conv, num_out_fc,num_channle,nsample, scope, bn_decay, bn=True,K=3):
    with tf.variable_scope(scope) as sc:
        new_points, net ,idx = plane_module(xyz, points, idx,centralize=True, num_channle=num_channle, npoint=1024,
                                                          nsample=nsample,
                                                          is_training=is_training, scope='Tnet-plane', pool='avg',
                                                          use_xyz=True, bn=bn, bn_decay=bn_decay,weight_decay=0.0)


        batch_size = tf.shape(xyz)[0]
        for i in range(len(num_out_conv)):
            net = util.conv2d(net, num_out_conv[i], size_conv[i], scope='tconv' + str(i + 1), stride=[1, 1], bn=True,
                         bn_decay=bn_decay, is_training=is_training, padding='VALID')
        net = tf.squeeze(net, 2)
        net = tf.reduce_max(net, axis=1)

        for i in range(len(num_out_fc)):
            net = util.fully_connected(net, num_out_fc[i], scope='tfc' + str(i + 1), bn_decay=bn_decay, bn=True,
                                  is_training=is_training)
        with tf.variable_scope(scope) as sc:
            weights = tf.get_variable('weights', [256, K * K],
                                      initializer=tf.constant_initializer(0.0),
                                      dtype=tf.float32)
            biases = tf.get_variable('biases', [K * K],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
            transform = tf.matmul(net, weights)
            transform = tf.nn.bias_add(transform, biases)

        transform = tf.reshape(transform, [batch_size, K, K])

        xyz = tf.matmul(xyz, transform)
        if new_points is not None:
            new_points = tf.matmul(new_points, transform)
    return xyz,transform,new_points



def plane_module(xyz, points,idx,centralize,num_channle, npoint, nsample, is_training,scope, bn_decay,pool='avg',use_xyz=False,  bn=True,weight_decay=0,tnet=None):
    with tf.variable_scope(scope) as sc:

        if idx is not None:
            knn_xyz = group_point(xyz, idx)
            if centralize:
                knn_xyz = cen(xyz, knn_xyz, nsample)
            new_points = points
        else:
            new_xyz = xyz  # (batch_size, npoint, 3)
            new_points = points

            idx = knn_point3(nsample, xyz, new_xyz)
            knn_xyz2 = group_point(xyz, idx)
            if centralize:
                knn_xyz = cen(xyz, knn_xyz2, nsample)
            else:
                knn_xyz = knn_xyz2


        num_dim=knn_xyz.get_shape()[-1].value

        feature_plane = conv_plane(knn_xyz,num_channle,[1,num_dim],scope='planes0',pool=pool,bn=bn,is_training=is_training,bn_decay=bn_decay,weight_decay=weight_decay)


        if use_xyz:
            if new_points is not None:
                xyz = tf.concat([xyz, new_points], axis=-1)

            feature = tf.concat([tf.expand_dims(xyz,2),feature_plane],-1)
        else:
            if new_points is not None:
                feature = tf.concat([new_points,feature_plane], axis=-1)

    return new_points, feature,idx



def conv_module(points,new_points, mlp, is_training, scope, bn_decay, bn=True):
    with tf.variable_scope(scope) as sc:

        for i, num_out_channel in enumerate(mlp):
            new_points = util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv2d%d'%(i), bn_decay=bn_decay)



        new_points = tf.squeeze(new_points, [2])
        new_points_all = tf.reduce_max(new_points, axis=1)

    return new_points_all,new_points


def conv_module_seg(points,feature_plane, mlp, is_training, scope, bn_decay, bn=True):
    with tf.variable_scope(scope) as sc:

        new_points=feature_plane

        new_points64 = util.conv2d(new_points, mlp[0], [1, 1],
                                  padding='VALID', stride=[1, 1],
                                  bn=bn, is_training=is_training,
                                  scope='conv2d%d' % (64), bn_decay=bn_decay)

        new_points1281 = util.conv2d(new_points64, mlp[1], [1, 1],
                                  padding='VALID', stride=[1, 1],
                                  bn=bn, is_training=is_training,
                                  scope='conv2d%d' % (1281), bn_decay=bn_decay)

        new_points1282 = util.conv2d(new_points1281, mlp[2], [1, 1],
                                  padding='VALID', stride=[1, 1],
                                  bn=bn, is_training=is_training,
                                  scope='conv2d%d' % (1282), bn_decay=bn_decay)


        new_points1024 = util.conv2d(new_points1282, 1024, [1, 1],
                                  padding='VALID', stride=[1, 1],
                                  bn=bn, is_training=is_training,
                                  scope='conv2d%d' % (1024), bn_decay=bn_decay)

        #new_points1024 = tf.squeeze(new_points1024, [2])
        new_points_all = tf.reduce_max(new_points1024, axis=1)
            #new_points_all = tf.squeeze(new_points_all, [1])

    return new_points_all,new_points64,new_points1281,new_points1282