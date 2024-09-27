from flax import linen as nn
import jax.numpy as jnp


class InputTransformNet(nn.Module):

    @nn.compact
    def __call__(self, point_cloud, training: bool, bn_decay: float, K = 3):
        """ Input (XYZ) Transform Net, input is BxNx3 gray image
            Return:
                Transformation matrix of size 3xK """
        
        batch_size = point_cloud.shape[0]
        num_point = point_cloud.shape[1]

        input_image = point_cloud


        net = nn.Conv(features=64, kernel_size=1, padding='VALID', strides=1, use_bias = False)(input_image)
        net = nn.BatchNorm(use_running_average= not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)


        net = nn.Conv(features=128, kernel_size=1, padding='VALID', strides=1, use_bias = False)(net)
        net = nn.BatchNorm(use_running_average= not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)


        net = nn.Conv(features=1024, kernel_size=1, padding='VALID', strides=1, use_bias = False)(net)
        net = nn.BatchNorm(use_running_average= not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        net = jnp.max(net, axis = 1, keepdims=True)
        net = jnp.reshape(net, (batch_size, -1))


        net = nn.Dense(features=512)(net)
        net = nn.BatchNorm(use_running_average= not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)


        net = nn.Dense(features=256)(net)
        net = nn.BatchNorm(use_running_average= not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        net = nn.Dense(features=9)(net)

        iden = jnp.tile(jnp.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=jnp.float32), (batch_size, 1))
        
        net = net + iden

        net = jnp.reshape(net, (batch_size,3,3))

        return net


class FeatureTransformNet(nn.Module):

    @nn.compact
    def __call__(self, inputs, training: bool, bn_decay: float, K = 64):
        """ Feature Transform Net, input is BxNx1xK
            Return:
                Transformation matrix of size KxK """
        

        batch_size = inputs.shape[0]
        num_point = inputs.shape[1]


        net = nn.Conv(features=64, kernel_size=1, padding='VALID', strides=1, use_bias=False)(inputs)
        net = nn.BatchNorm(use_running_average= not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)


        net = nn.Conv(features=128, kernel_size=1, padding='VALID', strides=1, use_bias=False)(net)
        net = nn.BatchNorm(use_running_average= not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)


        net = nn.Conv(features=1024, kernel_size=1, padding='VALID', strides=1, use_bias=False)(net)
        net = nn.BatchNorm(use_running_average= not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        net = jnp.max(net, axis = 1, keepdims=True)
        net = jnp.reshape(net, (batch_size, -1))


        net = nn.Dense(features=512)(net)
        net = nn.BatchNorm(use_running_average= not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)


        net = nn.Dense(features=256)(net)
        net = nn.BatchNorm(use_running_average= not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        net = nn.Dense(features=K*K)(net)

        iden = jnp.tile(jnp.eye(K).flatten(), (batch_size, 1))
        
        net = net + iden

        net = jnp.reshape(net, (batch_size,K,K))

        return net


class PointNet(nn.Module):

    @nn.compact
    def __call__(self, inputs, training: bool, bn_decay: float):

        batch_size = inputs.shape[0]
        num_point = inputs.shape[1]
        end_points = {}

        transform = InputTransformNet()(inputs,training, bn_decay)
    
        point_cloud_transform = jnp.matmul(inputs, transform)

        net = nn.Conv(features=64, kernel_size=1,padding='VALID',strides=1, use_bias=False)(point_cloud_transform)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        transform = FeatureTransformNet()(net, training, bn_decay)

        end_points['transform'] = transform

        net_transformed = jnp.matmul(net, transform)

        net = nn.Conv(features=128, kernel_size=1,padding='VALID',strides=1, use_bias=False)(net_transformed)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        net = nn.Conv(features=1024, kernel_size=1,padding='VALID',strides=1, use_bias=False)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)

        net = jnp.max(net, axis = 1, keepdims=True)
        net = jnp.reshape(net, (batch_size,-1))


        net = nn.Dense(features=512)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        net = nn.Dense(features=256)(net)
        net = nn.Dropout(rate= 0.4, deterministic=not training)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        net = nn.Dense(features=40)(net)

        
        return net, end_points


class PointNetBasic(nn.Module):
    
    @nn.compact
    def __call__(self, inputs, training: bool, bn_decay: float):

        batch_size = inputs.shape[0]
        num_point = inputs.shape[1]
        end_points = {}

        input_image = inputs

        net = nn.Conv(features=64, kernel_size=1,padding='VALID',strides=1)(input_image)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net)
        net = nn.relu(net)


        net = nn.Conv(features=64, kernel_size=1,padding='VALID',strides=1)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net)
        net = nn.relu(net)


        net = nn.Conv(features=64, kernel_size=1,padding='VALID',strides=1)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net)
        net = nn.relu(net)


        net = nn.Conv(features=128, kernel_size=1,padding='VALID',strides=1)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net)
        net = nn.relu(net)


        net = nn.Conv(features=1024, kernel_size=1,padding='VALID',strides=1)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net)
        net = nn.relu(net)


        net = nn.max_pool(inputs = net, window_shape = (num_point,), padding = 'VALID') #, strides = (2,2))
        net = jnp.reshape(net, (batch_size,-1))


        net = nn.Dense(features=512)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net)
        net = nn.relu(net)


        net = nn.Dense(features=256)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net)
        net = nn.relu(net)


        net = nn.Dropout(rate= 0.3, deterministic=not training)(net)


        net = nn.Dense(features=40)(net)

        return  net, end_points


class NNSDF(nn.Module):

    @nn.compact
    def __call__(self, shape_code, x_hat, training: bool, bn_decay: float, p = 0.4):
        
        batch_size = x_hat.shape[0]
        num_points = x_hat.shape[1]
        # x_hat --> (BATCH, #POINTS, 3)

        # shape_code --> (BATCH, FEATURES)
        shape_code_extended = jnp.expand_dims(a = shape_code, axis = 1) 
        # shape_code_extended --> (BATCH, 1, FEATURES)

        # print('shape code extended: ',shape_code_extended.shape)

        shape_code_extended = jnp.repeat(shape_code_extended, num_points, axis = 1)
        # shape_code_extended --> (BATCH, #POINTS, FEATURES)

        # print('shape code extended: ',shape_code_extended.shape)

        embedded_input = jnp.concatenate([x_hat,shape_code_extended], axis = 2) # First points then shape_code
        # embedded_input --> (BATCH, #POINTS, FEATURES + 3)
        # FEATURES + 3 :6/7/8/9/10/11/12 (default: 6)

        # print('embedded input shape: ',embedded_input.shape)   

        # TOLTE 'epsilon = 1e-3'
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(embedded_input) 
        net = nn.Dense(features = 512)(net)
        net = nn.relu(net)
        net = nn.Dropout(rate = p, deterministic=not training)(net)
        # print(net.shape)

        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net) 
        net = nn.Dense(features = 512)(net)
        net = nn.relu(net)
        net = nn.Dropout(rate = p, deterministic=not training)(net)
        # print(net.shape)

        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net) 
        net = nn.Dense(features = 512)(net)
        net = nn.relu(net)
        net = nn.Dropout(rate = p, deterministic=not training)(net)
        # print(net.shape)

        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net) 
        net = nn.Dense(features = 512)(net)
        net = nn.relu(net)
        net = nn.Dropout(rate = p, deterministic=not training)(net)

        sdf_hat = nn.Dense(features = 1)(net)

        return sdf_hat
    

class NNShapeCode(nn.Module):

    @nn.compact
    def __call__(self, global_features, training: bool, bn_decay: float, shape_features = 3, p = 0.4):
        
        batch_size = global_features.shape[0]
        num_point = global_features.shape[1] # It's different for every input! Becareful!

        # TOLTE 'epsilon = 1e-3'
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(global_features) 
        net = nn.Dense(features = 512)(net)
        net = nn.relu(net)
        net = nn.Dropout(rate = p, deterministic=not training)(net)
        # print(net.shape)

        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net) 
        net = nn.Dense(features = 512)(net)
        net = nn.relu(net)
        net = nn.Dropout(rate = p, deterministic=not training)(net)
        # print(net.shape)

        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net) 
        net = nn.Dense(features = 512)(net)
        net = nn.relu(net)
        net = nn.Dropout(rate = p, deterministic=not training)(net)
        # print(net.shape)

        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net) 
        net = nn.Dense(features = shape_features)(net) # features = 512 PRINCETON
        net = nn.relu(net)
        shape_code = nn.Dropout(rate = p, deterministic=not training)(net)

        # shape_code = nn.Dense(features = shape_features)(net)

        return shape_code


class PointNetBasicBackbone(nn.Module):

    @nn.compact
    def __call__(self, inputs, training: bool, bn_decay: float):

        batch_size = inputs.shape[0]
        num_point = inputs.shape[1]
        end_points = {}

        input_image = inputs

        net = nn.Conv(features=64, kernel_size=1,padding='VALID',strides=1)(input_image)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net)
        net = nn.relu(net)
        # print(net.shape)

        net = nn.Conv(features=64, kernel_size=1,padding='VALID',strides=1)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net)
        net = nn.relu(net)
        # print(net.shape)

        net = nn.Conv(features=64, kernel_size=1,padding='VALID',strides=1)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net)
        net = nn.relu(net)
        # print(net.shape)

        net = nn.Conv(features=128, kernel_size=1,padding='VALID',strides=1)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net)
        net = nn.relu(net)
        # print(net.shape)

        net = nn.Conv(features=1024, kernel_size=1,padding='VALID',strides=1)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum = bn_decay)(net)
        net = nn.relu(net)
        # print(net.shape)

        net = nn.max_pool(inputs = net, window_shape = (num_point,), padding = 'VALID') #, strides = (2,2))
        global_feature = jnp.reshape(net, (batch_size,-1))

        return  global_feature, end_points
    

class PredictorBasic(nn.Module):

    @nn.compact
    def __call__(self, pointnet_input, x_hat, training: bool, bn_decay: float, shape_features = 3):
        
        # print('Pointnet Input Shape: ', pointnet_input.shape)
        global_features, end_points = PointNetBasicBackbone()(pointnet_input,training,bn_decay)
        # print(global_features.shape)

        shape_code = NNShapeCode()(global_features,training,bn_decay, shape_features)
        # print('shape code: ', shape_code.shape)

        # print('X_hat Input Shape: ', x_hat.shape)
        sdf_hat = NNSDF()(shape_code,x_hat,training,bn_decay)
        # print(sdf_hat.shape)

        return sdf_hat, end_points