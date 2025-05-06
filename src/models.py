from flax import linen as nn
import jax.numpy as jnp


class InputTransformNet(nn.Module):
    """
    A neural network module that performs input transformation on 3D point clouds.

    This network takes a point cloud as input and outputs a transformation matrix 
    (3x3) that can be applied to the input point cloud. The network consists of 
    several convolutional layers followed by dense layers to learn a transformation 
    that can be used for tasks such as point cloud alignment.

    Args:
        nn.Module: The base class for Flax neural network modules.
    
    Returns:
        The output transformation matrix (3x3) for the input point cloud.
    """

    @nn.compact
    def __call__(self, point_cloud, training: bool, bn_decay: float, K=3):
        """
        Forward pass for the InputTransformNet, which computes a transformation matrix 
        for the input point cloud.

        Args:
            point_cloud (jax.Array): The input point cloud of shape (batch_size, num_points, 3).
            training (bool): A flag indicating whether the model is in training mode.
            bn_decay (float): The decay factor for batch normalization.
            K (int, optional): A hyperparameter for the number of layers. Defaults to 3.
        
        Returns:
            jax.Array: The transformation matrix (3x3) for each input point cloud in the batch.
        """
        
        batch_size = point_cloud.shape[0]  # Number of samples in the batch
        num_point = point_cloud.shape[1]  # Number of points in each point cloud

        input_image = point_cloud

        # First convolutional layer
        net = nn.Conv(features=64, kernel_size=1, padding='VALID', strides=1, use_bias=False)(input_image)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        # Second convolutional layer
        net = nn.Conv(features=128, kernel_size=1, padding='VALID', strides=1, use_bias=False)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        # Third convolutional layer
        net = nn.Conv(features=1024, kernel_size=1, padding='VALID', strides=1, use_bias=False)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        # Max pooling operation
        net = jnp.max(net, axis=1, keepdims=True)
        net = jnp.reshape(net, (batch_size, -1))

        # Fully connected layers
        net = nn.Dense(features=512)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        net = nn.Dense(features=256)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        # Output layer: 9-dimensional vector representing a transformation matrix (3x3)
        net = nn.Dense(features=9)(net)

        # Identity matrix for initialization
        iden = jnp.tile(jnp.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=jnp.float32), (batch_size, 1))
        
        # Add the identity matrix to the output of the network to ensure valid transformations
        net = net + iden

        # Reshape the output into a 3x3 transformation matrix for each point cloud in the batch
        net = jnp.reshape(net, (batch_size, 3, 3))

        return net



class FeatureTransformNet(nn.Module):
    """
    A neural network module that computes a feature transformation matrix for point cloud data.

    This network processes the input features and outputs a transformation matrix that can 
    be applied to the feature vectors of the input points. The network consists of several 
    convolutional layers followed by fully connected layers, ultimately outputting a KxK 
    transformation matrix for each input sample.

    Args:
        nn.Module: The base class for Flax neural network modules.
    
    Returns:
        The output feature transformation matrix (KxK) for the input features.
    """

    @nn.compact
    def __call__(self, inputs, training: bool, bn_decay: float, K=64):
        """
        Forward pass for the FeatureTransformNet, which computes a feature transformation matrix 
        for the input features.

        Args:
            inputs (jax.Array): The input features of shape (batch_size, num_points, feature_dim).
            training (bool): A flag indicating whether the model is in training mode.
            bn_decay (float): The decay factor for batch normalization.
            K (int, optional): The size of the transformation matrix. Defaults to 64.
        
        Returns:
            jax.Array: The transformation matrix (KxK) for each input feature in the batch.
        """
        
        batch_size = inputs.shape[0]  # Number of samples in the batch
        num_point = inputs.shape[1]  # Number of points in each point cloud

        # First convolutional layer
        net = nn.Conv(features=64, kernel_size=1, padding='VALID', strides=1, use_bias=False)(inputs)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        # Second convolutional layer
        net = nn.Conv(features=128, kernel_size=1, padding='VALID', strides=1, use_bias=False)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        # Third convolutional layer
        net = nn.Conv(features=1024, kernel_size=1, padding='VALID', strides=1, use_bias=False)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        # Max pooling operation
        net = jnp.max(net, axis=1, keepdims=True)
        net = jnp.reshape(net, (batch_size, -1))

        # Fully connected layers
        net = nn.Dense(features=512)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        net = nn.Dense(features=256)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        # Output layer: KxK transformation matrix
        net = nn.Dense(features=K * K)(net)

        # Identity matrix for initialization
        iden = jnp.tile(jnp.eye(K).flatten(), (batch_size, 1))
        
        # Add the identity matrix to the output of the network to ensure valid transformations
        net = net + iden

        # Reshape the output into a KxK transformation matrix for each point cloud in the batch
        net = jnp.reshape(net, (batch_size, K, K))

        return net

class PointNet(nn.Module):
    """
    PointNet: A neural network architecture for point cloud processing, primarily designed for tasks like 
    classification and segmentation of 3D point clouds.

    The PointNet architecture consists of:
        - A transformation network for input points (`InputTransformNet`).
        - A feature transformation network for learned features (`FeatureTransformNet`).
        - Several convolutional layers followed by fully connected layers to process the features.

    Args:
        nn.Module: The base class for Flax neural network modules.
    
    Returns:
        jax.Array: The output logits for each input in the batch, which can be used for classification or regression.
        dict: A dictionary containing intermediate points such as transformation matrices (`transform`).
    """

    @nn.compact
    def __call__(self, inputs, training: bool, bn_decay: float):
        """
        Forward pass of the PointNet architecture. The input points are transformed using transformation networks, 
        followed by convolutional and fully connected layers to produce a set of logits for each input point cloud.

        Args:
            inputs (jax.Array): The input point cloud data of shape (batch_size, num_points, 3).
            training (bool): A flag indicating whether the model is in training mode.
            bn_decay (float): The decay factor for batch normalization.
        
        Returns:
            jax.Array: The logits for each sample in the batch (shape: batch_size x num_classes).
            dict: A dictionary containing additional points like the transformation matrices applied during the network.
        """

        batch_size = inputs.shape[0]  # Number of samples in the batch
        num_point = inputs.shape[1]   # Number of points per point cloud
        end_points = {}

        # Input transformation network (learned transformation for input point cloud)
        transform = InputTransformNet()(inputs, training, bn_decay)
        point_cloud_transform = jnp.matmul(inputs, transform)  # Apply the input transformation

        # First convolutional layer
        net = nn.Conv(features=64, kernel_size=1, padding='VALID', strides=1, use_bias=False)(point_cloud_transform)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        # Feature transformation network (learned transformation for feature space)
        transform = FeatureTransformNet()(net, training, bn_decay)
        end_points['transform'] = transform  # Store the feature transformation matrix

        # Apply the feature transformation
        net_transformed = jnp.matmul(net, transform)

        # Second convolutional layer
        net = nn.Conv(features=128, kernel_size=1, padding='VALID', strides=1, use_bias=False)(net_transformed)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        # Third convolutional layer
        net = nn.Conv(features=1024, kernel_size=1, padding='VALID', strides=1, use_bias=False)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)

        # Max pooling to aggregate the features
        net = jnp.max(net, axis=1, keepdims=True)
        net = jnp.reshape(net, (batch_size, -1))  # Flatten the features for the fully connected layers

        # Fully connected layers
        net = nn.Dense(features=512)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        net = nn.Dense(features=256)(net)
        net = nn.Dropout(rate=0.4, deterministic=not training)(net)  # Apply dropout during training
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay, epsilon=1e-3)(net)
        net = nn.relu(net)

        # Output layer (classification logits, assuming 40 classes for classification)
        net = nn.Dense(features=40)(net)

        return net, end_points

class PointNetBasic(nn.Module):
    """
    PointNetBasic: A simplified version of the PointNet architecture, designed for processing 3D point clouds.
    
    This version eliminates the input and feature transformation networks, focusing on the core feature extraction process.
    
    Args:
        nn.Module: The base class for Flax neural network modules.
    
    Returns:
        jax.Array: The output logits for classification tasks (for each input point cloud).
        dict: A dictionary containing intermediate endpoints (although in this case, it's empty).
    """
    
    @nn.compact
    def __call__(self, inputs, training: bool, bn_decay: float):
        """
        Forward pass for the simplified PointNet architecture. The input point clouds undergo convolutional layers,
        batch normalization, and pooling operations. Finally, fully connected layers produce the classification logits.
        
        Args:
            inputs (jax.Array): Input point cloud data of shape (batch_size, num_points, 3).
            training (bool): A flag indicating whether the model is in training mode.
            bn_decay (float): The decay factor for batch normalization during training.
        
        Returns:
            jax.Array: The output logits for each sample in the batch (shape: batch_size x num_classes).
            dict: A dictionary containing intermediate data (empty in this case).
        """

        batch_size = inputs.shape[0]  # Number of samples in the batch
        num_point = inputs.shape[1]   # Number of points in the point cloud
        end_points = {}

        # Input data
        input_image = inputs

        # First Convolutional Layer
        net = nn.Conv(features=64, kernel_size=1, padding='VALID', strides=1)(input_image)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay)(net)
        net = nn.relu(net)

        # Second Convolutional Layer
        net = nn.Conv(features=64, kernel_size=1, padding='VALID', strides=1)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay)(net)
        net = nn.relu(net)

        # Third Convolutional Layer
        net = nn.Conv(features=64, kernel_size=1, padding='VALID', strides=1)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay)(net)
        net = nn.relu(net)

        # Fourth Convolutional Layer
        net = nn.Conv(features=128, kernel_size=1, padding='VALID', strides=1)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay)(net)
        net = nn.relu(net)

        # Fifth Convolutional Layer
        net = nn.Conv(features=1024, kernel_size=1, padding='VALID', strides=1)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay)(net)
        net = nn.relu(net)

        # Max Pooling across all points
        net = nn.max_pool(inputs=net, window_shape=(num_point,), padding='VALID')
        net = jnp.reshape(net, (batch_size, -1))  # Flatten the features

        # First Fully Connected Layer
        net = nn.Dense(features=512)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay)(net)
        net = nn.relu(net)

        # Second Fully Connected Layer
        net = nn.Dense(features=256)(net)
        net = nn.BatchNorm(use_running_average=not training, momentum=bn_decay)(net)
        net = nn.relu(net)

        # Dropout for regularization during training
        net = nn.Dropout(rate=0.3, deterministic=not training)(net)

        # Output Layer (logits for classification)
        net = nn.Dense(features=40)(net)  # 40 classes (can be changed based on the dataset)

        return net, end_points
