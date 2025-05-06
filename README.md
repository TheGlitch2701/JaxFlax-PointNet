# JaxFlax-PointNet

JaxFlax-PointNet is an implementation of the PointNet architecture using JAX and Flax. This repository aims to provide a modern and efficient version of PointNet for researchers, students, and developers working on 3D point cloud data.

## Features

- **PointNet Architecture**: A deep learning model designed for processing 3D point cloud data.
- **JAX and Flax**: Leverages the power of JAX for high-performance numerical computing and Flax for flexible neural network modeling.
- **Educational Resource**: Ideal for university projects, research, or learning about 3D deep learning.

## Installation

1. **Clone the repository** :
    ```bash
    git clone https://github.com/yourusername/JaxFlax-PointNet.git
    cd JaxFlax-PointNet/
    ```

2. **Conda Environment Setup** :

    To run this first part of the project you need to create a **Conda Environment** with **python 3.12.5** following the instructions one can find on the [Anaconda Official Site](https://www.anaconda.com/).

    Moreover, be sure to set ***conda-forge*** as main channel for Anaconda.


3. **Install dependencies** :

    If you **DON'T HAVE** a **NVIDIA GPU**, please refer to **Part 3.1** to prepare a **CPU ONLY** environment (notice that this will heavily affect the total training time).

    3.1 **CPU Only Environment Installation (Not Recommended)** :

        pip install -r cpu_only_requirements.txt

    3.2 **NVIDIA GPU Environment Installation (Recommended)**:

        pip install -r gpu_requirements.txt

    

## Usage

1. **Training Phase:**
    ```bash
    cd src/
    sh training.sh
    ```

    Inside the **training.sh** file, one can find the full signature to run our presented simulation (note that since the algorithm is stochastic, the result might be different but the trend should be the same).

_____________________________________________________________

$$\bold{IMPORTANT}\\
\text{This error might occur if the ShapeNet website is down.}
$$

When running ```sh training.sh```, if the following error shows up: 

```
--  https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
Resolving shapenet.cs.stanford.edu (shapenet.cs.stanford.edu)... 171.67.77.19
Connecting to shapenet.cs.stanford.edu (shapenet.cs.stanford.edu)|171.67.77.19|:443... failed: Connection refused.
```

Please download the data manually from the following link:

```
https://polimi365-my.sharepoint.com/:f:/g/personal/10787939_polimi_it/EiEMCNq6_udCtkbJ9lrBAFUBEHESb8Nu1LJU-BwDmLLrDw
```

and put it inside the following directory after you have unzip it:

$$\text{src/data/} $$

_____________________________________________________________

In particular, if one want to customize some settings, here we listed all parameters that can be changed:

- ***log_dir*** : the directory where training files and figures are saved

- ***model*** : the selected model to use (can be either **pointnet_basic** or **pointnet**)

- ***num_point*** : the number of point to consider of each point cloud can be either one of the following: $$\bold{n} \in [\bold{256,512,1024,2048}] $$

- ***max_epoch*** : maximum number of epoch before the training ends: $$Default = \bold{250}$$

- ***batch_size*** : maximum number of **point clouds** inside each batch: $$Default = \bold{32} $$

- ***learning_rate*** : value of the learning rate for the optimizer used for the training of the Neural Network: $$Default = \bold{0.001}$$

- ***momentum*** : value for the momentum of the Convolutional layers inside the Neural Network: $$Default = \bold{0.9}$$

- ***optimizer*** : optimizer used for the training of the Neural Network; it can be either **'adam'** or **'momentum'**

- ***decay_step*** : number of step before having a decay inside the exponential decay scheduler: $$Default = \bold{200000}$$

- ***decay_rate*** : decay rate used by the exponential decay scheduler: $$Default = \bold{0.7}$$

At the end of the Training Loop, you'll find inside ***log_dir***:
- **directory called as a number** which contains the checkpoint of the Neural Network,
- a **log_train.txt** file, in which you can see some info about the training process
- a **Loss & Accuracy** picture where both the Loss (CrossEntropy) and the Accuracy of the Training and Validation Dataset are shown.
________________________________________________________________

2. **Evaluation Phase:**
    ```bash
    python evaluate_jax.py --log_dir=log_prova --model=pointnet --num_point=1024 --batch_size=4 --dump_dir=dump_prova --optimizer=adam --checkpoint=best_checkpoint_from_training
    ```

At the end of this phase, you'll find inside the directory ***dump_dir*** the following files:

- **log_evaluate.txt**, in which the accuracy of each class is listed combined with the eval mean loss, the eval accuracy among all object and the eval accouracy among all classes

- **Flax_Accuracy.tar**, a file containing the accuracies for further comparisons.

________________________________________________________________

3. **Comparing Phase:**
    ```bash
    python compare.py --dump_flax=dump_prova --dump_tf=../old_repo_result/dump_tf_250_1024
    ```

In the end you'll find inside the ***dump_dir*** (same directory as for the previous point), a plot with the Jax&Flax Accuracies on the x-axis and the Tensorflow Version Accuracies on the y-axis.

This is usefull to see the comparison between the Tensorflow approach with our Jax&Flax one.

Note that the times for training is really low for our approach with respect to the Tensorflow one.

## Results
This is the result that we achieve:

-   **Maximum Training Accuracy : 89.016 %**
-   **Respective CrossEntropy Loss : 0.218748**


-   **Maximum Validation Accuracy : 88.938 %**
-   **Respective CrossEntropy Loss : 0.388915**

Here's a picture presenting the Loss and the Accuracy troughout epochs:


![Results](./doc/training_files/Loss%20&%20Accuracy.png)


For what concerns the evaluation part, here we show our results:

![Evaluation Results](./doc/Compared%20Accuracies.png)

Note that you can find more details about the training, evaluation and *comparing* results inside the ```doc``` directory.

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation is inspired by the original [PointNet paper](https://arxiv.org/abs/1612.00593) and aims to provide a JAX/Flax-based alternative for the community.

