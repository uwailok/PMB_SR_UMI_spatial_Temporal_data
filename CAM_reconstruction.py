#%% This script tests the trained fully convolutmpimgnal network based on the 
# saved training weights, and normalizatmpimgn created using train_model.

# Import Libraries and model
import time
import numpy as np
import os
import h5py
import argparse
import scipy.io as sio

# Import modules and libraries
from keras.models import Model
from keras.layers import Input, Activation, UpSampling2D, regularizers,Convolution2D,Conv2DTranspose,MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import losses


def L2loss(input_shape):
    def bump_mse(output_true, ground_truth):

       
        # MSE
        loss_heatmaps = losses.mean_squared_error(output_true,ground_truth)
        return loss_heatmaps 
    return bump_mse


# Define the concatenated conv2, batch normalization, and relu block
def conv_bn_relu(nb_filter, rk, ck, name):
    def netout(input):
        conv = Convolution2D(nb_filter, kernel_size=(rk, ck), strides=(1,1),kernel_regularizer=regularizers.l2(0.001),\
                               padding="same", use_bias=False,\
                               kernel_initializer="Orthogonal",name='conv-'+name)(input)
        conv_norm = BatchNormalization(name='BN-'+name)(conv)
        conv_norm_relu = Activation(activation = "relu",name='tanh-'+name)(conv_norm)
        return conv_norm_relu
    return netout



# Define the model architechture

def FCN(input,names):
    
 
    Features1 = conv_bn_relu(8,3,3,names+'F1')(input)
    pool1 = MaxPooling2D(pool_size=(2,2),name=names+'Pool1')(Features1)
    Features2 = conv_bn_relu(16,3,3,names+'F2')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2),name=names+'Pool2')(Features2)
    Features3 = conv_bn_relu(32,3,3,names+'F3')(pool2)
    up5 = UpSampling2D(size=(2, 2),name=names+'Upsample1')(Features3)
    Features5 = conv_bn_relu(64,3,3,names+'F5')(up5)
    up6 = UpSampling2D(size=(2, 2),name=names+'Upsample2')(Features5)
    Features6 = conv_bn_relu(32,3,3,names+'F6')(up6)
    up7 = UpSampling2D(size=(2, 2),name=names+'Upsample4')(Features6)
    Features7 = conv_bn_relu(16,3,3,names+'F8')(up7)
    up8 = UpSampling2D(size=(2, 2),name=names+'Upsample5')(Features7)
    Features8 = conv_bn_relu(8,3,3,names+'F9')(up8)
    return Features8

# Define the Model building for an arbitrary input size
def buildModel(input_dim,output_dim):
    input_ = Input (shape = (input_dim))
    front_model = FCN (input_,'FCN')
    Final_layer = Conv2DTranspose(1, kernel_size=(1, 1), strides=(1, 1), padding="same",activation="linear", use_bias = False,\
                                  kernel_initializer="Orthogonal",name='Prediction')(front_model)
                   
    model = Model (inputs= input_, outputs=(Final_layer))
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss = L2loss(output_dim))
    
    return model


# Read CAM data
datafile = "CAM_Valid_data.mat"

# Read Weights
weights_file= "Weight_output.hdf5"


# For MATLAB reading v7 data, we should set the locking to FALSE
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def CAM_recon(datafile,weights_file):
    
       
    # load the tiff data
    matfile0    = h5py.File(datafile, 'r')
    Images1     = np.array(matfile0['Valid_data_input'])
    Images      = Images1;
    
    (U,M,N,P) = Images.shape
    
    frame_no = 5;
    # Build the model for a bigger image
    model = buildModel((M, N, frame_no),(M, N,1))

    # Load the trained weights
    model.load_weights(weights_file)
    
    
	# Setting input outptut data size
    x_in_size 	= 128;      # input x
    z_in_size  	= 128;		# input z
    
    Images_norm1 =(Images.reshape(Images.shape[0], z_in_size,x_in_size,frame_no))
    
    # Calculate reconstruction time
    start = time.time()
        
    
    predicted_output = model.predict(Images_norm1, batch_size=1)
    end = time.time()
    print(end - start)
    
    # threshold negative values
    predicted_output[predicted_output < 0] = 0
    
    CAM_SR_image = np.squeeze(predicted_output)
    
    

    # Save as a matfile to open later in matlab
    mdict = {"CAM_SR_image": CAM_SR_image}
    sio.savemat("CAM_SR_image.mat",mdict)
    


if __name__ == '__main__':
    
    # start a parser
    parser = argparse.ArgumentParser()
    
    # path of the tiff stack to be reconstructed
    parser.add_argument('--datafile', help="path to tiff stack for reconstructmpimgn")
    
    parser.add_argument('--weights_name', help="path to the trained model weights as hdf5-fil,saving using -v7.3")
    
    parser.add_argument('--debug', type=int, default=0, help="boolean (0/1) for saving individual predictmpimgns")
    
    # parse the input arguments
    args = parser.parse_args()
    
    # run the testing/reconstruction process
    CAM_recon(datafile, weights_file)
#    
    