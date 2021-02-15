#%% This script trains the fully convolutional network using keras with a 
# tensorflow backend, based on the saved training examples created in MATLAB.

# Import libraries and modules
import argparse
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import os
import numpy as np
import h5py

# Import modules and libraries
from keras.models import Model
from keras.layers import Input, Activation, UpSampling2D, regularizers,Convolution2D,Conv2DTranspose,MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import losses


# For MATLAB reading v7 data, we should set the locking to FALSE
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


filename0 		= "Input_data.mat"
filename1 		= "Ground_truth_label.mat"
weights_name 	= "Weight_output.hdf5"



def L2loss(input_shape):
    def bump_mse(output_true, ground_truth):

       
        # MSE
        loss_heatmaps = losses.mean_squared_error(output_true,ground_truth)
        return loss_heatmaps 
    return bump_mse


# Define the concatenated conv2, batch normalization, and relu block
def conv_bn_relu(nb_filter, rk, ck, name):
    def f(input):
        conv = Convolution2D(nb_filter, kernel_size=(rk, ck), strides=(1,1),kernel_regularizer=regularizers.l2(0.001),\
                               padding="same", use_bias=False,\
                               kernel_initializer="Orthogonal",name='conv-'+name)(input)
        conv_norm = BatchNormalization(name='BN-'+name)(conv)
        conv_norm_relu = Activation(activation = "relu",name='Relu-'+name)(conv_norm)
        return conv_norm_relu
    return f


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


# define a function that trains a model for a given data SNR and density
def train_model():
    
    
    # Read network input data
    
    matfile0    = h5py.File(filename0, 'r')
    patches     = np.array(matfile0['Input_data'])
    patches1    = patches;
    
    # Read network output data
    
    matfile1 	= h5py.File(filename1,'r')
    heatmaps 	= np.array(matfile1['Ground_truth_label'])
    heatmaps1 	= 100*np.array(heatmaps);
    
    
    # Assign training and validataion data into 75% v.s. 25%
    X_train, X_test, y_train, y_test = train_test_split(patches1, heatmaps1, test_size=0.25)
    
    print('Number of Training Examples: %d' % X_train.shape[0])
    print('Number of Validation Examples: %d' % X_test.shape[0])
    print('Number of Validation 1: %d' % X_test.shape[1])
    print('Number of Validation 2: %d' % X_test.shape[2])     
   
    # Setting type
    X_train 	= X_train.astype('float32')
    X_test 		= X_test.astype('float32')
    y_train 	= y_train.astype('float32')
    y_test 		= y_test.astype('float32')

	
	# Setting input outptut data size
    x_in_size 	= 128;      # input x
    z_in_size  	= 128;		# input z
    x_out_size 	= 512; 		# output x
    z_out_size 	= 512;		# output z
    
    frame_no = 5;  
    
    # Reshaping labels
    X_train_norm 	=	(X_train.reshape(X_train.shape[0], z_in_size,x_in_size,frame_no))
    X_test_norm 	=   (X_test.reshape(X_test.shape[0], z_in_size,x_in_size,frame_no))
    Y_train 		= 	(y_train.reshape(y_train.shape[0], z_out_size,x_out_size,1))
    Y_test  		= 	(y_test.reshape(y_test.shape[0], z_out_size,x_out_size,1))
    
    # Save the model weights after each epoch if the validation loss decreased
    save_best_check = ModelCheckpoint(weights_name, verbose=1,save_best_only=True)
        
    # Change learning when loss reaches a plataeu --> Added this due to the reviewer comment 
    modified_leaning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00005)
    
    # Model building and complitation
    model = buildModel((z_in_size,x_in_size,frame_no),(z_out_size,x_out_size, 1))
    
       
    # Create an image data generator for real time data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0.,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.,  # randomly shift images vertically (fraction of total height)
        zoom_range=0.,
        shear_range=0.,
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        fill_mode='nearest',
        data_format='channels_last')
    
    X_train_norm =(X_train_norm.reshape(X_train_norm.shape[0],z_in_size,x_in_size,frame_no))    
    
    # Inform user training begun
    print('Training model...')

    
    # Setting number of epoch, batch size and shuffle data, and save strategy
    train_history = model.fit_generator(datagen.flow(X_train_norm, Y_train, batch_size=32), \
          steps_per_epoch=200,epochs=200,verbose=1,\
          validation_data=(X_test_norm, Y_test),\
          shuffle=True,\
          callbacks=[save_best_check])

       
    # Inform user training ended
    print('Training Completed!')
    
 
    return
    
if __name__ == '__main__':
    
    # start a parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filename', type=str, help="path to generated training data m-file")
    
    
    parser.add_argument('--weights_name', type=str, help="path to save model weights as hdf5-file")
    parser.add_argument('--meanstd_name', type=str, help="path to save normalization factors as m-file")
    
    args = parser.parse_args()
    
    # run the training process
    train_model()
    
    