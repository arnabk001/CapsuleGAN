# math libraries
import numpy as np

# ml libraries
import tensorflow as tf
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Concatenate, Multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

import numpy as np
from capsulelayers import CapsuleLayer, PrimaryCap, Length

K.set_image_data_format('channels_last')
# visualization
import skimage
from skimage import data, color, exposure
from skimage.transform import resize
import matplotlib.pyplot as plt
#%matplotlib inline

# sys and helpers
import sys
import os
import glob
from tqdm import tqdm

print('Modules imported.')

# device check
from tensorflow.python.client import device_lib
print('Devices:', device_lib.list_local_devices())

# GPU check
if not tf.test.gpu_device_name():
    print('No GPU found.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    
#%% LOADING THE DATA
def load_dataset(dataset, width, height, channels):
    
    if dataset == 'mnist':
        # load MNIST data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        
    if dataset == 'cifar10':
        # load CIFAR10 data
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

                
    # defining input dims
    img_rows = width
    img_cols = height
    channels = channels
    img_shape = [img_rows, img_cols, channels]
    
    return X_train, img_shape
    

# if MNIST ('mnist', 28, 28, 1) if CIFAR10 ('cifar10', 32, 32, 3)
#dataset, shape = load_dataset('cifar10', 32, 32, 3)
dataset, shape = load_dataset('mnist', 28, 28, 1)
print('Dataset shape: {0}, Image shape: {1}'.format(dataset.shape, shape))

#%% SQUASH LAYER FOR CAPSULE
# squash function of capsule layers, borrowed from Xifeng Guo's implementation of Keras CapsNet `https://github.com/XifengGuo/CapsNet-Keras`
def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

#%% DEFINING THE MODEL
# discriminator structure
#def build_discriminator():
#   
#    # depending on dataset we define input shape for our network
#    img = Input(shape=(shape[0], shape[1], shape[2]))
#
#    # first typical convlayer outputs a 20x20x256 matrix
#    x = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', name='conv1')(img)
#    #x = LeakyReLU()(x)
##    tf.get_variable_scope().reuse_variables()
##    x = BatchNormalization(momentum=0.8)(x)
#    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(x)
#    
#    # original 'Dynamic Routing Between Capsules' paper does not include the batch norm layer after the first conv group
#    
#    
#    """
#    NOTE: Capsule architecture starts from here.
#    """
#    
#    # filters 256 (n_vectors=8 * channels=32)
#    x = Conv2D(filters=8 * 32, kernel_size=9, strides=2, padding='valid', name='primarycap_conv2')(x)
#    
#    # reshape into the 8D vector for all 32 feature maps combined
#    
#    x = Reshape(target_shape=[-1, 8], name='primarycap_reshape')(x)
#    
#    # the purpose is to output a number between 0 and 1 for each capsule where the length of the input decides the amount
#    x = Lambda(squash, name='primarycap_squash')(x)   
#    # x = BatchNormalization(momentum=0.8)(x)
#    
#    # digitcaps are here
#    
#    x = Flatten()(x)
#    
#    uhat = Dense(160, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(x)
#    
#    c = Activation('softmax', name='softmax_digitcaps1')(uhat) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
#    
#    # s_j (output of the current capsule level) = uhat * c
#    c = Dense(160)(c) # compute s_j
#    x = Multiply()([uhat, c])
#    """
#    NOTE: Squashing the capsule outputs creates severe blurry artifacts, thus we replace it with Leaky ReLu.
#    """
##    s_j = LeakyReLU()(x)
#    s_j = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(x)
#
#
#    #
#    # we will repeat the routing part 2 more times (num_routing=3) to unfold the loop
#    #
#    c = Activation('softmax', name='softmax_digitcaps2')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
#    c = Dense(160)(c) # compute s_j
#    x = Multiply()([uhat, c])
##    s_j = LeakyReLU()(x)
#    s_j = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(x)
#
#    c = Activation('softmax', name='softmax_digitcaps3')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
#    c = Dense(160)(c) # compute s_j
#    x = Multiply()([uhat, c])
##    s_j = LeakyReLU()(x)
#    s_j = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(x)
#
#    pred = Dense(1, activation='sigmoid')(s_j)
#
#    
#    return Model(img, pred)

#%%
#def build_discriminator():
#    img_shape = (28,28,1)
#
#    model = Sequential()
#
#    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
##        model.add(LeakyReLU(alpha=0.2))
#    model.add(Activation("relu"))
#    model.add(Dropout(0.25))
#    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
#    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
##        model.add(LeakyReLU(alpha=0.2))
#    model.add(Activation("relu"))
#    model.add(Dropout(0.25))
##        model.add(BatchNormalization(momentum=0.8))
#    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
##        model.add(LeakyReLU(alpha=0.2))
#    model.add(Activation("relu"))
#    model.add(Dropout(0.25))
##        model.add(BatchNormalization(momentum=0.8))
#    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
##        model.add(LeakyReLU(alpha=0.2))
#    model.add(Activation("relu"))
#    model.add(Dropout(0.25))
#
#    model.add(Flatten())
#    model.add(Dense(1, activation='sigmoid'))
#
#    model.summary()
#
#    img = Input(shape=img_shape)
#    validity = model(img)
#
#    return Model(img, validity)
    
#%% build my discriminator
import numpy as np
from capsulelayers import CapsuleLayer, PrimaryCap, Length

def build_discriminator():
    img = Input(shape=(28,28,1))
    n_class = 10
    num_routing = 5
    
    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=7, strides=1, padding='valid', activation='relu', name='conv1')(img)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=7, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, num_routing=num_routing,
                             name='digitcaps')(primarycaps)
    

    # Layer 4: This is an auxiliary layer
    
    x = Flatten()(digitcaps)
    
#    uhat = Dense(160, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(x)
#    
#    c = Activation('softmax', name='softmax_digitcaps1')(uhat) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
#    
#    # s_j (output of the current capsule level) = uhat * c
#    c = Dense(160)(c) # compute s_j
#    x = Multiply()([uhat, c])
#   
#    decoder = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(x)
    dense1 = Dense(64, activation = 'relu')(x)
    pred = Dense(1, activation='sigmoid')(dense1)
    
    return Model(img, pred)
    

#%%    
## build and compile the discriminator
discriminator = build_discriminator()
print('DISCRIMINATOR:')
discriminator.summary()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    
#%%
# generator structure
def build_generator():

    """
    Generator follows the DCGAN architecture and creates generated image representations through learning.
    """

    noise_shape = (100,)
    x_noise = Input(shape=noise_shape)

    # we apply different kernel sizes in order to match the original image size
    
    if (shape[0] == 28 and shape[1] == 28):
        x = Dense(128 * 7 * 7, activation="relu")(x_noise)
        x = Reshape((7, 7, 128))(x)
        #x = BatchNormalization(momentum=0.8)(x)
        x = UpSampling2D()(x)
        x = Conv2D(128, kernel_size=3, padding="same")(x)
        x = Activation("relu")(x)
        #x = BatchNormalization(momentum=0.8)(x)
        x = UpSampling2D()(x)
        x = Conv2D(64, kernel_size=3, padding="same")(x)
        x = Activation("relu")(x)
        #x = BatchNormalization(momentum=0.8)(x)
        x = Conv2D(1, kernel_size=3, padding="same")(x)
        gen_out = Activation("tanh")(x)
        
        return Model(x_noise, gen_out)

    if (shape[0] == 32 and shape[1] == 32):
        x = Dense(128 * 8 * 8, activation="relu")(x_noise)
        x = Reshape((8, 8, 128))(x)
        #x = BatchNormalization(momentum=0.8)(x)
        x = UpSampling2D()(x)
        x = Conv2D(128, kernel_size=3, padding="same")(x)
        x = Activation("relu")(x)
        #x = BatchNormalization(momentum=0.8)(x)
        x = UpSampling2D()(x)
        x = Conv2D(64, kernel_size=3, padding="same")(x)
        x = Activation("relu")(x)
        #x = BatchNormalization(momentum=0.8)(x)
        x = Conv2D(3, kernel_size=3, padding="same")(x)
        gen_out = Activation("tanh")(x)

        
        return Model(x_noise, gen_out)

#%%    
# build and compile the generator
generator = build_generator()
print('GENERATOR:')
generator.summary()
generator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

#%%        
# feeding noise to generator
z = Input(shape=(100,))
img = generator(z)

# for the combined model we will only train the generator
discriminator.trainable = True

# try to discriminate generated images
valid = discriminator(img)

# the combined model (stacked generator and discriminator) takes
# noise as input => generates images => determines validity 
combined = Model(z, valid)
print('COMBINED:')
combined.summary()
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# loss values for further plotting
D_L_REAL = []
D_L_FAKE = []
D_L = []
D_ACC = []
G_L = []

#%%
def train(dataset_title, epochs, batch_size=32, save_interval=50):

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # select a random half batch of images
            idx = np.random.randint(0, dataset.shape[0], half_batch)
            imgs = dataset[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # generate a half batch of new images
            gen_imgs = generator.predict(noise)

            # train the discriminator by feeding both real and fake (generated) images one by one
            d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1))) # 0.9 for label smoothing
            d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # the generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * 32)

            # train the generator
            g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

            
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            D_L_REAL.append(d_loss_real)
            D_L_FAKE.append(d_loss_fake)
            D_L.append(d_loss)
            D_ACC.append(d_loss[1])
            G_L.append(g_loss)

            # if at save interval => save generated image samples
            if epoch % save_interval == 0:
                save_imgs(dataset_title, epoch)
                
#%%
def save_imgs(dataset_title, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = generator.predict(noise)

        # rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        
        # iterate in order to create a subplot
        for i in range(r):
            for j in range(c):
                if dataset_title == 'mnist':
                    axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                    axs[i,j].axis('off')
                    cnt += 1
                elif dataset_title == 'cifar10':
                    axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                    axs[i,j].axis('off')
                    cnt += 1
                else:
                    print('Please indicate the image options.')
        
        if not os.path.exists('images_{0}'.format(dataset_title)):
            os.makedirs('images_{0}'.format(dataset_title))
        
        fig.savefig("images_{0}/{1}.png".format(dataset_title, epoch))
        plt.close()
        
        
#%%
history = train('mnist', epochs=10000, batch_size=32, save_interval=50)
#generator.save('mnist_model.h5')
generator.save('mnist_model.h5')

#%% Visualization

plt.plot(D_L)
plt.title('Discriminator results (MNIST)')
plt.xlabel('Epochs')
plt.ylabel('Discriminator Loss (blue), Discriminator Accuracy (orange)')
plt.legend(['Discriminator Loss', 'Discriminator Accuracy'])
plt.savefig('Discriminator results2')
plt.show()

plt.plot(G_L)
plt.title('Generator results (MNIST)')
plt.xlabel('Epochs')
plt.ylabel('Generator Loss (blue)')
plt.legend('Generator Loss')
plt.savefig('Generator results2')
plt.show()

plt.plot(D_L_REAL)
plt.title('Discriminator results (MNIST)')
plt.xlabel('Epochs')
plt.ylabel('Discriminator Loss REAL (blue)')
plt.legend('Discriminator Loss REAL')
plt.savefig('Discriminator Loss REAL')
plt.show()

plt.plot(D_L_FAKE)
plt.title('Discriminator results (MNIST)')
plt.xlabel('Epochs')
plt.ylabel('Discriminator Loss FAKE (blue)')
plt.legend('Discriminator Loss FAKE')
plt.savefig('Discriminator Loss FAKE')
plt.show()
