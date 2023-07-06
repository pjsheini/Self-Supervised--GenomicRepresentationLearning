import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from CNNGrimish2 import *
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import random
nk=3
BATCH_SIZE = 128
setAb = list("ATCG")
nkmlist = createallnkm(setAb, nk)
ncl = len(nkmlist)
#nrw = 150 - nk + 1
nrw =160
import resnet_cifar10_v2

N = 2
DEPTH = N * 9 + 2
NUM_BLOCKS = ((DEPTH - 2) // 9) - 1
LATENT_DIM = 128
WEIGHT_DECAY = 0.0005
PROJECT_DIM = 256
# Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def get_encoder():
    # Input and backbone.
    
    inputs = tf.keras.layers.Input((nrw, ncl,1))
    inpmasked = tf.keras.layers.Masking(mask_value=0.,input_shape=(nrw, ncl,1))(inputs)
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5 , offset=-1)(
        inpmasked
    )
    x = resnet_cifar10_v2.stem(x)
    x = resnet_cifar10_v2.learner(x, NUM_BLOCKS)
    x = tf.keras.layers.GlobalAveragePooling2D(name="backbone_pool")(x)
    # Projection head.
    x = tf.keras.layers.Reshape(( PROJECT_DIM, 1))(x)
    x = tf.keras.layers.Dense(
        PROJECT_DIM, use_bias=False,  activation= 'tanh')(x)    
    x = attention()(x)
    x = tf.keras.layers.Dense(
        PROJECT_DIM, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(inputs, outputs, name="encoder")

def get_predictor():  ## LATENT_DIM replace w Project
    activation='softmax'
    predictor_input = tf.keras.Input(shape = (PROJECT_DIM, 1) , name = "predictor_input")
    RNN_layer = tf.keras.layers.Dense(LATENT_DIM, trainable=True, activation=activation)(predictor_input)
    attention_layer = attention()(RNN_layer)
    outputs=tf.keras.layers.Dense(PROJECT_DIM, trainable=True, activation= 'ReLU')(attention_layer)
    return tf.keras.Model(predictor_input, outputs, name="encoder")
    

def cosinLoss(y_true,y_pred):
        
    #print(y_true.dtype,y_pred.dtype)
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    y_pred = tf.stop_gradient(y_pred)
    l2y_true = tf.math.l2_normalize(y_true, axis=1)
    l2y_pred = tf.math.l2_normalize(y_pred, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((l2y_true * l2y_pred), axis=1))
class SelfSupervisedLearning(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        encoder,
        predictor,
        name="Vautoencoder",
        **kwargs
        ):
        super(SelfSupervisedLearning, self).__init__(name=name, **kwargs)
        #self.original_dim = (nrw, ncl,1)
        self.encoder = encoder
        self.predictor = predictor
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        #self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
        #    name="reconstruction_loss"
        #)
        #self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.train_accuracy_tracker = tf.keras.metrics.Accuracy('train_accuracy')
        #self.reconstruction_loss_mean_tracker = tf.keras.metrics.Mean(name="reconstruction_loss_mean")
        #self.reconstruction_loss_log_tracker = tf.keras.metrics.Mean(name="reconstruction_loss_log")
        #self.bce_loss_tracker = tf.keras.metrics.Mean(name="bce_loss")
    def augmentingData(self, seqm):
        num =8  
        sq = custom_augment(seqm,num)
        return sq
    def train_step(self, data):
 
        #print (data.shape)
        ds = data
        ds_aug = tf.map_fn(self.augmentingData, data)
        #print(ds.shape)
        #(ds, ds_aug) = data
        
        #print(ds_aug.shape)
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds), self.encoder(ds_aug)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            #z_mean, z_log_var, z = self.encoder(ds)
            #z_mean_aug, z_log_var_aug, z_aug = self.encoder(ds_aug)
            #reconstruction = self.decoder(z)
            #reconstruction_aug = self.decoder(z_aug)
            #reconstruction_mean = self.decoder(z_mean)
            #reconstruction_log_var = self.decoder(z_log_var)
            cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,reduction=tf.keras.losses.Reduction.SUM)
            total_loss = cosine_loss(p1, z2) + cosine_loss(p2, z1)+ cosine_loss(p1, z1)
            #reconstruction_loss = cosinLoss(p1, z2)/2 + cosinLoss(p2, z1)/2
            """
            bce_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(reconstruction_log_var, reconstruction), axis=(0, 1)
                )
            )
            """
            #kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            #kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            #total_loss =  reconstruction_loss
            #total_loss = bce_loss + reconstruction_loss#+ reconstruction_loss_log/2+ reconstruction_loss_mean/2
        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        grads = tape.gradient(total_loss, learnable_params)
        self.optimizer.apply_gradients((grad, var)
                                    for (grad, var) in zip(grads, self.trainable_weights) 
                                    if grad is not None
                                    )
        self.total_loss_tracker.update_state(total_loss)
        #self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        #self.reconstruction_loss_mean_tracker.update_state(reconstruction_loss_mean)
        #self.reconstruction_loss_log_tracker.update_state(reconstruction_loss_log)
        #print(p1.shape,z1.shape, ds.shape)
        self.train_accuracy_tracker(p1, p2)
        #self.bce_loss_tracker.update_state(bce_loss)
        #self.kl_loss_tracker.update_state(kl_loss)
        return {
            "accuracy": self.train_accuracy_tracker.result(),
            "loss": self.total_loss_tracker.result(),
            #"reconstruction_loss": self.reconstruction_loss_tracker.result(),
            #"reconstruction_loss_mean": self.reconstruction_loss_mean_tracker.result(),
            #"reconstruction_loss_log": self.reconstruction_loss_log_tracker.result(),
            #"bce_loss": self.bce_loss_tracker.result(),
            #"kl_loss": self.kl_loss_tracker.result(),
        }
    @property
    def metrics(self):
        return [
            self.train_accuracy_tracker,
            self.total_loss_tracker,
            #self.reconstruction_loss_tracker,
            #self.reconstruction_loss_mean_tracker,
            #self.reconstruction_loss_log_tracker,
            #self.bce_loss_tracker,
            #self.kl_loss_tracker,
        ]
    def call(self, inputs):
        #x , y  = inputs
        return self.train_step(inputs)
    
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(160,64), mapping=False, n_channels=1, n_classes=10, shuffle=True): # labels,
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = list_IDs
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.mapping = mapping
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print(indexes)
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #print(list_IDs_temp)
        # Generate data
        X = self.__data_generation(list_IDs_temp) 

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def augmentingData(self, seqm,num):
        sq = seqm
        if self.mapping:
            sq = custom_augment(seqm,num)
        
        return sq
            

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))
        # print(X.shape)
        # Generate data
        for i , ID in enumerate(list_IDs_temp):
            # Store sample
            #print(np.load('data/db-2.npy').shape)
            sparse_matrix = (scipy.sparse.load_npz("../MLEnv-VCode/data/RpTN06-db-"+ str(ID) +".npz").todense())
            sparse_matrix = np.expand_dims(sparse_matrix,axis=2)
            #print(sparse_matrix.shape)
            X[i,]=np.r_[ sparse_matrix,np.zeros((12,64,1))]
            #sparse_matrix_aug = self.augmentingData(sparse_matrix,4)
            #padded =np.r_[ sparse_matrix_aug,np.zeros((13,256,1))]
            #y[i,] = padded

            # Store class
            #y[i,] = X[i,]#self.labels[ID]
        return  X
    
class TestGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(160,64), n_channels=1, n_classes=10, shuffle=False, mapping = False): # labels,
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = list_IDs
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print(indexes)
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #print(list_IDs_temp)
        # Generate data
        X = self.__test_generation(list_IDs_temp) 

        return X 

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __test_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #y = np.empty((self.batch_size, *self.dim, self.n_channels))
        # print(X.shape)
        # Generate data
        for i , ID in enumerate(list_IDs_temp):
            # Store sample
            #print(np.load('data/db-2.npy').shape)
            keyone = str(ID)+".1"
            #keytwo = str(ID)+".2"
            notpadded = np.load("../MLEnv-VCode/Test/testRaTG13-"+ keyone +".npy").reshape( 148, 64,1)
            padded =np.r_[ notpadded,np.zeros((12,64,1))]
            X[i,] = padded
            #X[i,] = np.load("test/testRaTG13-"+ keytwo +".npy")
            # Store class
            #y[i] = X[i,]#self.labels[ID]
        return X #, y #, keras.utils.to_categorical(y, num_classes=self.n_classes)
def test_generation( list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        dim=(160,64)
        X = np.empty((len(list_IDs_temp), *dim, 1))
        # print(X.shape)
        # Generate data
        for i , ID in enumerate(list_IDs_temp):
            # Store sample
            #print(np.load('data/db-2.npy').shape)
            keyone = str(ID)+".1"
            #keytwo = str(ID)+".2"
            notpadded = np.load("../MLEnv-VCode/Test/testRaTG13-"+ keyone +".npy").reshape( 148, 64,1)
            padded =np.r_[ notpadded,np.zeros((12,64,1))]
            X[i,] = padded

        return X 
def shift_horizental(seq,num, fill_value = np.nan):
    result = np.empty((seq.shape[0], seq.shape[1], seq.shape[2]))
    if num > 0:
        result[:num] = fill_value
        result[num:] = seq[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = seq[-num:]
    else:
        result[:] = seq
    return result
def shift_vertical(seq,num):
    n = random.randint(1, 11)
    shifted = np.roll(seq,num*n,axis=1)
    return shifted
def custom_augment(tensoor_seq,num):
    seq = tensoor_seq.numpy()[:-12,:,:]
    if seq.ndim >3 :
        seq = seq[:,:,:,0]
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations (except for random crops) need to be applied
    # randomly to impose translational invariance.
    seq = shift_vertical(seq,num)
    output =np.r_[ seq ,np.zeros((12,64,1))]
    return tf.convert_to_tensor(output, dtype=tf.float32)
