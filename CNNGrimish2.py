#!/usr/bin/env python

import tensorflow as tf
import random as rand
import screed
import numpy as np
import editdistance
import matplotlib.pyplot as plt
from sklearn import manifold
import scipy.sparse
def int2seq(d):
    switcher = {
        0: "A",
        1: "T",
        2: "C",
    }
    return switcher.get(d, "G")
def is_DNA(seq):
    legal_dna = "ACGTN"
    """
    Returns 1 if it contains only legal values for a DNA sequence.
    c.f.  http://www.ncbi.nlm.nih.gov/BLAST/fasta.html
    """
    for ch in seq:
        if ch not in legal_dna:
            print(ch)
            return 0
    return 1
def reverse_complement(kmere):
    return screed.rc(kmere)
def createallnkm(setab, k):
    n = len(setab)
    out = list()
    createallnkmrec(setab, "", n, k, out)
    return out
def createallnkmrec(setab, prefix, n, k, out):

    setAb = list("ATCG")
    if (k == 0):
        out.append(prefix)
        return
    for i in range(n):
        # Next character of input added
        newPrefix = prefix + setAb[i]
        createallnkmrec(setab, newPrefix, n, k - 1, out)

def creategrimfilter(nkmlist, _seq , nk):
    seq = _seq.replace(" ", "").replace("\n", "")
    arr = np.zeros((len(seq) - nk+1, len(nkmlist)), dtype=bool)#[0]*len(nkmlist)
    for i in range(len(seq) - nk+1):
        for l in range(0,len(nkmlist)):
            if  nkmlist[l] == seq[i:i+nk]:
                arr[i][l] = True
                break
            else:
                continue
    return arr
#"""
def reconstructSeqbyGF(nkmlist,GFMatrix):
    seq = ""
    kmlist = []
    for rNum in range(0, len(GFMatrix)) :
        
        for gfbi in range(0,len(nkmlist)):
            if (GFMatrix[rNum][gfbi] == 0) and (gfbi == 255):
                seq+= "N"
            if GFMatrix[rNum][gfbi] == 1:
                if rNum == len(GFMatrix)-1:
                    seq+= nkmlist[gfbi]
                else:
                    seq+= nkmlist[gfbi][0]
                kmlist.append(nkmlist[gfbi])             
                break
    return seq
#def pyspcreategrimfilter( _seq):
    seq = _seq.replace(" ", "").replace("\n", "")
    arr = np.empty((len(seq)-nk+1, len(nkmlist)), dtype=bool)#[0]*len(nkmlist)
    for i in range(0,len(seq) - nk+1):
        for l in range(0,len(nkmlist)):
            if (nkmlist[l] == seq[i:i+nk]):
                arr[i][l] = True
                break
            else:
                arr[i][l] = False
    return arr
#"""
def createGrimmdb(nkmlist,refdb, nk ):
    with open(refdb,'r',encoding = "utf-8") as f:

        refLine = list()
        cnt =0
        line= f.readline()
        while line.strip():
            cnt+=1
            if line.startswith('>'):
                seq = ""
                line= f.readline().replace('\n','').strip()
                #seq += line

                #print(line)
            while not (line.startswith('>')) and line:
                seq += line.replace('\n','').strip()
                if is_DNA(line)== 0 :
                    print("somthing wrong!!!")
                    print(line)
                line= f.readline()
            grimf=creategrimfilter(nkmlist, seq, nk)
            grimarray0 = np.array(grimf,dtype=np.uint8)
            sparse_matrix = scipy.sparse.csc_matrix(grimarray0)
            scipy.sparse.save_npz("data/RpTN06-db-"+ str(cnt) +".npz", sparse_matrix)
            #np.save("data/RpTN06-db-"+ str(cnt) +".npy", grimarray0)
            #refLine.append(seq)
        #arr0 = np.empty((0, 147, 256), dtype= np.uint8)
        #grimarray = [creategrimfilter(nkmlist, x, nk) for x in refLine]
        #refLine= None
        #for id in range(len(grimarray)):
        #    grimarray0 = np.array(grimarray[id].reshape( 147, 256,1),dtype=np.uint8)
            
        #refSet0 = np.append(arr0,grimarray, axis=0)
        #grimarray= None
        #'''
    #refSet = np.array(refSet0.reshape(len(refSet0), 147, 256,1),dtype=np.uint8)
    #np.save("data/db-2.npy", refSet)
    print("db Done!")
    #print(f"this is refset shape{refSet.size()},{refSet[0].size()}")
    #return refSet #, refLine
#"""
#"""
def createTestdb(nkmlist, nk ):
    fname1 = "./reads/reads_(RaTG13)_1.fq"
    fname2 = "./reads/reads_(RaTG13)_2.fq"
    with open(fname1,'r',encoding = "utf-8") as testfile1,open(fname2,'r',encoding = "utf-8") as testfile2:
        line1= testfile1.readline()
        line2 = testfile2.readline()
        #testdic = dict()
        cnt = 0
        while line1.strip():
            if line1.startswith('@') and line2.startswith('@'):
                cnt+=1
                line1 = testfile1.readline().strip()
                line2 = testfile2.readline().strip()
                keyone = str(cnt)+".1"
                keytwo = str(cnt)+".2"
                if is_DNA(line1)== 0 :
                    print("somthing wrong!!!")
                    print(line1)
                grim1 = creategrimfilter(nkmlist, line1 , nk)
                #print(grim1)
                #testdic[keyone] = grim1
                np.save("Test/testRaTG13-"+ keyone +".npy", grim1)
                grim2 = creategrimfilter(nkmlist, line2 , nk)
                #testdic[keytwo] = grim2
                np.save("Test/testRaTG13-"+ keytwo +".npy", grim2)
            else:
                line1 = testfile1.readline()
                line2 = testfile2.readline()
        print("Test Db is built")
    #return testdic
def searchquery(querykms,nkmlist):

    qvec = [0] * len(nkmlist)
    for l in range(len(nkmlist)):
        if nkmlist[l] in querykms:
            qvec[l]=1

    return np.array(qvec, dtype=bool)

# Plot t-SNE
def plot_tsne(X,outFile):

    def plot_embedding(X, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        plt.figure()
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], ".", fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title, fontsize=18)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=None)
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne, "t-SNE DNA Representation")
    if outFile is None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
    plt.close()
def findmindist(nkmlist,item,x_train):
    minset = set()
    for grimkmere in x_train:
        dist = editdistance.eval(item, reconstructSeqbyGF(nkmlist,grimkmere))
        minset.add(dist)
    minlist = list(minset)
    minlist.sort()
    outlist = minlist[:10]
    return outlist
""""
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(160,256), n_channels=1, n_classes=10, shuffle=True): # labels,
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
        X, y = self.__data_generation(list_IDs_temp) 

        return X , y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

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
            padded =np.r_[ np.load("data/RpTN06-db-"+ str(ID) +".npy"),np.zeros((13,256,1))]
            X[i,] = padded

            # Store class
            y[i,] = X[i,]#self.labels[ID]
        return X , y #, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
class TestGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(147,256), n_channels=1, n_classes=10, shuffle=True): # labels,
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
        X, y = self.__test_generation(list_IDs_temp) 

        return X , y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __test_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim))
        # print(X.shape)
        # Generate data
        for i , ID in enumerate(list_IDs_temp):
            # Store sample
            #print(np.load('data/db-2.npy').shape)
            keyone = str(ID)+".1"
            #keytwo = str(ID)+".2"
            
            padded =np.r_[ np.load("test/testRaTG13-"+ keyone +".npy"),np.zeros((13,256))]
            X[i,] = padded
            #X[i,] = np.load("test/testRaTG13-"+ keytwo +".npy")
            # Store class
            y[i] = X[i,]#self.labels[ID]
        return X , y #, keras.utils.to_categorical(y, num_classes=self.n_classes)

class Sampling(tf.keras.layers.Layer):
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE0(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x,y = data
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(y, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded[2])
        return decoded
        


def cosin_loss(y_true,y_pred):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    y_pred = tf.stop_gradient(y_pred)
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((y_true * y_pred), axis=1))

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x,y = data
            z_mean, z_log_var, z = self.encoder(x)
            
            reconstruction = self.decoder(z)
            reconstruction_loss = cosin_loss(y, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients((grad, var)
                                       for (grad, var) in zip(grads, self.trainable_weights) 
                                       if grad is not None
                                       )
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    def call(self, inputs):
        self.train_step(inputs)
"""

def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b