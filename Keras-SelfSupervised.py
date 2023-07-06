#/usr/bin/python3
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.utils import SequenceEnqueuer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from skbio.alignment import local_pairwise_align_ssw 
from skbio import DNA
import mlflow
import mlflow.sklearn
import logging
import random as rand
from urllib.parse import urlparse
import sys
from tqdm.keras import TqdmCallback
from tables import *
import swalign
import datetime
from sklearn.metrics.pairwise import kernel_metrics
import jax.numpy as np
from jax.numpy import count_nonzero
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics import confusion_matrix
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
#from CNNGrimish2 import *
from SSLModel import *
from scipy import spatial,sparse
if __name__ == "__main__":
    
    #nts = "ATCG"
    nk=3
    r=20
    totCount = 0
    fldb = "../ML3nv-VCode/Data/BatCov_RpYN06.unmapped.fa"
    with open(fldb, "r",encoding="utf-8",errors='ignore') as f:
        totCount = (sum(bl.count(">") for bl in blocks(f))+1)
    print(f" num of lines {totCount}")
    setAb = list("ATCG")
    nkmlist = createallnkm(setAb, nk)
    ncl = len(nkmlist)
    print(ncl)
    #nrw = 150 - nk + 1
    nrw =160
    # Parameters
   
    BATCH_SIZE = float(sys.argv[1]) if len(sys.argv) > 1 else 32 #256
    EPOCHS = float(sys.argv[2]) if len(sys.argv) > 2 else 5 #10
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    params = {'dim': (nrw, ncl),
              'batch_size': BATCH_SIZE,
              'n_channels': 1,
              'shuffle': True}
    params_augmented = {'dim': (nrw, ncl),
              'batch_size': BATCH_SIZE,
              'n_channels': 1,
              'mapping':True,
              'shuffle': True}
    testparams = {'dim': (nrw, ncl),
              'batch_size': BATCH_SIZE,
              'n_channels': 1,
              'shuffle': False}
    partition = {}
    num_samples = 50000000
    num_training_samples = int(num_samples * 0.80 )
    num_validation_samples = int(num_samples * 0.20)
    #createGrimmdb(nkmlist,fldb, nk)
    #createTestdb(nkmlist, nk)
    alldbidx  = list(range(int(totCount)))
    #partition['train'] = alldbidx[1:6400000]#(int(totCount * (0.9)))]
    #partition['validation'] = alldbidx[6400000:7172884]
    partition['train'] = alldbidx[1:num_training_samples]#(int(totCount * (0.9)))]
    partition['validation'] = alldbidx[num_training_samples:num_validation_samples]
    partition['smtrain'] = alldbidx[1:100000]#(int(totCount * (0.9)))]
    partition['train_tsne'] = np.random.choice(partition['train'], replace=False, size=10000)
    #partition['validation'] = alldbidx[640000:717288]
    partition['test'] = list(range(1,5000))
    training_generator = DataGenerator(partition['train'], **params)
    training_generator_aug = DataGenerator(partition['train'], **params_augmented)
    #augmented_training_dataset = training_generator.apply(custom_augment)
    training_generator_tsne = DataGenerator(partition['train_tsne'], **params)
    training_generatorsm = DataGenerator(partition['smtrain'], **testparams)
    validation_generator = DataGenerator(partition['validation'], **params)
    #test_generator = TestGenerator(partition['test'], **testparams)
    #training_data = (pair for pair in zip(training_generator, training_generator_aug))
    #training_data = tf.data.Dataset.zip((training_generator, training_generator_aug))
    #print(len(training_data))
    #steps = EPOCHS * (num_training_samples // BATCH_SIZE)
    #lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.01, decay_steps=steps)
    #opt = tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6)
    
    with mlflow.start_run():
        
        steps = EPOCHS * (num_training_samples // BATCH_SIZE)
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001, decay_steps=steps
        )
        mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
        mlflow.log_param("EPOCHS", EPOCHS)
        ssl = SelfSupervisedLearning(get_encoder(), get_predictor())
        #vae.encoder.summary
        #VAE model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        opt = tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6)
        #vae.compile(opt)
        log_dir = "logs/SGD-ssl/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        ssl.compile(opt, metrics=[tf.keras.metrics.Accuracy()], run_eagerly=True)
        filepath = "Models/ssl00"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".h5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True, mode='min')

        hist= ssl.fit(training_generator_aug,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=80,
                        epochs=EPOCHS,
                        verbose=0, 
                        callbacks=[TqdmCallback(verbose=1),early_stopping, tensorboard_callback ]
                        )
        mlflow.log_metric("loss", float(hist.history['loss'][-1]))
        #mlflow.log_metric("reconstruction_loss", float(hist.history['reconstruction_loss'][-1]))
        #mlflow.log_metric("accuracy", float(hist.history['accuracy'][-1]))
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr_decayed_fn, "model", registered_model_name="Self-supervised")
        else:
            mlflow.sklearn.log_model(lr_decayed_fn, "model")
        ssl.encoder.save("Models/V02ssl_encoder"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),save_format='tf') 
        ssl.predictor.save("Models/V02ssl_predictor"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),save_format='tf') 
        #ssl.save("Models/ssl00")
        """
        prev_encoder = load_model("Models/V02ssl_encoder20220301-112643")
        prev_predictor = load_model("Models/V02ssl_predictor20220301-112654")
        Prev_ssl = SelfSupervisedLearning(prev_encoder, prev_encoder)
        Prev_ssl.compile(opt, metrics=[tf.keras.metrics.Accuracy()], run_eagerly=True)
        
        
        
        #filepath0 = "Models/V25ssl_encoder.h5"
        #checkpoint = ModelCheckpoint(filepath0, monitor='loss', verbose=1, save_best_only=True, mode='min')
        #callbacks_list = [checkpoint]
        hist0= Prev_ssl.fit(training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=64,
                    epochs= 1
                    #,callbacks=callbacks_list
                    )
        
        
        """
        
        # summarize history for loss
        plt.plot(hist.history['loss'])
        #plt.plot(hist.history['reconstruction_loss'])
        #plt.plot(hist.history['kl_loss'])
        plt.title('ssl model loss')
        plt.ylabel('total_loss')

        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')#, 'kl_loss'
        plt.show()
        plt.savefig("Loss-Graph/loss-RpYN06"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".png", bbox_inches='tight')
        
        #tf.config.run_functions_eagerly(True)
        #tf.data.experimental.enable_debug_mode()
        #"""
        pencoder =ssl.encoder   #Prev_ssl.predictor
        #Prev_ssl = load_model("Models/V01ssl_encoder")
        #Prev_ssl.compile(opt)
        #encoder0 = tf.keras.Model(Prev_ssl.input, Prev_ssl.layers[-3].output)
        e_train = pencoder.predict(training_generatorsm)
        #print(e_train.shape)
        #e_train_tsne = pencoder.predict(training_generator_tsne)

        
        #print(e_train_tsne.shape)
        #plot_tsne(e_train_tsne, "tsne-plots/tsne-RpYN06"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"V27ssl.png")
        #plt.show()
        testdata  = test_generation(partition['test'])
        x_test =[]
        #arrtest = np.empty((0, nrw, ncl), dtype= np.uint8)
        np.set_printoptions(threshold=sys.maxsize)

        #for x_t in test_generator:
        #    for xx in x_t:
        #        x_test.append(xx)
        #print(x_test[0])

        #test3d00 = np.append(arrtest,x_test[:5], axis=0)    
        x_train =[]
        for x_tr in training_generatorsm:
            for x in x_tr: 
                x_train.append(x)
        example = pencoder.predict(testdata)
        #testshape = K.shape(test_generator).eval(session=tf_session)
        #enqueuertest = SequenceEnqueuer(test_generator)
        #enqueuertest.start() #workers=40, max_queue_size=100
        #testdata = enqueuertest.get()
        print(testdata.shape)
        knn = NearestNeighbors(n_neighbors=5,algorithm='kd_tree', n_jobs=-1)
        knn.fit(e_train)

        #match = 2
        #mismatch = -1

        #scoring = swalign.NucleotideScoringMatrix(match, mismatch)
        #sw = swalign.LocalAlignment(scoring)
        print("Performing sequence similar retrieval on test Seqs...")
        seqs_retrieval =[]
        #for k in range(len(example)):
        for i, emb_flatten in enumerate(example):
            _, indices = knn.kneighbors([emb_flatten]) # find k nearest train neighbours
            X_test = reconstructSeqbyGF(nkmlist,testdata[i][:-12,:])
            print(X_test)
            sparsity = 1.0 - count_nonzero(testdata[i]) / testdata[i].size
            print(sparsity)
            #query = x_test[i] # query image
            for idx in indices.flatten():
                
                X_train = reconstructSeqbyGF(nkmlist,x_train[idx][:-12,:])
                dist = editdistance.eval(X_test, X_train)
                #alignment = sw.align(X_test,X_train)
                alignment = local_pairwise_align_ssw(DNA(X_test),DNA(X_train))
                print(X_train)
                print(f"testSeq# {i}, read_id  {idx}, editdist = {dist} = sparsity={ 1.0 - count_nonzero(x_train[idx]) / x_train[idx].size}")
                print(alignment)
                #seqs_retrieval.append(X_test[i],)
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))