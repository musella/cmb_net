import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt
import logging
from keras import backend as K

#Clip the predicted logarithm to -val ... +val
log_r_clip_value = 10.0

#https://stackoverflow.com/questions/45250100/kerasregressor-coefficient-of-determination-r2-score
def r2_score(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def neg_r2_score(y_true, y_pred):
    return -r2_score(y_true, y_pred)
def on_epoch_end(epoch, logs):
    K.set_learning_phase(False)
    #get layer weight statistics
    #for layer in mod.layers:
    #    weights = layer.get_weights()
    #    means = []
    #    stds = []
    #    if len(weights)>0:
    #        for weight_mat in weights:
    #            weight_mat_flat = weight_mat.flatten()
    #            stds += [np.std(weight_mat_flat)]
    #            means += [np.mean(weight_mat_flat)]
    #            #if "dense_4" in layer.name:
    #            #    print(weight_mat)
    #    logging.info("epoch_weight {0} {1} means={2} stds={3}".format(epoch, layer.name, means, stds))
    #Xparton_pred = mod.predict(X[:5])
    #
    #for i in range(5):
    #    s = ""
    #    for j in range(Xparton.shape[1]):
    #        s += "{0:.2f} ({1:.2f}) | ".format(Xparton_pred[i,j], Xparton[i,j])
    #    print s

    #weights = mod.trainable_weights
    #gradients = K.gradients(mod.total_loss, weights)
    #for grad in gradients:
    #    gradvals = grad.eval(session=K.get_session(), feed_dict={
    #        mod.input: X_train[:1000],
    #        mod.sample_weights[0]: w_train[:1000],
    #        mod.targets[0]: y_train[:1000].reshape(1000,1)
    #    }).flatten()
    #    #if "dense_4" in grad.name:
    #    #    print(gradvals)
    #    logging.info("epoch_grad {0} {1} means={2} stds={3}".format(epoch, grad.name, np.mean(gradvals.flatten()), np.std(gradvals.flatten())))
    #logging.info("epoch_end {0} {1} {2} {3} {4}".format(epoch, logs["loss"], logs["val_loss"], logs["main_output_r2_score"], logs["val_main_output_r2_score"]))
    s = " ".join(["{0}={1:.4f}".format(k,v) for k, v in sorted(logs.items(), key=lambda x: x[0])])
    logging.info("epoch_end {0}".format(s))
    K.set_learning_phase(True)

def get_activation(activation):
    if activation == "relu":
        return keras.layers.Activation("relu")
    elif activation == "leakyrelu":
        return keras.layers.LeakyReLU(alpha=0.1)
    elif activation == "prelu":
        return keras.layers.PReLU()
    elif activation == "elu":
        return keras.layers.ELU()
    elif activation == "tanh":
        return keras.layers.Activation("tanh")
    elif activation == "selu":
        return keras.layers.Activation("selu")
    else:
        raise Exception("Unknown activation: {0}".format(activation))

def load_data(infile):
    #load the input data
    inf = open(infile, "rb")
    data = np.load(inf)
    Xreco = data["X_reco"]
    Xcart = data["X_cart"]
    Xparton_cart = data["X_parton_cart"]
    #Xadditional = data["Xadditional"]
    Xmatch = data["X_match"]
    
    #shuffle the input data
    shuf = np.random.permutation(range(Xreco.shape[0]))
    logging.info("Xreco={0}".format(Xreco[:5]))
    Xreco = Xreco[shuf]
    Xparton_cart = Xparton_cart[shuf]
    Xmatch = Xmatch[shuf]
    #Xadditional = Xadditional[shuf]
    y = data["y"][:, -1][shuf]
    
    logging.info("y={0}".format(y[:5]))
    
    cut = np.isfinite(y)
    logging.info("applying cut to be finite, passed {0}/{1}".format(np.sum(cut), y.shape[0]))
    Xreco = Xreco[cut]
    Xparton_cart = Xparton_cart[cut]
    #Xadditional = Xadditional[cut]
    Xmatch = Xmatch[cut]
    y = y[cut]
    return Xreco, Xmatch, Xparton_cart, y

def input_statistics(X, name, filename):
    ixs = []
    X_means = []
    X_stds = []
    X_maxs = []
    X_mins = []
    #print statistics for inputs
    for ix in range(X.shape[1]):
        logging.info("X[{0}] mean={1:.4f} std={2:.4f} min={3:.4f} max={4:.4f}".format(
            ix,
            np.mean(X[:, ix]),
            np.std(X[:, ix]),
            np.min(X[:, ix]),
            np.max(X[:, ix])
        ))
        ixs += [ix]
        X_means += [np.mean(X[:, ix])]
        X_stds += [np.std(X[:, ix])]
        X_maxs += [np.max(X[:, ix])]
        X_mins += [np.min(X[:, ix])]
    X_means = np.array(X_means)
    X_stds = np.array(X_stds)
    X_maxs = np.array(X_maxs)
    X_mins = np.array(X_mins)

    plt.figure()
    plt.errorbar(ixs, X_means, yerr=X_stds)
    plt.bar(ixs, width=1, bottom=X_mins, height=(X_maxs - X_mins), alpha=0.2, color="gray")
    plt.savefig("{0}/{1}".format(name, filename))

def build_densenet(X, nlayers, dropout, layersize, batchnorm, activation, layer_reg, reduce_layersize=True):
    mod = keras.models.Sequential()
    mod.add(keras.layers.InputLayer(input_shape=(X.shape[1], )))
    layersize = layersize * 2
    for i in range(nlayers):
        #redue the layer size by 2 for every next layer
        if reduce_layersize:
            layersize = int(layersize / 2)
    
        if batchnorm:
            mod.add(keras.layers.BatchNormalization())
        
        mod.add(keras.layers.Dense(layersize,
            kernel_regularizer=keras.regularizers.l2(layer_reg),
            bias_regularizer=keras.regularizers.l2(layer_reg)
        ))
        mod.add(get_activation(activation))
        if dropout > 0.0:
            dropout_amount = dropout
    
            #less dropout in first hidden layer
            if i == 0:
                dropout_amount = dropout_amount / 2.0
            mod.add(keras.layers.Dropout(dropout_amount))
    mod.add(keras.layers.Dense(1, activation="linear"))
    return mod

#loss function between predicted and true log values
def loss_function_ratio_regression(y_true, y_pred):
    from keras import backend as K
    r_loss = keras.losses.mean_squared_error(
        K.exp(K.clip(y_true, -log_r_clip_value, log_r_clip_value)),
        K.exp(K.clip(y_pred, -log_r_clip_value, log_r_clip_value)))
    return r_loss

def loss_function_p4(y_true, y_pred):
    from keras import backend as K
    r_loss = K.mean(K.square(y_pred - y_true), axis=-1)/K.std(y_true, axis=-1) 
    return r_loss

def build_parton_net(X, nlayers, dropout, layersize, batchnorm, activation, layer_reg):
    inputs = keras.layers.Input(shape=(X.shape[1], ))
    prev = inputs

    for i in range(nlayers):
        if batchnorm:
            prev = keras.layers.BatchNormalization()(prev)
        prev = keras.layers.Dense(layersize,
            kernel_regularizer=keras.regularizers.l2(layer_reg),
            bias_regularizer=keras.regularizers.l2(layer_reg)
        )(prev)
        prev = get_activation(activation)(prev)
        if dropout > 0.0:
            dropout_amount = dropout
            prev = keras.layers.Dropout(dropout_amount)(prev)
    
    ib_layer = keras.layers.Dense(4*4, name="ib_layer")(prev)
    model = keras.models.Model(inputs=inputs, outputs=ib_layer)
    return model

def build_ibnet(X, nlayers, dropout, layersize, batchnorm, activation, layer_reg):
    inputs = keras.layers.Input(shape=(X.shape[1], ))
    prev = inputs

    for i in range(nlayers):
        if batchnorm:
            prev = keras.layers.BatchNormalization()(prev)
        prev = keras.layers.Dense(layersize,
            kernel_regularizer=keras.regularizers.l2(layer_reg),
            bias_regularizer=keras.regularizers.l2(layer_reg)
        )(prev)
        prev = get_activation(activation)(prev)
        if dropout > 0.0:
            dropout_amount = dropout
            prev = keras.layers.Dropout(dropout_amount)(prev)
    
    ib_layer = keras.layers.Dense(4*4, name="ib_layer")(prev)
    aux_layer = keras.layers.Dense(128, name="aux")(prev)
    aux_layer_act = get_activation(activation)(aux_layer)

    concat = keras.layers.Concatenate()([ib_layer, aux_layer_act])
    prev = concat
    
    for i in range(nlayers):
        if batchnorm:
            prev = keras.layers.BatchNormalization()(prev)
        prev = keras.layers.Dense(layersize,
            kernel_regularizer=keras.regularizers.l2(layer_reg),
            bias_regularizer=keras.regularizers.l2(layer_reg)
        )(prev)
        prev = get_activation(activation)(prev)
        if dropout > 0.0:
            dropout_amount = dropout
            prev = keras.layers.Dropout(dropout_amount)(prev)
    pred_ratio = keras.layers.Dense(1, activation="linear", use_bias=False, name="main_output")(prev)

    model = keras.models.Model(inputs=inputs, outputs=[pred_ratio, ib_layer])
    return model
