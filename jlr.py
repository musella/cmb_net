from __future__ import print_function
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow
import keras
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging

from keras import losses
from keras import backend as K
from matplotlib.colors import LogNorm

#Clip the predicted logarithm to -val ... +val
log_r_clip_value = 10.0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", type=str,
    default="jlr_data_reco.npz",
    help="Input file",
)
parser.add_argument(
    "--layers", type=int,
    default=3, action="store",
    help="Number of intermediate layers"
)
parser.add_argument(
    "--verbosity", type=int,
    default=0, action="store",
    help="Training verbosity"
)
parser.add_argument(
    "--seed", type=int,
    default=1, action="store",
    help="The random seed"
)
parser.add_argument(
    "--batch_size", type=int,
    default=10000, action="store",
    help="Batch size"
)
parser.add_argument(
    "--layersize", type=int,
    default=128, action="store",
    help="Layer size"
)
parser.add_argument(
    "--activation", type=str,
    default="tanh", action="store",
    choices=["relu", "leakyrelu", "prelu", "tanh", "elu"],
    help="Activation function"
)
parser.add_argument(
    "--dropout", type=float,
    default=0.0, action="store",
    help="Amount of dropout"
)
parser.add_argument(
    "--lr", type=float,
    default=0.0001, action="store",
    help="Learning rate"
)
parser.add_argument(
    "--epochs", type=int,
    default=200, action="store",
    help="Number of training epochs"
)
parser.add_argument(
    "--ntrain", type=int,
    default=0, action="store",
    help="Number of training events"
)
parser.add_argument(
    "--ntest", type=int,
    default=0, action="store",
    help="number of testing events"
)
parser.add_argument(
    "--earlystop", type=int,
    default=20, action="store",
    help="Early stopping epochs"
)
parser.add_argument(
    "--clipnorm", type=float,
    default=1.0, action="store",
    help="Clip normalization"
)
parser.add_argument(
    "--layer_reg", type=float,
    default=0.00, action="store",
    help="Layer regularization (L2)"
)
parser.add_argument(
    "--batchnorm",
    action="store_true",
    help="Batch normalization"
)
parser.add_argument(
    "--do_norm",
    action="store_true",
    help="Normalize and standardize inputs"
)
parser.add_argument(
    "--do_weight",
    action="store_true",
    help="Reweight target distribution to be flat"
)
parser.add_argument(
    "--do_logtarget",
    action="store_true",
    help="Take log transform of target"
)
parser.add_argument(
    "--do_varplots",
    action="store_true",
    help="Do plots of input variables"
)
parser.add_argument(
    "--do_tensorboard",
    action="store_true",
    help="Do Tensorboard, slows the training somewhat"
)

args = parser.parse_args()

#set the random seed
from numpy.random import seed
seed(args.seed)
from tensorflow import set_random_seed
set_random_seed(args.seed)

#create a unique name for the training
name = "tr_l{layers}x{layersize}_d{dropout:.2f}_{activation}_lr{lr:.7f}_bn{batchnorm}_dn{do_norm}_w{do_weight}_{inp}_{ntrain}_{ntest}_cn{clipnorm:.2f}_reg{layer_reg:.2f}_b{batch_size}_s{seed}".format(
    layers=args.layers, layersize=args.layersize,
    dropout=args.dropout, activation=args.activation,
    lr=args.lr, batchnorm=int(args.batchnorm),
    do_norm=int(args.do_norm), do_weight=int(args.do_weight),
    inp=os.path.basename(args.input),
    ntrain=args.ntrain, ntest=args.ntest,
    clipnorm=args.clipnorm, layer_reg=args.layer_reg,
    batch_size=args.batch_size,
    seed=args.seed
)
os.makedirs(name)
logging.basicConfig(
    format='%(asctime)s %(name)s %(message)s',
    filename="{0}/log.log".format(name),
    level=logging.INFO,
    filemode="w"
)
print("name " + name)

def load_data(infile):
    #load the input data
    inf = open(infile, "rb")
    data = np.load(inf)
    X = data["X"]
    
    #shuffle the input data
    shuf = np.random.permutation(range(X.shape[0]))
    logging.info("X={0}".format(X[:5]))
    X = X[shuf]
    y = data["y"][:, -1][shuf]
    
    if args.do_logtarget:
        y = np.log(y)
    
    logging.info("y={0}".format(y[:5]))
    
    cut = np.isfinite(y)
    logging.info("applying cut to be finite, passed {0}/{1}".format(np.sum(cut), y.shape[0]))
    X = X[cut]
    y = y[cut]
    return X, y

X, y = load_data(args.input)
#create histogram for target
logging.info("shapes {0} {1}".format(X.shape, y.shape))
ybins = np.linspace(np.mean(y) - 6*np.std(y), np.mean(y) + 6*np.std(y), 100)
c, b = np.histogram(y, bins=ybins)
ib = np.searchsorted(b, y)

#compute weights for target to be flat
w = np.ones(X.shape[0])
if args.do_weight:
    w = np.array([c[_ib] if _ib < c.shape[0] else 0.0 for _ib in ib])
    w = 1000.0/w
    w[np.isinf(w)] = 0.0
    w[np.isnan(w)] = 0.0

#normalize the inputs
if args.do_norm:
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    logging.info("means={0}".format(means))
    logging.info("stds={0}".format(stds))
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - means[i])/stds[i]
    X[~np.isfinite(X)] = 0.0

    #don't want to normalize y as that changes the loss function value
    #mean = np.mean(y)
    #std = np.std(y)
    #y = (y-mean)/std

def input_statistics(X, filename):
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

logging.info("y mean={0:.4f} std={1:.4f} min={2:.4f} max={3:.4f}".format(
    np.mean(y),
    np.std(y),
    np.min(y),
    np.max(y)
))

#do plots of the input variables
if args.do_varplots:
    for ix in range(X.shape[1]):
        plt.figure()
        plt.hist(X[:, ix], bins=100)
        plt.savefig("{0}/src_{1}.pdf".format(name, ix), weights=w)
        
        plt.figure()
        plt.hexbin(X[:, ix], y[:], bins=100, norm=LogNorm(1, X.shape[0]))
        plt.savefig("{0}/src_tgt_{1}.pdf".format(name, ix), weights=w)

#plot the target
plt.figure()
plt.hist(y, bins=ybins)
plt.savefig("{0}/target_unw.pdf".format(name))

#choose test and training events
if args.ntrain == 0 and args.ntest == 0:
    ntrain = int(0.8*X.shape[0])
    X_train = X[:ntrain]
    y_train = y[:ntrain]
    w_train = w[:ntrain]
    
    X_test = X[ntrain:]
    y_test = y[ntrain:]
    w_test = w[ntrain:]
else:
    X_train = X[:args.ntrain]
    y_train = y[:args.ntrain]
    w_train = w[:args.ntrain]
    
    X_test = X[args.ntrain:args.ntrain+args.ntest]
    y_test = y[args.ntrain:args.ntrain+args.ntest]
    w_test = w[args.ntrain:args.ntrain+args.ntest]

input_statistics(X_train, "inputs_train.pdf")
input_statistics(X_test, "inputs_test.pdf")

#plot the target
plt.figure()
plt.hist(y_train, bins=ybins, weights=w_train)
plt.hist(y_test, bins=ybins, weights=w_test)
plt.savefig("{0}/target.pdf".format(name))

mod = keras.models.Sequential()
mod.add(keras.layers.InputLayer(input_shape=(X.shape[1], )))

for i in range(args.layers):
    #redue the layer size by 2 for every next layer
    layersize = int(args.layersize / pow(2, i))

    if args.batchnorm:
        mod.add(keras.layers.BatchNormalization())
    
    mod.add(keras.layers.Dense(layersize,
        kernel_regularizer=keras.regularizers.l2(args.layer_reg),
        bias_regularizer=keras.regularizers.l2(args.layer_reg)
    ))
    
    if args.dropout > 0.0:
        dropout_amount = args.dropout

        #less dropout in first hidden layer
        if i == 0:
            dropout_amount = dropout_amount / 2.0
        mod.add(keras.layers.Dropout(dropout_amount))
    if args.activation == "relu":
        mod.add(keras.layers.Activation("relu"))
    elif args.activation == "leakyrelu":
        mod.add(keras.layers.LeakyReLU(alpha=0.1))
    elif args.activation == "prelu":
        mod.add(keras.layers.PReLU())
    elif args.activation == "elu":
        mod.add(keras.layers.ELU())
    elif args.activation == "tanh":
        mod.add(keras.layers.Activation("tanh"))

mod.add(keras.layers.Dense(1, activation="linear", bias=False))

mod.summary()

#loss function between predicted and true log values
def loss_function_ratio_regression(y_true, y_pred):
    r_loss = 1000.0*losses.mean_squared_error(
        K.exp(K.clip(y_true, -log_r_clip_value, log_r_clip_value)),
        K.exp(K.clip(y_pred, -log_r_clip_value, log_r_clip_value)))
    return r_loss

opt = keras.optimizers.Adam(lr=args.lr, clipnorm=args.clipnorm)
mod.compile(loss=loss_function_ratio_regression, optimizer=opt)

def on_epoch_end(epoch, logs):
    #get layer weight statistics
    for layer in mod.layers:
        weights = layer.get_weights()
        means = []
        stds = []
        if len(weights)>0:
            for weight_mat in weights:
                weight_mat_flat = weight_mat.flatten()
                stds += [np.std(weight_mat_flat)]
                means += [np.mean(weight_mat_flat)]
                #if "dense_4" in layer.name:
                #    print(weight_mat)
        logging.info("epoch_weight {0} {1} means={2} stds={3}".format(epoch, layer.name, means, stds))

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
    logging.info("epoch_end {0} {1} {2}".format(epoch, logs["loss"], logs["val_loss"]))

logging_callback = keras.callbacks.LambdaCallback(
    on_epoch_end=on_epoch_end
)

callbacks = []
if args.do_tensorboard:
    tb = keras.callbacks.TensorBoard(log_dir='./{0}/tb'.format(name), histogram_freq=1, write_grads=True, batch_size=args.batch_size)
    callbacks += [tb]
es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=args.earlystop, verbose=0, mode='auto')
callbacks += [es, logging_callback]
ret = mod.fit(X_train, y_train, sample_weight=w_train, batch_size=args.batch_size, validation_data=(X_test, y_test, w_test), epochs=args.epochs, callbacks=callbacks, verbose=args.verbosity)

plt.figure()
plt.plot(ret.history["loss"][5:])
plt.ylim(0,60)
plt.savefig("{0}/loss_train.pdf".format(name))

plt.figure()
plt.plot(ret.history["val_loss"][5:])
plt.ylim(0,60)
plt.savefig("{0}/loss_test.pdf".format(name))

plt.figure()
plt.plot(ret.history["loss"][5:])
plt.plot(ret.history["val_loss"][5:])
plt.ylim(0,60)
plt.savefig("{0}/loss.pdf".format(name))

import matplotlib.pyplot as plt

y_pred_train = mod.predict(X_train[:50000], batch_size=args.batch_size)
y_pred_test = mod.predict(X_test[:50000], batch_size=args.batch_size)

plt.figure()
plt.scatter(y_train[:10000], y_pred_train[:10000], marker=".", alpha=0.2)
plt.xlabel("true")
plt.ylabel("pred")
plt.savefig("{0}/train.pdf".format(name))

plt.figure()
plt.scatter(y_test[:10000], y_pred_test[:10000], marker=".", alpha=0.2)
plt.xlabel("true")
plt.ylabel("pred")
plt.savefig("{0}/test.pdf".format(name))

