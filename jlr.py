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

batch_size = 1000
log_r_clip_value = 10.0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", type=str,
    default="jlr_data.npz",
    help="Input file",
)
parser.add_argument(
    "--layers", type=int,
    default=2, action="store",
    help="Number of intermediate layers"
)
parser.add_argument(
    "--layersize", type=int,
    default=256, action="store",
    help="Layer size"
)
parser.add_argument(
    "--activation", type=str,
    default="relu", action="store",
    choices=["relu", "leakyrelu", "prelu", "tanh", "elu"],
    help="Activation function"
)
parser.add_argument(
    "--dropout", type=float,
    default=0.0, action="store",
    help="Amount of droupout"
)
parser.add_argument(
    "--lr", type=float,
    default=0.001, action="store",
    help="Learning rate"
)
parser.add_argument(
    "--epochs", type=int,
    default=200, action="store",
    help="Number of epochs"
)
parser.add_argument(
    "--earlystop", type=int,
    default=20, action="store",
    help="Early stopping epochs"
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
    help="Reweight target distribution"
)
parser.add_argument(
    "--do_logtarget",
    action="store_true",
    help="Take log of target"
)
parser.add_argument(
    "--do_varplots",
    action="store_true",
    help="Do plots of input variables"
)

parser.add_argument(
    "--target",
    action="store",
    default="jlr",
    choices=["jlr", "fake"],
    type=str,
    help="Target type"
)

args = parser.parse_args()
name = "tr_l{0}x{1}_d{2:.2f}_{3}_lr{4:.5f}_bn{5}_dn{6}_w{7}_{8}".format(args.layers, args.layersize, args.dropout, args.activation, args.lr, int(args.batchnorm), int(args.do_norm), int(args.do_weight), args.target)
os.makedirs(name)
logging.basicConfig(
    format='%(asctime)s %(name)s %(message)s',
    filename="{0}/log.log".format(name),
    level=logging.INFO,
    filemode="w"
)
print("name " + name)

inf = open(args.input, "rb")
data = np.load(inf)
X = data["X"]
logging.info("X={0}".format(X[:5]))

if args.target == "jlr":
    y = data["y"][:, -1]
elif args.target == "fake":
    y = X[:, 0] + X[:, 4] + X[:, 8] + X[:, 12]
if args.do_logtarget:
    y = np.log(y)
logging.info("y={0}".format(y[:5]))

cut = np.isfinite(y)
logging.info("applying cut to be finite, passed {0}/{1}".format(np.sum(cut), y.shape[0]))
X = X[cut]
y = y[cut]

logging.info("shapes {0} {1}".format(X.shape, y.shape))
ybins = np.linspace(np.mean(y) - 6*np.std(y), np.mean(y) + 6*np.std(y), 100)
c, b = np.histogram(y, bins=ybins)
ib = np.searchsorted(b, y)

w = np.ones(X.shape[0])
if args.do_weight:
    w = np.array([X.shape[0]/c[_ib] if _ib < c.shape[0] else 0.0 for _ib in ib])
    w[np.isinf(w)] = 0.0
    w[np.isnan(w)] = 0.0

# In[115]:
if args.do_norm:
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    logging.info("means={0}".format(means)) 
    logging.info("stds={0}".format(stds)) 
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - means[i])/stds[i]
    
    mean, std = np.mean(y), np.std(y)
    y  = (y-mean)/std

if args.do_varplots:
    for ix in range(X.shape[1]):
        plt.figure()
        plt.hist(X[:, ix], bins=100)
        plt.savefig("{0}/src_{1}.pdf".format(name, ix), weights=w)
        
        plt.figure()
        plt.hexbin(X[:, ix], y[:], bins=100, norm=LogNorm(1, X.shape[0]))
        plt.savefig("{0}/src_tgt_{1}.pdf".format(name, ix), weights=w)
    
plt.figure()
plt.hist(y, bins=ybins)
plt.savefig("{0}/target_unw.pdf".format(name))

ntrain = int(0.8*X.shape[0])
X_train = X[:ntrain]
y_train = y[:ntrain]
w_train = w[:ntrain]

X_test = X[ntrain:]
y_test = y[ntrain:]
w_test = w[ntrain:]

plt.figure()
plt.hist(y_train, bins=ybins, weights=w_train)
plt.hist(y_test, bins=ybins, weights=w_test)
plt.savefig("{0}/target.pdf".format(name))

mod = keras.models.Sequential()
mod.add(keras.layers.InputLayer(input_shape=(X.shape[1], )))

for i in range(args.layers):
    if args.batchnorm:
        mod.add(keras.layers.BatchNormalization())
    mod.add(keras.layers.Dense(args.layersize))
    if args.dropout > 0.0:
        mod.add(keras.layers.Dropout(args.dropout))
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
    
#mod.add(keras.layers.Dense(512))
#if args.activation == "relu":
#    mod.add(keras.layers.Activation("relu"))
#elif args.activation == "leakyrelu":
#    mod.add(keras.layers.LeakyReLU(alpha=0.1))
#elif args.activation == "prelu":
#    mod.add(keras.layers.PReLU())
#elif args.activation == "elu":
#    mod.add(keras.layers.ELU())
#elif args.activation == "tanh":
#    mod.add(keras.layers.Activation("tanh"))
mod.add(keras.layers.Dense(1, activation="linear"))
mod.add(keras.layers.Lambda(lambda x,log_r_clip_value=log_r_clip_value: K.clip(x, -log_r_clip_value, +log_r_clip_value)))

mod.summary()

def loss_function_ratio_regression(y_true, y_pred):
    r_loss = losses.mean_squared_error(
        K.exp(K.clip(y_true, -log_r_clip_value, log_r_clip_value)),
        K.exp(K.clip(y_pred, -log_r_clip_value, log_r_clip_value)))
    return r_loss

opt = keras.optimizers.Adam(lr=args.lr)
mod.compile(loss=loss_function_ratio_regression, optimizer=opt)


# In[ ]:

import json
logging_callback = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: logging.info("epoch_end {0} {1} {2}".format(epoch, logs["loss"], logs["val_loss"])) 
)

tb = keras.callbacks.TensorBoard(log_dir='./{0}/tb'.format(name), histogram_freq=10, write_grads=True, batch_size=batch_size)
es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=args.earlystop, verbose=0, mode='auto')
ret = mod.fit(X_train, y_train, sample_weight=w_train, batch_size=batch_size, validation_data=(X_test, y_test, w_test), epochs=args.epochs, callbacks=[es, logging_callback, tb], verbose=1)

plt.figure()
plt.plot(ret.history["loss"][5:])
plt.savefig("{0}/loss_train.pdf".format(name))

plt.figure()
plt.plot(ret.history["val_loss"][5:])
plt.savefig("{0}/loss_test.pdf".format(name))

plt.figure()
plt.plot(ret.history["loss"][5:])
plt.plot(ret.history["val_loss"][5:])
plt.savefig("{0}/loss.pdf".format(name))

import matplotlib.pyplot as plt

y_pred_train = mod.predict(X_train[:50000], batch_size=batch_size)
y_pred_test = mod.predict(X_test[:50000], batch_size=batch_size)

plt.figure()
plt.scatter(y_train[:10000], y_pred_train[:10000], marker=".", alpha=0.2)
plt.xlabel("true {0}".format(args.target))
plt.ylabel("pred {0}".format(args.target))
plt.savefig("{0}/train.pdf".format(name))

plt.figure()
plt.scatter(y_test[:10000], y_pred_test[:10000], marker=".", alpha=0.2)
plt.xlabel("true {0}".format(args.target))
plt.ylabel("pred {0}".format(args.target))
plt.savefig("{0}/test.pdf".format(name))

