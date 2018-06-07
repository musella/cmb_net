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

from jlr_util import build_ibnet, build_densenet, loss_function_ratio_regression, load_data, input_statistics, r2_score, on_epoch_end, loss_function_p4, build_parton_net

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
    default=1, action="store",
    help="Training verbosity"
)
parser.add_argument(
    "--seed", type=int,
    default=1, action="store",
    help="The random seed"
)
parser.add_argument(
    "--batch_size", type=int,
    default=1000, action="store",
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
    choices=["relu", "leakyrelu", "prelu", "tanh", "elu", "selu"],
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
name = "tr_l{layers}x{layersize}_d{dropout:.2f}_{activation}_lr{lr:.7f}_bn{batchnorm}_dn{do_norm}_w{do_weight}_{inp}_{ntrain}_{ntest}_cn{clipnorm:.2f}_reg{layer_reg:.2E}_b{batch_size}_s{seed}".format(
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


X, Xadditional, Xmatch, Xparton, y = load_data(args.input)

#create histogram for target
logging.info("shapes {0} {1}".format(X.shape, y.shape))
ybins = np.linspace(np.mean(y) - 6*np.std(y), np.mean(y) + 6*np.std(y), 100)
c, b = np.histogram(y, bins=ybins)
ib = np.searchsorted(b, y)

#compute sample weights for target to be flat
w = np.ones(X.shape[0])
if args.do_weight:
    w = np.array([c[_ib] if _ib < c.shape[0] else 0.0 for _ib in ib])
    w = 100.0/w
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
    Xadditional_train = Xadditional[:ntrain]
    Xmatch_train = Xmatch[:ntrain]
    Xparton_train = Xparton[:ntrain]
    y_train = y[:ntrain]
    w_train = w[:ntrain]
    
    X_test = X[ntrain:]
    Xparton_test = Xparton[ntrain:]
    Xadditional_test = Xadditional[:ntest]
    Xmatch_test = Xmatch[:ntest]
    y_test = y[ntrain:]
    w_test = w[ntrain:]
else:
    X_train = X[:args.ntrain]
    Xparton_train = Xparton[:args.ntrain]
    Xadditional_train = Xadditional[:args.ntrain]
    Xmatch_train = Xmatch[:args.ntrain]
    y_train = y[:args.ntrain]
    w_train = w[:args.ntrain]
    
    X_test = X[args.ntrain:args.ntrain+args.ntest]
    Xparton_test = Xparton[args.ntrain:args.ntrain+args.ntest]
    Xadditional_test = Xadditional[args.ntrain:args.ntrain+args.ntest]
    Xmatch_test = Xmatch[args.ntrain:args.ntrain+args.ntest]
    y_test = y[args.ntrain:args.ntrain+args.ntest]
    w_test = w[args.ntrain:args.ntrain+args.ntest]

input_statistics(X_train, name, "inputs_train.pdf")
input_statistics(X_test, name, "inputs_test.pdf")

#plot the target
plt.figure()
plt.hist(y_train, bins=ybins, weights=w_train)
plt.hist(y_test, bins=ybins, weights=w_test)
plt.savefig("{0}/target.pdf".format(name))

K.set_learning_phase(True)
mod = build_densenet(X, args.layers, args.dropout, args.layersize, args.batchnorm, args.layer_reg)
#mod = build_ibnet(X, args.layers, args.dropout, args.layersize, args.batchnorm, args.activation, args.layer_reg)
#mod = build_parton_net(X, args.layers, args.dropout, args.layersize, args.batchnorm, args.activation, args.layer_reg)
mod.summary()
opt = keras.optimizers.Adam(lr=args.lr, clipnorm=args.clipnorm)
mod.compile(loss=loss_function_ratio_regression, optimizer=opt, metrics=[r2_score])
#mod.compile(loss={"main_output": loss_function_ratio_regression, "ib_layer": loss_function_p4}, optimizer=opt, metrics=[r2_score], loss_weights={"main_output": 1.0, "ib_layer": 0.1})
#mod.compile(loss=loss_function_p4, optimizer=opt)

logging_callback = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda x,y,mod=mod,X_test=X_test,Xparton_test=Xparton_test: on_epoch_end(X_test, Xparton_test, mod, x, y)
)

callbacks = []
if args.do_tensorboard:
    tb = keras.callbacks.TensorBoard(log_dir='./{0}/tb'.format(name), histogram_freq=0, write_grads=False, batch_size=args.batch_size)
    callbacks += [tb]
es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=args.earlystop, verbose=0, mode='auto')
callbacks += [es, logging_callback]
#ret = mod.fit(X_train, [y_train, Xparton_train], sample_weight=[w_train, w_train], batch_size=args.batch_size, validation_data=(X_test, [y_test, Xparton_test], [w_test, w_test]), epochs=args.epochs, callbacks=callbacks, verbose=args.verbosity)
ret = mod.fit(X_train, Xparton_train, sample_weight=w_train, batch_size=args.batch_size, validation_data=(X_test, Xparton_test, w_test), epochs=args.epochs, callbacks=callbacks, verbose=args.verbosity)
K.set_learning_phase(False)

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

#plt.figure()
#plt.plot(ret.history["main_output_r2_score"][5:])
#plt.plot(ret.history["val_main_output_r2_score"][5:])
#plt.ylim(-5,1)
#plt.savefig("{0}/r2_score.pdf".format(name))

import matplotlib.pyplot as plt

y_pred_train = mod.predict(X_train[:50000], batch_size=args.batch_size)[0]
y_pred_test = mod.predict(X_test[:50000], batch_size=args.batch_size)[0]

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
