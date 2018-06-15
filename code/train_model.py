#!/usr/bin/env python

import numpy as np

import os
import json

from pprint import pprint

from sklearn.model_selection import train_test_split, KFold

from optparse import OptionParser, make_option, OptionGroup

##
default_features = []


## command_line options
parser = OptionParser(option_list=[
    make_option("--inp-dir",type='string',dest="inp_dir",default=os.environ['SCRATCH']+'/tth/delphes_had'),
    make_option("--out-dir",type='string',dest="out_dir",default='.'),

    make_option("--normalize-target",action="store_true",dest="normalize_target",default=True),
    make_option("--no-normalize-target",action="store_false",dest="normalize_target"),
    make_option("--exp-target",action="store_true",dest="exp_target",default=False),
    make_option("--no-exp-target",action="store_false",dest="exp_target"),

    make_option("--loss",type='string',dest="loss",default="mse"),
    make_option("--loss-params",type='string',dest="loss_params",default=dict()),
    
    make_option("--test-frac",type='float',dest='test_frac',default=0.1),
    make_option("--valid-frac",type='float',dest='valid_frac',default=0.1),
    
    make_option("--batch-size",type='int',dest='batch_size',default=512),
    make_option("--epochs",type='int',dest='epochs',default=100),
    
    make_option("--hparams",type='string',dest='hparams',default=None),
    make_option("--seed",type='int',dest='seed',default=873462),
    
    make_option("--architecture",type='string',dest='architecture',default="cmb"),
    make_option("--features",type='string',dest='features',default="jets"),

    make_option("--output-activation",type='string',dest='output_activation',default=None),
    make_option("--optimizer-params",type="string",dest="optimizer_params=",default=dict(lr=1.e-6,decay=2e-5)),#default=dict(lr=1.e-5,decay=2e-5)),
    
    make_option("--kfolds",type="int",dest="kfolds",default=None),
    make_option("--ifold",type="int",dest="ifold",default=None),
    make_option("--fold-seed",type='int',dest='fold_seed',default=123767),
])

ffwd_opts = [    ## FFWD network 
    make_option("--dropout",type='float',dest='dropout',default=0.5),
    make_option("--batch-norm",dest='batch_norm',action="store_true",default=True),
    make_option("--no-batch-norm",dest='batch_norm',action="store_false"),
    make_option("--layers",type='string',dest='layers',default=[1024]*2+[512,256,128]),# [512]*2+[256,128,64]),# ##default=[2048,1024]+[512,256,128,64] ),
    make_option("--activations",type='string',dest='activations',default="lrelu"),
    
]

cmb_opts = [
    ## CMB network 
    make_option("--jets-dropout",type="float",action="store",dest="jets_dropout",default=None),
    
    make_option("--dijet-layers",type="string",action="store",dest="dijet_layers",default=[64,32,16]),
    make_option("--dijet-dropout",type="string",action="store",dest="dijet_dropout",default=None),
    make_option("--dijet-noise",type="string",action="store",dest="dijet_noise",default=None),
    make_option("--dijet-activations",type="string",action="store",dest="dijet_activations",default="relu"),

    make_option("--lep-layers",type="string",action="store",dest="lep_layers",default=[64,32,16]),
    make_option("--lep-dropout",type="string",action="store",dest="lep_dropout",default=None),
    make_option("--lep-noise",type="string",action="store",dest="lep_noise",default=None),
    make_option("--lep-activations",type="string",action="store",dest="lep_activations",default="relu"),
    
    make_option("--trijet-layers",type="string",action="store",dest="trijet_layers",default=[128,64,32]),
    make_option("--trijet-dropout",type="string",action="store",dest="trijet_dropout",default=None),
    make_option("--trijet-noise",type="string",action="store",dest="trijet_noise",default=None),
    make_option("--trijet-activations",type="string",action="store",dest="trijet_activations",default="relu"),

    make_option("--do-rnn",action="store",type="int",dest="do_rnn",default=False),
    make_option("--dijet-rnn-units",action="store",type="string",dest="dijet_rnn_units",default=None),
    make_option("--lep-rnn-units",action="store",type="string",dest="lep_rnn_units",default=None),
    make_option("--trijet-rnn-units",action="store",type="string",dest="trijet_rnn_units",default=None),
    
    make_option("--do-attention",action="store_true",dest="do_attention",default=True),
    make_option("--no-do-attention",action="store_false",dest="do_attention"),
        
    make_option("--dijet-attention-layers",type="string",action="store",dest="dijet_attention_layers",default=[8,4]),
    make_option("--dijet-attention-dropout",type="string",action="store",dest="dijet_attention_dropout",default=None),
    make_option("--dijet-attention-noise",type="string",action="store",dest="dijet_attention_noise",default=None),
    make_option("--dijet-attention-activations",type="string",action="store",dest="dijet_attention_activations",default="relu"),

    make_option("--lep-attention-layers",type="string",action="store",dest="lep_attention_layers",default=[8,4]),
    make_option("--lep-attention-dropout",type="string",action="store",dest="lep_attention_dropout",default=None),
    make_option("--lep-attention-noise",type="string",action="store",dest="lep_attention_noise",default=None),
    make_option("--lep-attention-activations",type="string",action="store",dest="lep_attention_activations",default="relu"),
    
    make_option("--trijet-attention-layers",type="string",action="store",dest="trijet_attention_layers",default=[8,4]),
    make_option("--trijet-attention-dropout",type="string",action="store",dest="trijet_attention_dropout",default=None),
    make_option("--trijet-attention-noise",type="string",action="store",dest="trijet_attention_noise",default=None),
    make_option("--trijet-attention-activations",type="string",action="store",dest="trijet_attention_activations",default="relu"),
    
    make_option("--attention-layers",type="string",action="store",dest="attention_layers",default=[128,64]),
    make_option("--attention-noise",type="string",action="store",dest="attention_noise",default=None),
    make_option("--attention-activations",type="string",action="store",dest="attention_activations",default="relu"),
    make_option("--attention-dropout",type="string",action="store",dest="attention_dropout",default=None),
    
    make_option("--fc-layers",type="string",action="store",dest="fc_layers",default=[128,64]),
    make_option("--fc-dropout",type="string",action="store",dest="fc_dropout",default=None),
    make_option("--fc-noise",type="string",action="store",dest="fc_noise",default=None),
    make_option("--fc-activations",type="string",action="store",dest="fc_activations",default="relu"),
]

for grp,name in zip([ffwd_opts,cmb_opts],["ffwd network options","cmb network options"]):
    group = OptionGroup(parser,name)
    for opt in grp: group.add_option(opt)
    parser.add_option_group(group)
    

## parse options
(options, args) = parser.parse_args()

#convert layer string to list of layer sizes
if isinstance(options.layers, str):
    options.layers = [int(x) for x in options.layers.split(",")]


import pyjlr.cmb as cmb
import pyjlr.ffwd as ffwd

if options.kfolds is not None:
    if not os.path.exists(options.out_dir):
        os.mkdir(options.out_dir)
    options.out_dir += "/fold_%d_%d" % (options.kfolds, options.ifold)

hparams = {}
if options.hparams is not None:
    with open(options.hparams) as hf:
        hparams = json.loads(hf.read())        

print("input features: " + options.features )
allX = { feat : np.load(options.inp_dir+'/%s.npy' % feat) for feat in options.features.split(",") }
y = np.load(options.inp_dir+'/target.npy')
if options.exp_target:
    y = np.exp(-y)

y_mean = np.median(y)
y_std = y.std()
print("target mean,std", y_mean,y_std)

# normalize target
if options.loss == "binary_crossentropy":
    if options.normalize_target:
        y = y.mean() / ( y.mean() + y )
    else:
        y = 1. / ( 1. + y )
    if options.output_activation is None:
        options.output_activation = "sigmoid"

elif options.normalize_target:
    y -= y_mean
    y /= y_std
    

## determine architecture type and associated inputs
if options.architecture == "cmb":
    regressor = cmb.CMBRegression
    init_args = []
    X = []
    for feat in "jets","leps","met":
        if feat in allX.keys():
            X.append(allX.pop(feat))
            init_args.append(X[-1].shape[1:])
        else:
            init_args.append(None)
    assert( len(allX.values()) == 0 )

elif options.architecture == "ffwd":
    regressor = ffwd.FFWDRegression
    assert( len(allX.values()) == 1 )
    X = list(allX.values())
    init_args = [X[0].shape[1:]]
else:
    assert( 0 )

# sort out model parameters
def get_kwargs(fn,**kwargs):
    params = set(fn.__code__.co_varnames[:fn.__code__.co_argcount]+tuple(kwargs.keys()))
    for par in params:
        if hasattr(options,par):
            kwargs[par] = getattr(options,par)
        if par in hparams:
            kwargs[par] = hparams.pop(par)
    return kwargs

init_kwargs = get_kwargs(regressor.__init__,monitor_dir=options.out_dir)
fit_kwargs = get_kwargs(regressor.fit,batch_size=options.batch_size,epochs=options.epochs)

## instantiate regressor
reg = regressor(options.architecture,*init_args,**init_kwargs)

pprint(reg.get_params())
pprint(fit_kwargs)

# get associated model
model = reg()

print(model.summary())

# prepare output folder and store input parameters
if not os.path.exists(options.out_dir):
    os.mkdir(options.out_dir)

store = dict( y_mean=float(y_mean), y_std=float(y_std),
              model_params = reg.get_params(),
              fit_kwargs = fit_kwargs,
              options = options.__dict__
              )
with open(options.out_dir+'/config.json','w+') as fo:
    fo.write( json.dumps( store, indent=4 ) )
    fo.close()


# split data
# keep test sample aside
split_inds = np.arange(0, y.shape[0]) #to keep track of which event was splitted to which set
split = train_test_split(split_inds, *X,y,test_size=options.test_frac,random_state=options.seed)
split = [ split[ix] for ix in range(0,len(split),2) ]
for x in split:
    print(x.shape)

# split train/validation sample
if options.kfolds is None:
    split = train_test_split(*split,test_size=options.valid_frac,random_state=options.fold_seed)
else:
    kf = KFold(n_splits=int(1./options.valid_frac),shuffle=True,random_state=options.fold_seed) 
    folds = iter(kf.split(split[-1]))
    for fold in range(options.ifold-1): next(folds)
    train_idx, valid_idx = next(folds)
    isplit = []
    for x in split:
        isplit += [ x[train_idx],x[valid_idx]  ]
    split = isplit
    
# collect X and y
if len(split) == 6:
    inds_train, inds_valid, X_train, X_valid, y_train, y_valid = split
else:
    inds_train, inds_valid = split[:2]
    y_train, y_valid = split[-2:]
    X_split = split[2:-2]
    X_train = []
    X_valid = []
    while len(X_split) > 0:
        X_train.append( X_split.pop(0) )
        X_valid.append( X_split.pop(0) )

#save the indices of the events used for training and validation
np.save(options.out_dir+"/idx_valid", inds_valid)
np.save(options.out_dir+"/idx_train", inds_train)

#save options as json
of = open(options.out_dir + "/options.json", "w")
of.write(json.dumps(options.__dict__, indent=2))
of.close()

# ok we can start training
reg.fit(X_train,y_train,
        validation_data=(X_valid,y_valid),
        **fit_kwargs)

#save prediction values
pred = reg.predict(X)
print("pred shape {0}, saving to {1}".format(pred.shape, options.out_dir + "/pred.npy"))
np.save(options.out_dir+"/pred", pred)