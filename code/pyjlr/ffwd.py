import numpy as np

from keras.layers import Input, Dense, Add, Multiply
from keras.layers import Reshape, UpSampling1D, Flatten, concatenate, Cropping1D
from keras.layers import Activation, LeakyReLU, PReLU, Lambda
from keras.layers import BatchNormalization, Dropout, GaussianNoise
from keras.models import Model, Sequential
from keras.layers import Layer
from keras.constraints import non_neg

import keras.optimizers

from keras.regularizers import l1,l2

from keras import backend as K

from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, ReduceLROnPlateau

from sklearn.base import BaseEstimator

from . import losses 

# --------------------------------------------------------------------------------------------------
class ConstOffsetLayer(Layer):

    def __init__(self, values, **kwargs):
        self.values = values
        super(ConstOffsetLayer, self).__init__(**kwargs)

    def get_config(self):
        return { 'values' : self.values }
        
    def call(self, x):
        return x+K.constant(self.values)

# --------------------------------------------------------------------------------------------------
def get_block(L,name,do_bn0,batch_norm,noise,use_bias,dropout,layers,activations,core=Dense):
    if do_bn0:
        L = BatchNormalization(name="%s_bn0" % name)(L)

    for ilayer,(isize,iact) in enumerate(zip(layers,activations)):
        il = ilayer + 1
        L = core(isize,use_bias=use_bias,name="%s_dense%d" % (name,il))(L)
        ## L = Dense(isize,use_bias=use_bias,name="%s_dense%d" % (name,il))(L)
        if batch_norm:
            L = BatchNormalization(name="%s_bn%d" % (name,il))(L)
        if noise is not None:
            L = GaussianNoise(noise,name="%s_noi%d" % (name,il))(L)
        if dropout is not None:
            if il == 0:
                L = Dropout(dropout*0.5, name="%s_do%d" % (name,il))(L)
            else:
                L = Dropout(dropout, name="%s_do%d" % (name,il))(L)
        if iact is None:
            pass
        elif iact == "lrelu" or "lrelu_" in iact:
            nslope = 0.2
            if "_" in iact:
                nslope = float(iact.rsplit("_",1))
            L = LeakyReLU(nslope,name="%s_act%d_%s" % (name,il,iact))(L)
        elif iact == "prelu":
            L = PReLU(name="%s_act%d_%s" % (name,il,iact))(L)
        else:
            L = Activation(iact,name="%s_act%d_%s" % (name,il,iact))(L)
    return L
    
# --------------------------------------------------------------------------------------------------
class FFWDRegression(BaseEstimator):

    def __init__(self,name,input_shape,output_shape=None,
                 non_neg=False,
                 dropout=0.2, # 0.5 0.2
                 batch_norm=True,activations="lrelu",
                 layers=[1024]*5+[512,256,128], # 1024 / 2048
                 output_activation=None,
                 do_bn0=True,
                 noise=None,
                 optimizer="Adam", optimizer_params=dict(lr=1.e-3), # mse: 1e-3/5e-4
                 loss="RegularizedGaussNll",
                 loss_params=dict(),# dict(reg_sigma=3.e-2),
                 monitor_dir=".",
                 save_best_only=True,
                 valid_frac=None,
    ):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_activation = output_activation

        self.non_neg = non_neg
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.use_bias = not batch_norm
        self.layers = layers
        if type(activations) == str:
            self.activations = [activations]*len(layers)
        elif len(activations) < len(layers):
            rat = len(layers) // len(activations)
            remind = len(layers) % len(activations)
            self.activations = activations*rat
            if remind > 0:
                self.activations += activations[:remind]
        else:
            self.activations = activations
        self.do_bn0 = do_bn0
        self.noise = noise
        
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.loss = loss
        self.loss_params = loss_params

        self.valid_frac = valid_frac
        self.save_best_only = save_best_only
        self.monitor_dir = monitor_dir
        
        self.model = None

        super(FFWDRegression,self).__init__()
        
    # ----------------------------------------------------------------------------------------------
    def __call__(self,docompile=False):
        
        if hasattr(losses,self.loss):
            loss = getattr(losses,self.loss)
            ## print(loss,isinstance(loss,object))
            if isinstance(loss,object):
                loss = loss(**self.loss_params)
            ## print(loss)
        else:
            loss = self.loss

        output_shape = self.output_shape
        if output_shape is None:
            output_shape = (getattr(loss,"n_params",1),)
            
        if self.model is None:
            inputs = Input(shape=self.input_shape,name="%s_inp" % self.name)
            
            if len(self.input_shape)>1:
                L = Flatten(name="%s_flt" % self.name)(inputs)
            else:
                L = inputs
                
            L = get_block(L,self.name,self.do_bn0,
                         self.batch_norm,self.noise,self.use_bias,self.dropout,
                          self.layers,self.activations)            
            reshape = False
            out_size = output_shape[0]
            if len(output_shape) > 1:
                reshape = True
                for idim in output_shape[1:]:
                    out_size *= output_shape[idim]
            constr = None
            output = Dense(out_size,activation=self.output_activation,
                           use_bias=True,
                           name="%s_out" % self.name)(L)
            if reshape:
                output = Reshape(output_shape,name="%s_rshp" % self.name)(output)

            self.model = Model( inputs=inputs, outputs=output )
            
        if docompile:
            optimizer = getattr(keras.optimizers,self.optimizer)(**self.optimizer_params)

            self.model.compile(optimizer=optimizer,loss=loss,metrics=[losses.mse0,losses.mae0,
                                                                      losses.r2_score0])
        return self.model

    # ----------------------------------------------------------------------------------------------
    def get_callbacks(self,has_valid=False,monitor='loss',save_best_only=True):
        if has_valid:
            monitor = 'val_'+monitor
        monitor_dir = self.monitor_dir
        csv = CSVLogger("%s/metrics.csv" % monitor_dir)
        ## checkpoint = ModelCheckpoint("%s/model-{epoch:02d}.hdf5" % monitor_dir,
        checkpoint = ModelCheckpoint("%s/model.hdf5" % monitor_dir,
                                     monitor=monitor,save_best_only=save_best_only,
                                     save_weights_only=False)
        ## reducelr = ReduceLROnPlateau(patience=5,factor=0.2)
        return [csv,checkpoint] ## ,reducelr]
    
    # ----------------------------------------------------------------------------------------------
    def fit(self,X,y,**kwargs):

        model = self(True)
        
        has_valid = kwargs.get('validation_data',None) is not None
        if not has_valid and self.valid_frac is not None:
            last_train = int( X.shape[0] * (1. - self.valid_frac) )
            X_train = X[:last_train]
            X_valid = X[last_train:]
            y_train = y[:last_train]
            y_valid = y[last_train:]
            kwargs['validation_data'] = (X_valid,y_valid)
            has_valid = True
        else:
            X_train, y_train = X, y
            
        if not 'callbacks' in kwargs:
            save_best_only=kwargs.pop('save_best_only',self.save_best_only)
            kwargs['callbacks'] = self.get_callbacks(has_valid=has_valid,
                                                     save_best_only=save_best_only)
        
        with open(self.monitor_dir+"/model.json", "w") as fi:
            fi.write(model.to_json())

        return model.fit(X_train,y_train,**kwargs)
    
    # ----------------------------------------------------------------------------------------------
    def predict(self,X,p0=True,**kwargs):
        y_pred =  self.model.predict(X,**kwargs)
        if p0:
            return y_pred[:,0]
        else:
            return y_pred
    
    # ----------------------------------------------------------------------------------------------
    def score(self,X,y,**kwargs):
        return -self.model.evaluate(X,y,**kwargs)
    
