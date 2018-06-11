import numpy as np

from keras.layers import Input, Dense, Add, Multiply
from keras.layers import Reshape, UpSampling1D, Flatten, Concatenate, Cropping1D, Conv1D
from keras.layers import Activation, LeakyReLU, PReLU, Lambda, Merge
from keras.layers import BatchNormalization, Dropout, GaussianNoise
from keras.models import Model, Sequential
from keras.layers import SimpleRNN
from keras.layers import Layer
from keras.constraints import non_neg

import keras.optimizers

from keras.regularizers import l1,l2

from keras import backend as K

from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, ReduceLROnPlateau

from sklearn.base import BaseEstimator

import itertools

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
def get_block(L,name,do_bn0,batch_norm,noise,use_bias,dropout,layers,activations,core=Dense,core_name="dense"):
    if do_bn0:
        L = BatchNormalization(name="%s_bn0" % name)(L)

    if type(activations) != list:
        activations = [activations]*len(layers)
    if type(batch_norm) != list:
        batch_norm = [batch_norm]*len(layers)
    if type(dropout) != list:
        dropout = [dropout]*len(layers)
    if type(noise) != list:
        noise = [noise]*len(layers)
        
    for ilayer,(isize,iact,ibn,ino,ido) in enumerate(zip(layers,activations,batch_norm,noise,dropout)):
        il = ilayer + 1
        ibias = (not ibn) and use_bias
        L = core(isize,use_bias=ibias,name="%s_%s%d" % (name,core_name,il))(L)
        if ibn:
            L = BatchNormalization(name="%s_bn%d" % (name,il))(L)
        if ino is not None:
            L = GaussianNoise(ino,name="%s_noi%d" % (name,il))(L)
        if ido is not None:
            L = Dropout(ido, name="%s_do%d" % (name,il))(L)
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

def get_list(inp,typ):    
    if type(inp) == str:
        return [ typ(x) for x in inp.split(",") ]
    return inp

def get_dict(inp):
    if type(inp) == str:
        if not inp.startswith("{"):
            inp = "{" + inp + "}"
        return json.loads(inp)
    return inp
    
def get_prop(inp,nl,typ):
    inp = get_list(inp,typ)
    if type(inp) != list:
        inp = [inp]
    if len(inp) < nl:
        if nl % len(inp) == 0:
            return inp*(nl//len(inp))
        return inp + [inp[-1]]*(nl-len(inp))
    return inp
        

# --------------------------------------------------------------------------------------------------
class CMBRegression(BaseEstimator):

    def __init__(self,name,
                 jets_shape,
                 leps_shape,
                 met_shape,
                 output_shape=None,
                 output_activation=None,
                 
                 jets_dropout=None,
                 
                 dijet_layers=[64,32,16],
                 dijet_dropout=None,
                 dijet_activations="relu",
                 dijet_noise=None,
                 
                 trijet_layers=[128,64,32],
                 trijet_dropout=None,
                 trijet_activations="relu",
                 trijet_noise=None,

                 do_rnn=False,
                 dijet_rnn_units=None,
                 trijet_rnn_units=None,
                 
                 do_attention=True,

                 dijet_attention_layers=[8,4],
                 dijet_attention_dropout=None,
                 dijet_attention_activations="relu",
                 dijet_attention_noise=None,
                 
                 trijet_attention_layers=[8,4],
                 trijet_attention_dropout=None,
                 trijet_attention_activations="relu",
                 trijet_attention_noise=None,
                 
                 attention_layers = [128,64],
                 attention_dropout=None,
                 attention_activations="relu",
                 attention_noise=None,

                 fc_layers = [128,64],
                 fc_dropout=None,
                 fc_activations="relu",
                 fc_noise=None,

                 optimizer="Adam", optimizer_params=dict(lr=1.e-3), # mse: 1e-3/5e-4
                 loss="mse",
                 loss_params=dict(),# dict(reg_sigma=3.e-2),
                 monitor_dir=".",
                 save_best_only=True,
                 valid_frac=None,
    ):
        self.name = name
        self.leps_shape = leps_shape 
        self.jets_shape = jets_shape
        self.met_shape = met_shape 
        self.output_shape = output_shape
        self.output_activation = output_activation

        self.jets_dropout = jets_dropout
                     
        self.dijet_layers = get_list(dijet_layers,int)
        self.dijet_activations = get_prop(dijet_activations,len(self.dijet_layers),str)
        self.dijet_dropout = get_prop(dijet_dropout,len(self.dijet_layers),float)
        self.dijet_noise = get_prop(dijet_noise,len(self.dijet_layers),float)

        self.trijet_layers = get_list(trijet_layers,int)
        self.trijet_activations = get_prop(trijet_activations,len(self.trijet_layers),str)
        self.trijet_dropout = get_prop(trijet_dropout,len(self.trijet_layers),float)
        self.trijet_noise = get_prop(trijet_noise,len(self.trijet_layers),float)

        self.do_rnn = do_rnn
        print(dijet_rnn_units,trijet_rnn_units)
        self.dijet_rnn_units = get_list(dijet_rnn_units,int)
        self.trijet_rnn_units = get_list(trijet_rnn_units,int)
        
        self.do_attention = do_attention
        
        self.dijet_attention_layers = get_list(dijet_attention_layers,int)
        self.dijet_attention_activations = get_prop(dijet_attention_activations,len(self.dijet_attention_layers),str)
        self.dijet_attention_dropout = get_prop(dijet_attention_dropout,len(self.dijet_attention_layers),float)
        self.dijet_attention_noise = get_prop(dijet_attention_noise,len(self.dijet_attention_layers),float)

        self.trijet_attention_layers = get_list(trijet_attention_layers,int)
        self.trijet_attention_activations = get_prop(trijet_attention_activations,len(self.trijet_attention_layers),str)
        self.trijet_attention_dropout = get_prop(trijet_attention_dropout,len(self.trijet_attention_layers),float)
        self.trijet_attention_noise = get_prop(trijet_attention_noise,len(self.trijet_attention_layers),float)
        
        self.attention_layers = get_list(attention_layers,int)
        self.attention_activations = get_prop(attention_activations,len(self.attention_layers),str)
        self.attention_dropout = get_prop(attention_dropout,len(self.attention_layers),float)
        self.attention_noise = get_prop(attention_noise,len(self.attention_layers),float)

        self.fc_layers = get_list(fc_layers,int)
        self.fc_activations = get_prop(fc_activations,len(self.fc_layers),str)
        self.fc_dropout = get_prop(fc_dropout,len(self.fc_layers),float)
        self.fc_noise = get_prop(fc_noise,len(self.fc_layers),float)

        self.optimizer = optimizer
        self.optimizer_params = get_dict(optimizer_params)
        
        self.loss = loss
        self.loss_params = get_dict(loss_params)
        
        ### 
        self.valid_frac = valid_frac
        self.save_best_only = save_best_only
        self.monitor_dir = monitor_dir
        
        self.model = None

        super(CMBRegression,self).__init__()
        
    # ----------------------------------------------------------------------------------------------
    def __call__(self,docompile=False):
        
        if hasattr(losses,self.loss):
            loss = getattr(losses,self.loss)
            if isinstance(loss,object):
                loss = loss(**self.loss_params)
        else:
            loss = self.loss

        output_shape = self.output_shape
        if output_shape is None:
            output_shape = (getattr(loss,"n_params",1),)
            
        if self.model is None:
            
            ## input layers
            inputs = []
            njet, nlep = None, None
            if self.jets_shape is not None:
                input_jet = Input(shape=self.jets_shape,name="%s_jet" % self.name)
                inputs.append(input_jet)
                njet = self.jets_shape[0]
                if self.jets_dropout is not None:
                    input_jet = Dropout(self.jets_dropout,noise_shape=(None,1,self.jets_shape[1]),name="%s_jets_do" % self.name)(input_jet)
            if self.leps_shape is not None:
                input_lep = Input(shape=self.leps_shape,name="%s_lep" % self.name)
                inputs.append(input_lep)
                nlep = self.leps_shape[0]

            ## make dijet and trijet combinations
            ## input
            ## [ jet0,
            ##   jet1, 
            ##   ...
            ##   jetN ]
            ## output 
            ## [ [ jet0 jet1 ],
            ##   [ jet0 jet2 ],
            ##   [ jet0 jet3 ],
            ##   ....
            ##   [ jetN-1, jetN]
            ## ]
            dijets = Lambda(lambda x: K.concatenate( [K.concatenate( [ x[:,ijet0:ijet0+1,:],x[:,ijet1:ijet1+1,:] ] ) 
                                                      for ijet0,ijet1 in itertools.combinations(reversed(range(njet)),2) ], axis=1),
                            name="%s_dijets" % (self.name) )(input_jet)
            
            trijets = Lambda(lambda x: K.concatenate( [K.concatenate( [ x[:,ijet0:ijet0+1,:],x[:,ijet1:ijet1+1,:],x[:,ijet2:ijet2+1,:] ] ) 
                                                       for ijet0,ijet1,ijet2 in itertools.combinations(reversed(range(njet)),3)], axis=1),
                             name="%s_trijets" % (self.name) )(input_jet)
            
            ## 1x1 convolutions for dijet and trijet
            ndijet = int(dijets.get_shape()[1])
            ntrijet = int(trijets.get_shape()[1])
            
            dijets = get_block(dijets,self.name+"_dijets",do_bn0=True,batch_norm=True,
                               noise=self.dijet_noise,use_bias=True,dropout=self.dijet_dropout,
                               layers=self.dijet_layers,
                               activations=self.dijet_activations,
                               core=lambda x,**k: Conv1D(x,kernel_size=1,**k),core_name="conv1x1")

            trijets = get_block(trijets,self.name+"_trijets",do_bn0=True,batch_norm=True,
                                noise=self.trijet_noise,use_bias=True,dropout=self.trijet_dropout,
                                layers=self.trijet_layers,
                                activations=self.trijet_activations,
                                core=lambda x,**k: Conv1D(x,kernel_size=1,**k),core_name="conv1x1")

            ## recursive layers
            if self.do_rnn is not None and self.do_rnn > 0:
                dijets = SimpleRNN(self.dijet_rnn_units[0],return_sequences=True,name="%s_dijets_rnn_prea" % self.name)(dijets)
                trijets = SimpleRNN(self.trijet_rnn_units[0],return_sequences=True,name="%s_trijets_rnn_prea" % self.name)(trijets)
                
            ## attention network
            ## first reduce dijet and trijet dimensionality 
            if self.do_attention:
                dijetsA = get_block(dijets,self.name+"_dijetsA",do_bn0=False,batch_norm=True,
                                    noise=self.dijet_attention_noise,use_bias=True,dropout=self.dijet_attention_dropout,
                                    layers=self.dijet_attention_layers,
                                    activations=self.dijet_attention_activations,
                                    core=lambda x,**k: Conv1D(x,kernel_size=1,**k),core_name="conv1x1")
                
                trijetsA = get_block(trijets,self.name+"_trijetsA",do_bn0=False,batch_norm=True,
                                     noise=self.trijet_attention_noise,use_bias=True,dropout=self.trijet_attention_dropout,
                                     layers=self.trijet_attention_layers,
                                     activations=self.trijet_attention_activations,
                                     core=lambda x,**k: Conv1D(x,kernel_size=1,**k),core_name="conv1x1")
                
                ## then concatenate and add fc layers
                att = Concatenate(axis=1,name="%s_A_inp"%self.name)([dijetsA,trijetsA])
                att = Flatten(name="%s_A_flt"%self.name)(att)
                att = get_block(att,self.name+"_A",do_bn0=False,batch_norm=True,
                                noise=self.attention_noise,use_bias=True,dropout=self.attention_dropout,
                                layers=self.attention_layers,
                                activations=self.attention_activations,
                            )

                ## final attention softmax            
                dijetsA = Dense(ndijet, activation="softmax",name="%s_dijetaA" % self.name)(att)
                trijetsA = Dense(ntrijet, activation="softmax",name="%s_trijetaA" % self.name)(att)
                dijetsA = Reshape((-1,1),name="%s_dijetaA_rshp" % self.name)(dijetsA)
                trijetsA = Reshape((-1,1),name="%s_trijetaA_rshp" % self.name)(trijetsA)

                ## multiply dijet and trijets by attention
                dijets = Multiply(name="%s_dijets_prod"%self.name)([dijets,dijetsA])
                trijets = Multiply(name="%s_trijets_prod"%self.name)([trijets,trijetsA])

                ## recursive layers
                if self.do_rnn is not None and self.do_rnn > 1:
                    dijets = SimpleRNN(self.dijet_rnn_units[1],return_sequences=True,name="%s_dijets_posta" % self.name)(dijets)
                    trijets = SimpleRNN(self.trijet_rnn_units[1],return_sequences=True,name="%s_trijets_posta" % self.name)(trijets)
            
            ## flatten combinations
            dijets = Flatten(name="%s_dijets_flt"%self.name)(dijets)
            trijets = Flatten(name="%s_trijets_flt"%self.name)(trijets)
                        
            ## final fc layers
            fc = Concatenate(axis=1,name="%s_fc_inp"%self.name)([dijets,trijets])
            fc = get_block(fc,self.name+"_fc",do_bn0=False,batch_norm=True,
                                noise=self.fc_noise,use_bias=True,dropout=self.fc_dropout,
                                layers=self.fc_layers,
                                activations=self.fc_activations,
                                )
            output = Dense(1,activation=None,
                           use_bias=True,
                           name="%s_out" % self.name)(fc)
            
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
        return [csv,checkpoint] ##,reducelr]
    
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
    
