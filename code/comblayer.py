class TrainableCombinationLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CombinationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #weight is trained
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.output_dim, input_shape[1]),
            initializer='uniform',
            trainable=True,
            #constraint=keras.constraints.UnitNorm(axis=1)
        )
        super(CombinationLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim, input_shape[0][2])
    
    def call(self, x):
        prod = K.batch_dot(self.kernel, x, axes=[2,1])
        prod = tf.where(tf.is_nan(prod), tf.zeros_like(prod), prod)
        return prod


class CombinationLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CombinationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CombinationLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim, input_shape[0][2])
    
    def call(self, inp):
        x, kernel = inp
        prod = K.batch_dot(kernel, x, axes=[2,1])
        prod = tf.where(tf.is_nan(prod), tf.zeros_like(prod), prod)
        return prod

def minkowski_init(shape, dtype=None):
    return K.cast(tf.diag([1,1,1,-1]), dtype=K.floatx())


class InnerProductLayer(Layer):
    def __init__(self, **kwargs):
        super(BilinearLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(4, 4),
            initializer=minkowski_init,
            trainable=False,
            #constraint=keras.constraints.UnitNorm(axis=1),
        )
        super(BilinearLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][1], input_shape[0][1])
    
    def matprod(self, x1, x2):
        x1t = tf.transpose(x1,[0,2,1])
        a = tf.transpose(K.dot(self.kernel, x1t), [1,0,2])
        prod = K.batch_dot(x2, a, axes=[2,1])
        return prod
    
    def norms(self, x):
        p = self.matprod(x, x)
        return tf.matrix_diag_part(p)
    
    def call(self, xs):
        x1, x2 = xs
        #print("norm", x1n.shape)
        #x1n = K.repeat(self.norms(x1), x1.shape[1])
        #x2n = K.repeat(self.norms(x2), x1.shape[1])
        prod = self.matprod(x1, x2)
        #prod = tf.divide(prod, x1n)
        #prod = tf.divide(prod, x2n)
        #prod = tf.where(tf.is_nan(prod), tf.zeros_like(prod), prod)
        return prod
