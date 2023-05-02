import tensorflow as tf
from copy import deepcopy


########################################################################################


class SELayer(tf.keras.layers.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(channel // reduction, use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(channel, use_bias=False),
            tf.keras.layers.Activation('sigmoid')
        ])

    def call(self, inputs):
        shape = inputs.shape
        y = self.avg_pool(inputs)
        y = tf.reshape(y, [-1, shape[1]])
        y = self.fc(y)
        y = tf.reshape(y, [-1, shape[1], 1])
        return tf.multiply(inputs, y)


class SEBasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(planes, stride)
        self.bn1 = tf.keras.layers.BatchNormalization(planes)
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv1D(planes, 1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def call(self, inputs):
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        out += residual
        out = self.relu(out)

        return out

class GELU(tf.keras.layers.Layer):
    def __init__(self):
        super(GELU, self).__init__()

    def call(self, x):
        x = tf.keras.activations.gelu(x)
        return x
        
        
class MRCNN(tf.keras.Model):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.GELU = tf.keras.layers.Activation('gelu')
        self.features1 = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, kernel_size=50, strides=6, use_bias=False, padding='valid'),
            tf.keras.layers.BatchNormalization(),
            self.GELU,
            tf.keras.layers.MaxPool1D(pool_size=8, strides=2, padding='same'),
            tf.keras.layers.Dropout(drate),

            tf.keras.layers.Conv1D(128, kernel_size=8, strides=1, use_bias=False, padding='same'),
            tf.keras.layers.BatchNormalization(),
            self.GELU,

            tf.keras.layers.Conv1D(128, kernel_size=8, strides=1, use_bias=False, padding='same'),
            tf.keras.layers.BatchNormalization(),
            self.GELU,

            tf.keras.layers.MaxPool1D(pool_size=4, strides=4, padding='same')
        ])

        self.features2 = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, kernel_size=400, strides=50, use_bias=False, padding='valid'),
            tf.keras.layers.BatchNormalization(),
            self.GELU,
            tf.keras.layers.MaxPool1D(pool_size=4, strides=2, padding='same'),
            tf.keras.layers.Dropout(drate),

            tf.keras.layers.Conv1D(128, kernel_size=7, strides=1, use_bias=False, padding='same'),
            tf.keras.layers.BatchNormalization(),
            self.GELU,

            tf.keras.layers.Conv1D(128, kernel_size=7, strides=1, use_bias=False, padding='same'),
            tf.keras.layers.BatchNormalization(),
            self.GELU,

            tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')
        ])
        self.dropout = tf.keras.layers.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)


    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                tf.keras.layers.Conv1D(planes * block.expansion, kernel_size=1, strides=stride, use_bias=False),
                tf.keras.layers.BatchNormalization(),
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return tf.keras.Sequential(layers)

    def call(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = tf.concat([x1, x2], axis=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat

##########################################################################################


def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.shape[-1]
    scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d_k)

    p_attn = tf.nn.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return tf.matmul(p_attn, value), p_attn


class CausalConv1d(tf.keras.layers.Conv1D):
    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rate=1,
                 **kwargs):
        self.__padding = (kernel_size - 1) * dilation_rate
        super(CausalConv1d, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='valid',
            dilation_rate=dilation_rate,
            **kwargs)

    def call(self, inputs):
        result = super(CausalConv1d, self).call(inputs)
        if self.__padding != 0:
            return result[:, :-self.__padding, :]
        return result

class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        
        self.convs = [CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1) for _ in range(3)]
        self.linear = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, query, key, value):
        nbatches = tf.shape(query)[0]

        query = tf.transpose(tf.reshape(query, [nbatches, -1, self.h, self.d_k]), [0, 2, 1, 3])
        key = tf.transpose(tf.reshape(self.convs[1](key), [nbatches, -1, self.h, self.d_k]), [0, 2, 1, 3])
        value = tf.transpose(tf.reshape(self.convs[2](value), [nbatches, -1, self.h, self.d_k]), [0, 2, 1, 3])

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [nbatches, -1, self.h * self.d_k])

        return self.linear(x)


class LayerNorm(tf.keras.layers.Layer):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = tf.Variable(tf.ones([features]), trainable=True)
        self.b_2 = tf.Variable(tf.zeros([features]), trainable=True)
        self.eps = eps

    def call(self, x):
        mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(tf.keras.layers.Layer):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, sublayer, *args, **kwargs):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)), *args, **kwargs)


def clones(layer, N):
    return [deepcopy(layer) for _ in range(N)]


class TCE(tf.keras.layers.Layer):
    """
    Transformer Encoder
    It is a stack of N layers.
    """

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class EncoderLayer(tf.keras.layers.Layer):
    '''
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''
    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = [SublayerOutput(size, dropout) for _ in range(2)]
        self.size = size
        self.conv = tf.keras.layers.Conv1D(afr_reduced_cnn_size, kernel_size=7, strides=1, dilation_rate=1)

    def call(self, x_in, training=False):
        "Transformer Encoder"
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in), training=training)  # Encoder self-attention
        return self.sublayer_output[1](x, self.feed_forward, training=training)


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.w_2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(self.w_1(x)))


class AttnSleep(tf.keras.Model):
    def __init__(self):
        super(AttnSleep, self).__init__()

        N = 2  # number of TCE clones
        d_model = 80  # set to be 100 for SHHS dataset
        d_ff = 120   # dimension of feed forward
        h = 5  # number of attention heads
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30

        self.mrcnn = MRCNN(afr_reduced_cnn_size) # use MRCNN_SHHS for SHHS dataset

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x_feat = self.mrcnn(x)
        encoded_features = self.tce(x_feat)
        encoded_features = tf.reshape(encoded_features, (encoded_features.shape[0], -1))
        final_output = self.fc(encoded_features)
        return final_output

######################################################################

class MRCNN_SHHS(tf.keras.Model):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN_SHHS, self).__init__()
        drate = 0.5
        self.GELU = tf.keras.layers.Activation(tf.nn.gelu)
        self.features1 = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, kernel_size=50, strides=6, use_bias=False, padding='same'),
            tf.keras.layers.BatchNormalization(),
            self.GELU,
            tf.keras.layers.MaxPooling1D(pool_size=8, strides=2, padding='same'),
            tf.keras.layers.Dropout(drate),

            tf.keras.layers.Conv1D(128, kernel_size=8, strides=1, use_bias=False, padding='same'),
            tf.keras.layers.BatchNormalization(),
            self.GELU,

            tf.keras.layers.Conv1D(128, kernel_size=8, strides=1, use_bias=False, padding='same'),
            tf.keras.layers.BatchNormalization(),
            self.GELU,

            tf.keras.layers.MaxPooling1D(pool_size=4, strides=4, padding='same')
        ])

        self.features2 = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, kernel_size=400, strides=50, use_bias=False, padding='same'),
            tf.keras.layers.BatchNormalization(),
            self.GELU,
            tf.keras.layers.MaxPooling1D(pool_size=4, strides=2, padding='same'),
            tf.keras.layers.Dropout(drate),

            tf.keras.layers.Conv1D(128, kernel_size=6, strides=1, use_bias=False, padding='same'),
            tf.keras.layers.BatchNormalization(),
            self.GELU,

            tf.keras.layers.Conv1D(128, kernel_size=6, strides=1, use_bias=False, padding='same'),
            tf.keras.layers.BatchNormalization(),
            self.GELU,

            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
        ])

        self.dropout = tf.keras.layers.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                tf.keras.layers.Conv1D(planes * block.expansion,
                                       kernel_size=1, strides=stride, use_bias=False),
                tf.keras.layers.BatchNormalization(),
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return tf.keras.Sequential(layers)

    def call(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = tf.concat([x1, x2], axis=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat