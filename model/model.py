from copy import deepcopy

import tensorflow as tf

########################################################################################
def clones(layer, N):
    return [deepcopy(layer) for _ in range(N)]

class SELayer(tf.keras.layers.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.fc = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(channel // reduction, use_bias=False),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(channel, use_bias=False),
                tf.keras.layers.Activation(tf.keras.activations.sigmoid),
            ]
        )

    def call(self, inputs):
        # Input shape (batch_size, 78, 30)
        y = self.avg_pool(inputs)
        y = self.fc(y)
        y = tf.expand_dims(y, 1)
        output = tf.multiply(inputs, tf.broadcast_to(y, tf.shape(inputs)))
        return output


class SEBasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        *,
        reduction=16
    ):
        super(SEBasicBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(planes, stride)
        self.bn1 = tf.keras.layers.BatchNormalization()
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


class MRCNN(tf.keras.Model):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.features1 = tf.keras.Sequential(
            [
                # Convolution 1
                tf.keras.layers.Conv1D(
                    64, kernel_size=50, strides=6, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                tf.keras.layers.MaxPool1D(pool_size=8, strides=2, padding="same"),
                tf.keras.layers.Dropout(drate),
                # Convolution 2
                tf.keras.layers.Conv1D(
                    128, kernel_size=8, strides=1, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                # Convolution 3
                tf.keras.layers.Conv1D(
                    128, kernel_size=8, strides=1, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                tf.keras.layers.MaxPool1D(pool_size=4, strides=4, padding="same"),
            ]
        )

        self.features2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, kernel_size=400, strides=50, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                tf.keras.layers.MaxPool1D(pool_size=4, strides=2, padding="same"),
                tf.keras.layers.Dropout(drate),
                tf.keras.layers.Conv1D(
                    128, kernel_size=7, strides=1, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                tf.keras.layers.Conv1D(
                    128, kernel_size=7, strides=1, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding="same"),
            ]
        )
        self.dropout = tf.keras.layers.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1D(
                        planes * block.expansion,
                        kernel_size=1,
                        strides=stride,
                        use_bias=False,
                    ),
                    tf.keras.layers.BatchNormalization(),
                ]
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return tf.keras.Sequential(layers)

    def call(self, x):
        # Input Shape: (batch_size, 3000, 1)
        x1 = self.features1(x) # Output Shape: (128, 63, 128)
        x2 = self.features2(x) # Output Shape: (128, 15, 128)
        x_concat = tf.concat([x1, x2], axis=1) # Output Shape: (128, 78, 128)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat


##########################################################################################

class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, model_dim, afr_reduced_cnn_size, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.afr_reduced_cnn_size = afr_reduced_cnn_size

        self.convs = [tf.keras.layers.Conv1D(filters=afr_reduced_cnn_size, kernel_size=7, strides=1, padding='causal') for _ in range(3)]
        self.linear = tf.keras.layers.Dense(model_dim)

        self.multihead_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=model_dim//num_heads, dropout=dropout)
        self.reshape2 = tf.keras.layers.Reshape((num_heads, model_dim // num_heads, -1))

    def call(self, query, key, value):
        # Input shape
        # query: [batch_size, 78, 30]
        # key: [batch_size, 78, 30]
        # value: [batch_size, 78, 30]
        # may not be necessary
        query = tf.keras.layers.Reshape((-1, self.afr_reduced_cnn_size))(query) 

        # query = self.convs[0](query)
        key = self.convs[1](key)
        value = self.convs[2](value)

        query = self.reshape2(query)
        key = self.reshape2(key)
        value = self.reshape2(value)

        x = self.multihead_attention(query, value, key)
        x = tf.keras.layers.Reshape((-1, self.model_dim))(x)
        x = self.linear(x)

        # Output shape [batch_size, 78, 30]
        return tf.transpose(x, [0, 2, 1])

class SublayerOutput(tf.keras.layers.Layer):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, sublayer, *args, **kwargs):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)), *args, **kwargs)

class TCE(tf.keras.layers.Layer):
    """
    Transformer Encoder
    It is a stack of N layers.
    """

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EncoderLayer(tf.keras.layers.Layer):
    """
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    """

    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = [SublayerOutput(size, dropout) for _ in range(2)]
        self.size = size
        self.conv = tf.keras.layers.Conv1D(
            afr_reduced_cnn_size, kernel_size=7, strides=1, dilation_rate=1, padding='causal'
        )

    def call(self, x_in, training=False):
        "Transformer Encoder"
        query = self.conv(x_in)
        x = self.sublayer_output[0](
            query, lambda x: self.self_attn(query, x_in, x_in), training=training
        )  # Encoder self-attention
        return self.sublayer_output[1](x, self.feed_forward, training=training)


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = tf.keras.layers.Dense(d_ff, activation="relu")
        self.w_2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        "Implements FFN equation."
        x = tf.transpose(x, [0, 2, 1])
        outputs =  self.w_2(self.dropout(self.w_1(x)))
        return tf.transpose(outputs, [0, 2, 1])

class AttnSleep(tf.keras.Model):
    def __init__(self):
        super(AttnSleep, self).__init__()

        N = 2  # number of TCE clones
        d_model = 78  #TODO: d_model needs to be divisible by h
        d_ff = 120  # dimension of feed forward
        h = 6  #TODO: Originally 5
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30

        self.mrcnn = MRCNN(afr_reduced_cnn_size)  # use MRCNN_SHHS for SHHS dataset

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(
            EncoderLayer(
                d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout
            ),
            N,
        )

        self.fc = tf.keras.layers.Dense(num_classes, activation="softmax")

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
        self.features1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, kernel_size=50, strides=6, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                self.GELU,
                tf.keras.layers.MaxPooling1D(pool_size=8, strides=2, padding="same"),
                tf.keras.layers.Dropout(drate),
                tf.keras.layers.Conv1D(
                    128, kernel_size=8, strides=1, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                self.GELU,
                tf.keras.layers.Conv1D(
                    128, kernel_size=8, strides=1, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                self.GELU,
                tf.keras.layers.MaxPooling1D(pool_size=4, strides=4, padding="same"),
            ]
        )

        self.features2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, kernel_size=400, strides=50, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                self.GELU,
                tf.keras.layers.MaxPooling1D(pool_size=4, strides=2, padding="same"),
                tf.keras.layers.Dropout(drate),
                tf.keras.layers.Conv1D(
                    128, kernel_size=6, strides=1, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                self.GELU,
                tf.keras.layers.Conv1D(
                    128, kernel_size=6, strides=1, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                self.GELU,
                tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="same"),
            ]
        )

        self.dropout = tf.keras.layers.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1D(
                        planes * block.expansion,
                        kernel_size=1,
                        strides=stride,
                        use_bias=False,
                    ),
                    tf.keras.layers.BatchNormalization(),
                ]
            )

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
