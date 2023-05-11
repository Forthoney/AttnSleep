import tensorflow as tf


class AdaptiveAveragePooling1D(tf.keras.layers.Layer):

    """TensorFlow Implementation of Pytorch's AdaptiveAvgPool1D. Inspired heavily by
    TFA's AdaptivePooling1D

    Attributes:
        output_size: output size of the channel to be reduced
    """

    def __init__(
        self,
        output_size: int,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.output_size = output_size

    def call(self, inputs):
        splits = tf.split(inputs, self.output_size, axis=1)
        splits = tf.stack(splits, axis=1)
        out_vect = tf.reduce_mean(splits, axis=2)
        return out_vect

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        shape = tf.TensorShape([input_shape[0], self.output_size, input_shape[2]])
        return shape


class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, channel: int, reduction: int = 16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = AdaptiveAveragePooling1D(1)
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
        output = tf.multiply(inputs, tf.broadcast_to(y, tf.shape(inputs)))
        return output


class SqueezeExcitationBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(
        self,
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
        super(SqueezeExcitationBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(planes, stride)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv1D(planes, 1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SqueezeExcitation(planes, reduction)
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


class MultiresolutionCNN(tf.keras.Model):
    def __init__(self, afr_reduced_cnn_size):
        super(MultiresolutionCNN, self).__init__()
        drate = 0.5
        self.features1 = tf.keras.Sequential(
            name="high_res_features",
            layers=[
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
            ],
        )

        self.features2 = tf.keras.Sequential(
            name="low_res_features",
            layers=[
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
            ],
        )
        self.dropout = tf.keras.layers.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_afr_layer(afr_reduced_cnn_size, 1)

    def _make_afr_layer(
        self, planes: int, n_blocks: int, stride=1
    ):  # makes residual SE block
        downsample = None
        block = SqueezeExcitationBlock
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
        layers.append(block(planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(planes))

        return tf.keras.Sequential(layers, "adaptive_feature_recalibration")

    def call(self, x):
        # Input Shape: (batch_size, 3000, 1)
        x1 = self.features1(x)  # Output Shape: (128, 63, 128)
        x2 = self.features2(x)  # Output Shape: (128, 15, 128)
        x_concat = tf.concat([x1, x2], axis=1)  # Output Shape: (128, 78, 128)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(
        self, num_heads: int, model_dim: int, afr_reduced_cnn_size: int, dropout=0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.afr_reduced_cnn_size = afr_reduced_cnn_size

        self.convs = [
            tf.keras.layers.Conv1D(
                filters=afr_reduced_cnn_size, kernel_size=7, strides=1, padding="causal"
            )
            for _ in range(3)
        ]
        self.linear = tf.keras.layers.Dense(model_dim)

        self.multihead_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=model_dim // num_heads, dropout=dropout
        )
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


class TemporalContextEncoder(tf.keras.layers.Layer):
    """
    Transformer Encoder
    It is a stack of n_tce layers.
    """

    def __init__(self, d_model, d_cnn, d_ff, n_heads, dropout, n_tce):
        super(TemporalContextEncoder, self).__init__()
        self.encoder_layers = tf.keras.Sequential(
            [
                EncoderLayer(
                    self_attn=MultiHeadedAttention(n_heads, d_model, d_cnn),
                    feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout),
                    filter_size=d_cnn,
                    dropout=dropout,
                )
                for _ in range(n_tce)
            ]
        )
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        encoded_x = self.encoder_layers(x)
        return self.norm(encoded_x)


class EncoderLayer(tf.keras.layers.Layer):
    """
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm.
    """

    def __init__(
        self,
        self_attn: tf.keras.layers.Layer,
        feed_forward: tf.keras.layers.Layer,
        filter_size: int,
        dropout: float,
    ):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.norm_2 = tf.keras.layers.LayerNormalization()

        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)

        self.conv = tf.keras.layers.Conv1D(
            filter_size,
            kernel_size=7,
            strides=1,
            dilation_rate=1,
            padding="causal",
        )

    def call(self, x_in):
        "Transformer Encoder"
        query = self.conv(x_in)

        x = self.self_attn(self.norm_1(query), x_in, x_in)
        x = query + self.dropout_1(x)

        ff = self.feed_forward(self.norm_1(x))
        return x + self.dropout_2(ff)


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = tf.keras.layers.Dense(d_ff, activation="relu")
        self.w_2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        "Implements FFN equation."
        x = tf.transpose(x, [0, 2, 1])
        outputs = self.w_2(self.dropout(self.w_1(x)))
        return tf.transpose(outputs, [0, 2, 1])


class AttnSleep(tf.keras.Model):
    def __init__(self):
        super(AttnSleep, self).__init__()

        n_tce: int = 2  # number of TCE clones
        d_model: int = 78  # TODO: d_model needs to be divisible by h
        d_ff: int = 120  # dimension of feed forward
        n_heads: int = 6  # TODO: Originally 5
        dropout: float = 0.1
        num_classes: int = 5
        d_cnn: int = 30

        self.mrcnn = MultiresolutionCNN(d_cnn)  # use MRCNN_SHHS for SHHS dataset
        self.tce = TemporalContextEncoder(d_model, d_cnn, d_ff, n_heads, dropout, n_tce)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, x):
        x_features = self.mrcnn(x)
        encoded_features = self.tce(x_features)
        encoded_features = self.flatten(encoded_features)
        final_output = self.fc(encoded_features)
        return final_output


######################################################################


class MultiresolutionCNN_SHHS(tf.keras.Model):
    def __init__(self, afr_reduced_cnn_size):
        super(MultiresolutionCNN_SHHS, self).__init__()
        drate = 0.5
        self.features1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, kernel_size=50, strides=6, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                tf.keras.layers.MaxPooling1D(pool_size=8, strides=2, padding="same"),
                tf.keras.layers.Dropout(drate),
                tf.keras.layers.Conv1D(
                    128, kernel_size=8, strides=1, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                tf.keras.layers.Conv1D(
                    128, kernel_size=8, strides=1, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                tf.keras.layers.MaxPooling1D(pool_size=4, strides=4, padding="same"),
            ]
        )

        self.features2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, kernel_size=400, strides=50, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                tf.keras.layers.MaxPooling1D(pool_size=4, strides=2, padding="same"),
                tf.keras.layers.Dropout(drate),
                tf.keras.layers.Conv1D(
                    128, kernel_size=6, strides=1, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                tf.keras.layers.Conv1D(
                    128, kernel_size=6, strides=1, use_bias=False, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="same"),
            ]
        )

        self.dropout = tf.keras.layers.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_afr_layer(afr_reduced_cnn_size, 1)

    def _make_afr_layer(
        self, planes: int, n_blocks: int, stride: int = 1
    ):  # makes residual SE block
        block = SqueezeExcitationBlock
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
        for i in range(1, n_blocks):
            layers.append(block(self.inplanes, planes))

        return tf.keras.Sequential(layers, name="adaptive_feature_recalibration")

    def call(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = tf.concat([x1, x2], axis=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat
