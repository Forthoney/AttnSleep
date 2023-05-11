Model: "attn_sleep"
____________________________________________________________________________
 Layer (type)                Output Shape              Param #   Trainable
============================================================================
 multiresolution_cnn (Multir  multiple                 409060    Y
 esolutionCNN)
|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
| high_res_features (Sequenti  (128, 63, 128)         201088    Y          |
| al)                                                                      |
||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
|| conv1d (Conv1D)         (128, 500, 64)            3200      Y          ||
||                                                                        ||
|| batch_normalization (BatchN  (128, 500, 64)       256       Y          ||
|| ormalization)                                                          ||
||                                                                        ||
|| activation (Activation)  (128, 500, 64)           0         Y          ||
||                                                                        ||
|| max_pooling1d (MaxPooling1D  (128, 250, 64)       0         Y          ||
|| )                                                                      ||
||                                                                        ||
|| dropout (Dropout)       (128, 250, 64)            0         Y          ||
||                                                                        ||
|| conv1d_1 (Conv1D)       (128, 250, 128)           65536     Y          ||
||                                                                        ||
|| batch_normalization_1 (Batc  (128, 250, 128)      512       Y          ||
|| hNormalization)                                                        ||
||                                                                        ||
|| activation_1 (Activation)  (128, 250, 128)        0         Y          ||
||                                                                        ||
|| conv1d_2 (Conv1D)       (128, 250, 128)           131072    Y          ||
||                                                                        ||
|| batch_normalization_2 (Batc  (128, 250, 128)      512       Y          ||
|| hNormalization)                                                        ||
||                                                                        ||
|| activation_2 (Activation)  (128, 250, 128)        0         Y          ||
||                                                                        ||
|| max_pooling1d_1 (MaxPooling  (128, 63, 128)       0         Y          ||
|| 1D)                                                                    ||
|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
| low_res_features (Sequentia  (128, 15, 128)         198912    Y          |
| l)                                                                       |
||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
|| conv1d_3 (Conv1D)       (128, 60, 64)             25600     Y          ||
||                                                                        ||
|| batch_normalization_3 (Batc  (128, 60, 64)        256       Y          ||
|| hNormalization)                                                        ||
||                                                                        ||
|| activation_3 (Activation)  (128, 60, 64)          0         Y          ||
||                                                                        ||
|| max_pooling1d_2 (MaxPooling  (128, 30, 64)        0         Y          ||
|| 1D)                                                                    ||
||                                                                        ||
|| dropout_1 (Dropout)     (128, 30, 64)             0         Y          ||
||                                                                        ||
|| conv1d_4 (Conv1D)       (128, 30, 128)            57344     Y          ||
||                                                                        ||
|| batch_normalization_4 (Batc  (128, 30, 128)       512       Y          ||
|| hNormalization)                                                        ||
||                                                                        ||
|| activation_4 (Activation)  (128, 30, 128)         0         Y          ||
||                                                                        ||
|| conv1d_5 (Conv1D)       (128, 30, 128)            114688    Y          ||
||                                                                        ||
|| batch_normalization_5 (Batc  (128, 30, 128)       512       Y          ||
|| hNormalization)                                                        ||
||                                                                        ||
|| activation_5 (Activation)  (128, 30, 128)         0         Y          ||
||                                                                        ||
|| max_pooling1d_3 (MaxPooling  (128, 15, 128)       0         Y          ||
|| 1D)                                                                    ||
|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
| dropout_2 (Dropout)       multiple                  0         Y          |
|                                                                          |
| AFR (Sequential)          (128, 78, 30)             9060      Y          |
||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
|| se_basic_block (SEBasicBloc  (128, 78, 30)        9060      Y          ||
|| k)                                                                     ||
|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
 transformer_encoder (Transf  multiple                 107568    Y
 ormerEncoder)
|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
| encoder_layer (EncoderLayer  multiple               53754     Y          |
| )                                                                        |
|                                                                          |
| encoder_layer (EncoderLayer  multiple               53754     Y          |
| )                                                                        |
¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
 dense_5 (Dense)             multiple                  11705     Y

============================================================================
Total params: 528,333
Trainable params: 526,873
Non-trainable params: 1,460
____________________________________________________________________________
