Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
AttnSleep                                          [128, 5]                  --
├─MRCNN: 1-1                                       [128, 30, 80]             --
│    └─Sequential: 2-1                             [128, 128, 64]            197,120
│    │    └─Conv1d: 3-1                            [128, 64, 500]            3,200
│    │    └─BatchNorm1d: 3-2                       [128, 64, 500]            128
│    └─Sequential: 2-6                             --                        (recursive)
│    │    └─GELU: 3-3                              [128, 64, 500]            --
│    └─Sequential: 2-7                             --                        (recursive)
│    │    └─MaxPool1d: 3-4                         [128, 64, 251]            --
│    │    └─Dropout: 3-5                           [128, 64, 251]            --
│    │    └─Conv1d: 3-6                            [128, 128, 252]           65,536
│    │    └─BatchNorm1d: 3-7                       [128, 128, 252]           256
│    └─Sequential: 2-6                             --                        (recursive)
│    │    └─GELU: 3-8                              [128, 128, 252]           --
│    └─Sequential: 2-7                             --                        (recursive)
│    │    └─Conv1d: 3-9                            [128, 128, 253]           131,072
│    │    └─BatchNorm1d: 3-10                      [128, 128, 253]           256
│    └─Sequential: 2-6                             --                        (recursive)
│    │    └─GELU: 3-11                             [128, 128, 253]           --
│    └─Sequential: 2-7                             --                        (recursive)
│    │    └─MaxPool1d: 3-12                        [128, 128, 64]            --
│    └─Sequential: 2-8                             [128, 128, 16]            --
│    │    └─Conv1d: 3-13                           [128, 64, 61]             25,600
│    │    └─BatchNorm1d: 3-14                      [128, 64, 61]             128
│    │    └─GELU: 3-15                             [128, 64, 61]             --
│    │    └─MaxPool1d: 3-16                        [128, 64, 31]             --
│    │    └─Dropout: 3-17                          [128, 64, 31]             --
│    │    └─Conv1d: 3-18                           [128, 128, 31]            57,344
│    │    └─BatchNorm1d: 3-19                      [128, 128, 31]            256
│    │    └─GELU: 3-20                             [128, 128, 31]            --
│    │    └─Conv1d: 3-21                           [128, 128, 31]            114,688
│    │    └─BatchNorm1d: 3-22                      [128, 128, 31]            256
│    │    └─GELU: 3-23                             [128, 128, 31]            --
│    │    └─MaxPool1d: 3-24                        [128, 128, 16]            --
│    └─Dropout: 2-9                                [128, 128, 80]            --
│    └─Sequential: 2-10                            [128, 30, 80]             --
│    │    └─SEBasicBlock: 3-25                     [128, 30, 80]             --
│    │    │    └─Conv1d: 4-1                       [128, 30, 80]             3,870
│    │    │    └─BatchNorm1d: 4-2                  [128, 30, 80]             60
│    │    │    └─ReLU: 4-3                         [128, 30, 80]             --
│    │    │    └─Conv1d: 4-4                       [128, 30, 80]             930
│    │    │    └─BatchNorm1d: 4-5                  [128, 30, 80]             60
│    │    │    └─SELayer: 4-6                      [128, 30, 80]             --
│    │    │    │    └─AdaptiveAvgPool1d: 5-1       [128, 30, 1]              --
│    │    │    │    └─Sequential: 5-2              [128, 30]                 60
│    │    │    └─Sequential: 4-7                   [128, 30, 80]             --
│    │    │    │    └─Conv1d: 5-3                  [128, 30, 80]             3,840
│    │    │    │    └─BatchNorm1d: 5-4             [128, 30, 80]             60
│    │    │    └─ReLU: 4-8                         [128, 30, 80]             --
├─TCE: 1-2                                         [128, 30, 80]             --
│    └─ModuleList: 2-11                            --                        --
│    │    └─EncoderLayer: 3-26                     [128, 30, 80]             --
│    │    │    └─CausalConv1d: 4-9                 [128, 30, 80]             6,330
│    │    │    └─ModuleList: 4-14                  --                        (recursive)
│    │    │    │    └─SublayerOutput: 5-5          [128, 30, 80]             160
│    │    │    └─MultiHeadedAttention: 4-11        [128, 30, 80]             --
│    │    │    │    └─ModuleList: 5-6              --                        18,990
│    │    │    │    └─Dropout: 5-7                 [128, 5, 30, 30]          --
│    │    │    │    └─Linear: 5-8                  [128, 30, 80]             6,480
│    │    │    └─ModuleList: 4-14                  --                        (recursive)
│    │    │    │    └─SublayerOutput: 5-9          --                        (recursive)
│    │    │    │    └─SublayerOutput: 5-10         [128, 30, 80]             160
│    │    │    └─PositionwiseFeedForward: 4-13     [128, 30, 80]             --
│    │    │    │    └─Linear: 5-11                 [128, 30, 120]            9,720
│    │    │    │    └─Dropout: 5-12                [128, 30, 120]            --
│    │    │    │    └─Linear: 5-13                 [128, 30, 80]             9,680
│    │    │    └─ModuleList: 4-14                  --                        (recursive)
│    │    │    │    └─SublayerOutput: 5-14         --                        (recursive)
│    │    └─EncoderLayer: 3-27                     [128, 30, 80]             --
│    │    │    └─CausalConv1d: 4-15                [128, 30, 80]             6,330
│    │    │    └─ModuleList: 4-20                  --                        (recursive)
│    │    │    │    └─SublayerOutput: 5-15         [128, 30, 80]             160
│    │    │    └─MultiHeadedAttention: 4-17        [128, 30, 80]             --
│    │    │    │    └─ModuleList: 5-16             --                        18,990
│    │    │    │    └─Dropout: 5-17                [128, 5, 30, 30]          --
│    │    │    │    └─Linear: 5-18                 [128, 30, 80]             6,480
│    │    │    └─ModuleList: 4-20                  --                        (recursive)
│    │    │    │    └─SublayerOutput: 5-19         --                        (recursive)
│    │    │    │    └─SublayerOutput: 5-20         [128, 30, 80]             160
│    │    │    └─PositionwiseFeedForward: 4-19     [128, 30, 80]             --
│    │    │    │    └─Linear: 5-21                 [128, 30, 120]            9,720
│    │    │    │    └─Dropout: 5-22                [128, 30, 120]            --
│    │    │    │    └─Linear: 5-23                 [128, 30, 80]             9,680
│    │    │    └─ModuleList: 4-20                  --                        (recursive)
│    │    │    │    └─SublayerOutput: 5-24         --                        (recursive)
│    └─LayerNorm: 2-12                             [128, 30, 80]             160
├─Linear: 1-3                                      [128, 5]                  12,005
====================================================================================================
Total params: 719,925
Trainable params: 719,925
Non-trainable params: 0
Total mult-adds (G): 7.93
====================================================================================================
Input size (MB): 1.54
Forward/backward pass size (MB): 281.19
Params size (MB): 2.04
Estimated Total Size (MB): 284.76
====================================================================================================