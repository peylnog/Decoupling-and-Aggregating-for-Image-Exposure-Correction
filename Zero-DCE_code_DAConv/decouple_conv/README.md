<!--
 * @Date: 2023-07-17 09:58:05
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2023-07-17 10:12:42
 * @FilePath: /date/TMP_workshop/decouple_conv/README.md
-->

# FileDescription

* The ops.py file contains CAUnit and DAUnit class, corresponding to CA Unit and DA Unit in the paper.
* The ops_decouple.py file contains DAConv class, corresponding DAConv  in the paper.
* You can use decouple_conv_layer function in ops_decouple.py. It has almost the same API as nn.Conv2d as follows:


```
import torch
import torch.nn as nn 
from decouple_conv.ops_decouple import decouple_conv_layer

a = torch.randn(size=(1,3,256,256))
cnn_layer = nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,padding=1,stride=1)
daconv_layer = decouple_conv_layer(in_channels=3,out_channels=8,kernel_size=3,padding=1,stride=1)

print(cnn_layer(a).shape)
print(daconv_layer(a).shape)
```
* merge_network.py provides the fusion process of the PRenet model. You can use the merge_network_re function in this file to merge all DAconv-based model to VC-based model and get the same structure and computational cost as the original CNN-based network.

Please refer to the code file for more details.