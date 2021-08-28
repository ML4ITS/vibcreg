In this folder we have made an easy-to-use pip installable package for using VIbCReg with minimal requirements. 
To install, type:
```
pip install vibcreg
```


To use the VIbCReg loss presented in the paper, simply import and use it like this: 
```
from vibcreg import VIbCRegLoss

view1 = get view 1 of data
view2 = get view 2 of data
crit = VIbCRegLoss()
loss = crit(view1, view2)
```
This can then be used in a training loop as usual.

In addition, one can use a pre-implemeted module, which contains a backbone and projector as described in the paper.
To use it, import it:

```
from vibcreg import VIbCReg

bacbone_constructor = some callable that constructs a backbone network.
vibcreg = VIbCReg(bacbone_constructor, backbone_output_dimensionality)
```