Simple Web demostration for U2Net

Run on python 3.10

#### GetImageMask.py

```
GetImageMask: Image mask generator using U2Net family models.
```


```
options:-h, --help            show this help message and exit
--img_input IMG_INPUT          Input image path
--img_output IMG_OUTPUT        Output image path, png format
--msk_output MSK_OUTPUT        Output mask path, png format
--model MODEL         Model name: u2net / u2netp(default) / u2netportrait
--sigma SIGMA         Gaussian blur sigma, 2(default), used for u2netportrait only
--alpha ALPHA         Alpha blending factor, 0~1, 0.5(default), used for u2netportrait only
```
