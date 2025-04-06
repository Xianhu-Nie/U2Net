import psutil                           # psutil==7.0.0
import torch
import os
import time
import numpy as np                      # numpy==1.26.4
from PIL import Image
from skimage import io, transform       # scikit-image==0.25.2
from skimage.filters import gaussian

# ===== Step timing tracker =====
version = "1.0.0"
step_times = {}

def getVersion(bPrint=True):
    if bPrint:
        print(f"[FuncLib Ver] {version}")
    return version

# CPU and GPU memory usage
def printMemUsage(bPrint=True,note=""):
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 ** 2
    gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2
    if bPrint:
        print(f"[{note}] CPU Memory: {cpu_mem:.2f} MB | GPU Memory: {gpu_mem:.2f} MB")
    return cpu_mem, gpu_mem
# PyTorch Version
def printPytorchVer(bPrint=True,note=""):
    if bPrint:
        print(f"[{note}]: {torch.__version__}")
    return torch.__version__
# CurrentTime, seconds since Jan 1, 1970 UTC
def printCurrentTime(bPrint=True,note=""):
    dtNow = time.time()
    if bPrint:
        print(f"[{note}]: {dtNow}")
    return dtNow

# Begin a Log, it will show time elapsed between begin and close
def log_begin(bPrint=True,step_name="", bFlash=True):
    step_times[step_name] = time.time()
    if bPrint: print(f"🟢 {step_name}...", flush=bFlash)
    return step_times[step_name]
def log_close(bPrint=True,step_name="",bFlash=True):
    elapsed = time.time() - step_times[step_name]
    if bPrint: print(f"✅ {step_name} completed in {elapsed:.2f} seconds.", flush=bFlash)
    step_times[step_name] = elapsed  
    return elapsed

# Normalize all value between 0~1, input is a tensor variable
def doNormalizeTensor(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

# Append Alpha Channel to given image
# type(alpha_ts): <class 'torch.Tensor'> size:torch.Size([1, 320, 320])
# type(imgIn): <class 'PIL.Image.Image'> size:(1280, 1707) (W*H)
# type(imgOut): <class 'PIL.Image.Image'> size:(1280, 1707) (W*H)(same as imgIn)
def imageAppendAlpha( alpha_ts, imgIn ):
    alpha_np = alpha_ts.squeeze().cpu().data.numpy()  # Tensor -> numpy, squeeze to W*H (320*320), value from 0.0~1.0
    alpha_np = (alpha_np * 255).astype(np.uint8)      # float(0~1) -> uint8(0~255)
    alpha_np = Image.fromarray(alpha_np).resize(imgIn.size, resample=Image.BILINEAR)  #resize to the same size as imgIn

    # Create RGBA image: original image with alpha mask
    image = imgIn.convert('RGBA')               # change mode from RGB to RGBA
    image_np = np.array(image)                  # image_np will be (1707, 1280, 4)
    image_np[:, :, 3] = np.array(alpha_np)      # append alpha channel

    imgOut = Image.fromarray(image_np)
    return imgOut

# Export Alpha Channel as Grayscale image
# type(alpha_ts): <class 'torch.Tensor'> size:torch.Size([1, 320, 320])
# type(imgSize): <class 'tuple'> (1280, 1707) (W*H)
# type(imgOut): <class 'PIL.Image.Image'> size:(1280, 1707) (W*H)(as defined in imgSize)
def imageFromAlpha( alpha_ts, imgSize ):
    alpha_np = alpha_ts.squeeze().cpu().data.numpy()        # Tensor -> numpy, squeeze to W*H (320*320), value from 0.0~1.0
    imgOut = Image.fromarray(alpha_np*255).convert('L')     # float(0~1) -> uint8(0~255) and convert to Image
    imgOut = imgOut.resize(imgSize,resample=Image.BILINEAR) # resize to the given size
    return imgOut

# Fusion the alpha channel and original RGB channel
# type(alpha_ts): <class 'torch.Tensor'> size:torch.Size([1, 320, 320])
# type(imgIn): <class 'PIL.Image.Image'> size:(1280, 1707) (W*H)
# sigma: affect the gaussian result, the bigger, the more blur. 
# 0.5:Very light blur, details sharp 
# 2:Moderate blur, softens edges. 
# 5:Strong blur, details fade away. 
# 10:Very strong blur, image looks "foggy" or abstract.
# alpha: 0: alpha only 1:image only
# NOTICE: when using np.array(imgIn) change image to array, the array will be H*W
def imageFusionAlpha(alpha_ts, imgIn ,sigma=2, alpha=0.35):
    alpha_np = alpha_ts.squeeze().cpu().data.numpy()                                # Tensor -> numpy, squeeze to W*H (320*320), value from 0.0~1.0
    alpha_np = (alpha_np * 255).astype(np.float32)                                  # float(0~1) -> float(0~255)
    alpha_np = transform.resize(alpha_np,(imgIn.size[1],imgIn.size[0]),order=2)     # Array should be H*W
    alpha_np = alpha_np[:,:,np.newaxis]                                             # Add new dimension 

    imgArray = gaussian(np.array(imgIn), sigma=sigma, preserve_range=True)          # imgArray will be 1707*1280 (H*W)
    imgArray = imgArray*alpha+alpha_np*(1-alpha)                                    # Average with weight
    imgArray = imgArray.astype(np.uint8)                                            # convert to uint8
    imgOut = Image.fromarray(imgArray).convert('RGB')                               # Convert to Image
    return imgOut
