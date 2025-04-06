# GetImageMask runs on python 3.10.0
import torch
from model.u2net import U2NET,U2NETP  # Make sure this path matches your project
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from skimage import io, transform
import argparse
from FuncLib import *

PATH_CHECKPOINT = "../checkpoints"
FILENAME_U2NET  = "u2net.pth"
FILENAME_U2NETP = "u2netp.pth"
FILENAME_U2NET_PORTRAIT = "u2net_portrait.pth"

CHECKPOINT_U2NET = os.path.join(PATH_CHECKPOINT,FILENAME_U2NET)
CHECKPOINT_U2NETP = os.path.join(PATH_CHECKPOINT,FILENAME_U2NETP)
CHECKPOINT_U2NET_PORTRAIT = os.path.join(PATH_CHECKPOINT,FILENAME_U2NET_PORTRAIT)

MODEL_INPUT_W = 320
MODEL_INPUT_H = 320

def get_opt():
    parser = argparse.ArgumentParser(description='GetImageMask: Image mask generator using U2Net family models.')
    parser.add_argument('--img_input', type=str, default='Input01.jpg', help='Input image path')
    parser.add_argument('--img_output', type=str, default='Output.png', help='Output image path, png format')
    parser.add_argument('--msk_output', type=str, default='MaskOutput.png', help='Output mask path, png format')
    parser.add_argument('--model', type=str, default='u2netp',help='Model name: u2net / u2netp(default) / u2netportrait')
    parser.add_argument('--sigma', type=float, default='2',help='Gaussian blur sigma, 2(default), used for u2netportrait only') 
    parser.add_argument('--alpha', type=float, default='0.5', help='Alpha blending factor, 0~1, 0.5(default), used for u2netportrait only')
    opt = parser.parse_args()
    return opt

def main():
    printMemUsage(note="Before loading")
    opt = get_opt()

    log_begin(step_name="WarmupModel")
    # Step 1: Load PyTorch model
    if opt.model=="u2net":
        net = U2NET(3,1)    # 3 ChannelIn, 1 ChannelOut
        chkPointPath = CHECKPOINT_U2NET
    elif opt.model=="u2netportrait":
        net = U2NET(3,1)    # 3 ChannelIn, 1 ChannelOut
        chkPointPath = CHECKPOINT_U2NET_PORTRAIT
    else:
        net = U2NETP(3,1)   # 3 ChannelIn, 1 ChannelOut
        chkPointPath = CHECKPOINT_U2NETP

    # Step 2: Load CheckPoint
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(chkPointPath, weights_only=True))
        net.cuda()
    else:
        net.load_state_dict(torch.load(modelCHECKPOINT_U2NET_dir, map_location='cpu', weights_only=True))
    net.eval()
    log_close(step_name="WarmupModel")

    # Step 3: Prepare Input
    image = Image.open(opt.img_input).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((MODEL_INPUT_W, MODEL_INPUT_H)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    # printPytorchVer(note="PyTorch Version")

    # Step 4: Run model
    log_begin(step_name="RunningModel")
    d1,d2,d3,d4,d5,d6,d7= net(image_tensor)    #d1: final mask d2~d6:Intermediate masks, d7:Early, low-level mask
    if opt.model=="u2netportrait":
        pred = 1.0 - d1[:,0,:,:]
    else:
        pred = d1[:,0,:,:]
    log_close(step_name="RunningModel")

    # Step 5: Pose process and save output
    pred = doNormalizeTensor(pred)
    if opt.model=="u2netportrait":
        imgOut = imageFusionAlpha( pred, image, sigma=opt.sigma, alpha=opt.alpha )
    else:
        imgOut = imageAppendAlpha(pred, image )
    mskOut=imageFromAlpha(pred, image.size )

    imgOut.save(opt.img_output)
    mskOut.save(opt.msk_output)
    printMemUsage(note="Process Finished")

if __name__ == '__main__':
    main()
