# Server runs on python 3.10.0
from fastapi import FastAPI, UploadFile, Form, Request      # fastapi==0.115.11
from fastapi.responses import FileResponse, HTMLResponse    
from fastapi.staticfiles import StaticFiles
from model.u2net import U2NET, U2NETP
from PIL import Image                                       # pillow==11.1.0
from torchvision import transforms                          # torchvision==0.20.1+cu121
import torch                                                # torch==2.5.1+cu121
import os
import tempfile
import uuid
from FuncLib import *
import io
import sys
from datetime import datetime
import time

# Ensure UTF-8 output for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = FastAPI()

# Static folder for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/IO", StaticFiles(directory="IO"), name="IO")

# Globals
models = {}
MODEL_INPUT_W, MODEL_INPUT_H = 320, 320
CHECKPOINTS = {
    "u2net": "../checkpoints/u2net.pth",
    "u2netp": "../checkpoints/u2netp.pth",
    "u2netportrait": "../checkpoints/u2net_portrait.pth"
}

@app.on_event("startup")
def load_models():
    for name, path in CHECKPOINTS.items():
        if name == "u2netp":
            net = U2NETP(3, 1)
        else:
            net = U2NET(3, 1)

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(path, weights_only=True))
            net.cuda()
        else:
            net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))

        net.eval()
        models[name] = net
    print("Models loaded:", list(models.keys()))

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html",  encoding='utf-8') as f:
        return f.read()

@app.post("/predict")
async def predict(
    img_input: UploadFile,
    model_name: str = Form(...),
    sigma: float = Form(...),
    alpha: float = Form(...)
):
    DIRECTORY_IO = "./IO"
    uuidName = uuid.uuid4()
    logs = []
    
    strStartTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmStart = time.time()
    logs.append(f"🟢 Process begin at {strStartTime}")
    temp_input = os.path.join(DIRECTORY_IO, f"{uuidName}_{img_input.filename}")
    with open(temp_input, "wb") as f:
        f.write(await img_input.read())

    image = Image.open(temp_input).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((MODEL_INPUT_W, MODEL_INPUT_H)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    model = models.get(model_name)
    with torch.no_grad():
        d1, *_ = model(image_tensor)

    pred = 1.0 - d1[:, 0, :, :] if model_name == "u2netportrait" else d1[:, 0, :, :]
    pred = doNormalizeTensor(pred)

    if model_name == "u2netportrait":
        imgOut = imageFusionAlpha(pred, image, sigma=sigma, alpha=alpha)
    else:
        imgOut = imageAppendAlpha(pred, image)

    mskOut = imageFromAlpha(pred, image.size)

    output_image_path = os.path.join(DIRECTORY_IO, f"{uuidName}_output.png")
    output_mask_path = os.path.join(DIRECTORY_IO, f"{uuidName}_mask.png")

    imgOut.save(output_image_path)
    mskOut.save(output_mask_path)
    strCloseTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmClose = time.time()
    logs.append(f"✅ Process finished at {strCloseTime}")
    logs.append(f"✅ Summary: {(tmClose-tmStart): .2f} seconds")

    return {
        "output_image": f"{output_image_path}",
        "mask_image": f"{output_mask_path}",
        "logs": logs
    }
