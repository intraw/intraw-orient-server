from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import base64
import uvicorn

from paths import DINO_LARGE
from vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor
from inference import get_3angle, get_3angle_infer_aug
from utils import background_preprocess, render_3D_axis, overlay_images_with_scaling
from huggingface_hub import hf_hub_download

app = FastAPI(title="Orient Anything API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model checkpoint from HuggingFace Hub...")
ckpt_path = hf_hub_download(
    repo_id="Viglong/Orient-Anything",
    filename="ronormsigma1/dino_weight.pt",
    repo_type="model",
    cache_dir="./",
    resume_download=True,
)
print(f"Checkpoint: {ckpt_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

dino = DINOv2_MLP(
    dino_mode="large",
    in_dim=1024,
    out_dim=360 + 180 + 360 + 2,
    evaluate=True,
    mask_dino=False,
    frozen_back=False,
)
dino.eval()
dino.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
dino = dino.to(device)
print("Model weights loaded.")

val_preprocess = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir="./")
print("Image processor ready.")
print("=" * 50)
print(f"Server running at http://localhost:8004")
print("=" * 50)


def pil_to_base64(image: Image.Image) -> str:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.get("/")
async def health_check():
    return {"status": "ok", "model": "Orient-Anything", "device": device}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    do_remove_background: bool = Form(True),
    do_infer_aug: bool = Form(False),
):
    try:
        raw = await file.read()
        origin_img = Image.open(io.BytesIO(raw)).convert("RGB")

        if do_infer_aug:
            rm_bkg_img = background_preprocess(origin_img, True)
            angles = get_3angle_infer_aug(origin_img, rm_bkg_img, dino, val_preprocess, device)
        else:
            rm_bkg_img = background_preprocess(origin_img, do_remove_background)
            angles = get_3angle(rm_bkg_img, dino, val_preprocess, device)

        phi = np.radians(float(angles[0]))
        theta = np.radians(float(angles[1]))
        gamma = float(angles[2])
        confidence = float(angles[3])

        if confidence > 0.5:
            render_axis = render_3D_axis(phi, theta, gamma)
            result_img = overlay_images_with_scaling(render_axis, rm_bkg_img)
        else:
            result_img = rm_bkg_img

        return JSONResponse({
            "azimuth":    round(float(angles[0]), 2),
            "polar":      round(float(angles[1]), 2),
            "rotation":   round(float(angles[2]), 2),
            "confidence": round(float(angles[3]), 4),
            "result_image": pil_to_base64(result_img),
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
