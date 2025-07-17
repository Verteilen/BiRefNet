# Imports
from PIL import Image
from pillow_lut import load_cube_file
import torch
from torchvision import transforms
from IPython.display import display

import sys
sys.path.insert(0, "../")
from models.birefnet import BiRefNet

from argparse import ArgumentParser, Namespace

lut = load_cube_file("B2048_add.cube")
parser = ArgumentParser(description="Training script parameters")
parser.add_argument('-i', type=str, default = None)
parser.add_argument('-o', type=str, default = None)
parser.add_argument('-f', type=str, default = None)
args = parser.parse_args(sys.argv[1:])

# Load Model
# Option 2 and Option 3 is better for local running -- we can modify codes locally.

# # # Option 1: loading BiRefNet with weights:
# from transformers import AutoModelForImageSegmentation
# birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet_HR', trust_remote_code=True)

# Option-2: loading weights with BiReNet codes:
birefnet = BiRefNet.from_pretrained(
    [
        'zhengpeng7/BiRefNet_HR',
        'zhengpeng7/BiRefNet',
        'zhengpeng7/BiRefNet-portrait',
        'zhengpeng7/BiRefNet-legacy', 'zhengpeng7/BiRefNet-DIS5K-TR_TEs', 'zhengpeng7/BiRefNet-DIS5K', 'zhengpeng7/BiRefNet-HRSOD', 'zhengpeng7/BiRefNet-COD',
        'zhengpeng7/BiRefNet_lite',     # Modify the `bb` in `config.py` to `swin_v1_tiny`.
    ][6]
)

# # Option-3: Loading model and weights from local disk:
# from utils import check_state_dict

# birefnet = BiRefNet(bb_pretrained=False)
# state_dict = torch.load('../BiRefNet-general-epoch_244.pth', map_location='cpu', weights_only=True)
# state_dict = check_state_dict(state_dict)
# birefnet.load_state_dict(state_dict)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_float32_matmul_precision(['high', 'highest'][0])

birefnet.to(device)
birefnet.eval()
print('BiRefNet is ready to use.')

# Input Data
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import os
from glob import glob
import numpy as np
from image_proc import refine_foreground

autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=[torch.float16, torch.bfloat16][0])
src_dir = args.i
image_paths = sorted(glob(os.path.join(src_dir, '*')))
dst_dir = args.o
final_dir = args.f
com_dir = '../comparisons'
os.makedirs(dst_dir, exist_ok=True)
os.makedirs(final_dir, exist_ok=True)
for image_path in image_paths[:]:
    if image_path.endswith('.jpg') or image_path.endswith('.png'):
        print('Processing {} ...'.format(image_path))
        image = Image.open(image_path)
        image = image.convert("RGB") if image.mode != "RGB" else image
        input_images = transform_image(image).unsqueeze(0).to(device)

        # Prediction
        with autocast_ctx, torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().to(torch.float32).cpu()
        pred = preds[0].squeeze()

        # Show Results
        pred_pil = transforms.ToPILImage()(pred)
        pred_pil.resize(image.size).save(image_path.replace(src_dir, dst_dir))

        image_masked = refine_foreground(image, pred_pil)
        image_masked.putalpha(pred_pil.resize(image.size))

        # Comparison Results
        array_foreground = np.array(image_masked)[:, :, :3].astype(np.float32)
        array_mask = (np.array(image_masked)[:, :, 3:] / 255).astype(np.float32)
        array_background = np.zeros_like(array_foreground)
        array_background[:, :, :] = (0, 0, 0)
        array_foreground_background = (array_foreground * array_mask + array_background * (1 - array_mask)).astype(np.uint8)
        com_img = Image.new('RGB', (image.width, image.height))
        # com_img.paste(pred_pil.resize(image.size), (0, 0))
        # com_img.paste(image, (image.width, 0))
        com_img.paste(Image.fromarray(array_foreground_background), (0, 0))
        com_img.save(image_path.replace(src_dir, final_dir))

