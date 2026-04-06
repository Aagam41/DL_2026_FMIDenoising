"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import ndimage as nd
from einops import rearrange
import traceback

# INPUT_PATH = Path("/input")
# OUTPUT_PATH = Path("/output")
INPUT_PATH = Path("/home/aagamsheth/Documents/Education/Ahmedabad University/MTECH CSE 2025 - 2027/SEM 2/CSE 602/Project/AI4Life-CIDC25/test/input/interf0")
OUTPUT_PATH = Path("/home/aagamsheth/Documents/Education/Ahmedabad University/MTECH CSE 2025 - 2027/SEM 2/CSE 602/Project/AI4Life-CIDC25/test/output/interf0")

RESOURCE_PATH = Path("resources")

# --- MODEL / INFERENCE CONFIG ---
# MODEL_PATH = "/opt/ml/model/best_model_n2v.pth"
MODEL_PATH = "/home/aagamsheth/Documents/Education/Ahmedabad University/MTECH CSE 2025 - 2027/SEM 2/CSE 602/Project/AI4Life-CIDC25/best_model_n2v.pth"
TILE_SIZE = (32, 128, 128)
OVERLAP = (4, 16, 16)

SEED = 3407
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_num = 450
max_epoch = 5

FM2S_CONFIG = {
    'n_chan': 1,
    'w_size': 3,
    'stride': 16,
    'g_map': 1.0,
    'p_map': 1.0,
    'lam_p': 200.0,
}


# ===================== N2V 3D UNet =====================
class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, 3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.final = nn.Conv3d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        d1 = self.dec1(torch.cat([self.up1(e2), e1], dim=1))
        return self.final(d1)


# ===================== FM2S Network =====================
class FM2SNetwork(nn.Module):
    def __init__(self, n_chan):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=1e-3)
        self.conv1 = nn.Conv2d(n_chan, 24, 5, padding=2)
        self.conv2 = nn.Conv2d(24, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, n_chan, 5, padding=2)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


def noise_addition(img, config):
    B, C, H, W = img.shape
    stride = config['stride']
    new_H = ((H + stride - 1) // stride) * stride
    new_W = ((W + stride - 1) // stride) * stride
    pad_h = new_H - H
    pad_w = new_W - W

    img_padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
    patches = img_padded.unfold(2, stride, stride).unfold(3, stride, stride)
    noise_idx = patches.mean(dim=(1, 4, 5)).clamp(1e-5, 0.15)
    gaussian_level = config['g_map'] * noise_idx
    poisson_level = config['p_map'] / noise_idx
    gaussian_level_exp = rearrange(gaussian_level, 'b n_h n_w -> b 1 n_h n_w 1 1')
    poisson_level_exp = rearrange(poisson_level, 'b n_h n_w -> b 1 n_h n_w 1 1')
    patches_noisy = torch.poisson(patches * poisson_level_exp) / poisson_level_exp
    gaussian_noise = torch.normal(mean=torch.zeros_like(patches_noisy),
                                  std=(gaussian_level_exp / 255))
    patches_noisy = patches_noisy + gaussian_noise
    patches_noisy = torch.clamp(patches_noisy, 0, 1)
    noisy_img_padded = rearrange(patches_noisy, 'b c n_h n_w new_h new_w -> b c (n_h new_h) (n_w new_w)')
    noisy_img = noisy_img_padded[:, :, :H, :W]
    noisy_img = torch.poisson(noisy_img * config['lam_p']) / config['lam_p']
    noisy_img = torch.clamp(noisy_img, 0, 1)
    return noisy_img


def FM2S(raw_noisy_img, config):
    """Denoise a single 2D slice using FM2S. Input/output are numpy float32 arrays."""
    raw_noisy_img = raw_noisy_img / 255.0
    clean_img = nd.median_filter(raw_noisy_img, config['w_size'])
    clean_img = torch.tensor(clean_img, dtype=torch.float32, device=device)
    clean_img = clean_img.unsqueeze(0).unsqueeze(0).repeat(1, config['n_chan'], 1, 1)
    raw_noisy_t = torch.tensor(raw_noisy_img, dtype=torch.float32, device=device)
    raw_noisy_t = raw_noisy_t.unsqueeze(0).unsqueeze(0).repeat(1, config['n_chan'], 1, 1)

    model = FM2SNetwork(config['n_chan']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(5):
        pred = model(raw_noisy_t)
        loss = criterion(pred, clean_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for _ in range(train_num):
        noisy_img = noise_addition(clean_img, config)
        for _ in range(max_epoch):
            pred = model(noisy_img)
            loss = criterion(pred, clean_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        denoised_img = model(raw_noisy_t)
    denoised_img = torch.clamp(denoised_img, 0, 1) * 255.0
    denoised_img = torch.mean(denoised_img, dim=1).squeeze()
    return denoised_img.cpu().numpy().astype(np.float32)


def apply_fm2s_to_volume(volume, config):
    """Apply FM2S slice-by-slice to a 3D volume (D, H, W)."""
    D, H, W = volume.shape
    result = np.zeros_like(volume, dtype=np.float32)
    vol_clipped = np.clip(volume, 0, 255).astype(np.float32)

    for z in range(D):
        print(f"  FM2S slice {z+1}/{D}...")
        result[z] = FM2S(vol_clipped[z], config)

    return result


# ===================== N2V Sliding Window =====================
def predict_sliding_window(model, volume):
    d, h, w = volume.shape
    td, th, tw = TILE_SIZE
    od, oh, ow = OVERLAP
    sd, sh, sw = td - od, th - oh, tw - ow

    counts = np.zeros(volume.shape, dtype=np.float32)
    prediction = np.zeros(volume.shape, dtype=np.float32)

    p3, p97 = np.percentile(volume, [3, 97])
    scale = p97 - p3 + 1e-6
    volume_norm = (volume - p3) / scale

    for z in range(0, d, sd):
        for y in range(0, h, sh):
            for x in range(0, w, sw):
                z_end = min(z + td, d)
                y_end = min(y + th, h)
                x_end = min(x + tw, w)
                z_start = max(0, z_end - td)
                y_start = max(0, y_end - th)
                x_start = max(0, x_end - tw)

                crop = volume_norm[z_start:z_end, y_start:y_end, x_start:x_end]
                inp = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).float().to(device)

                with torch.no_grad():
                    out = model(inp).cpu().numpy()[0, 0]

                prediction[z_start:z_end, y_start:y_end, x_start:x_end] += out
                counts[z_start:z_end, y_start:y_end, x_start:x_end] += 1.0

    prediction /= counts
    prediction = (prediction * scale) + p3
    return prediction.astype(np.float32)


# ===================== TEMPLATE FUNCTIONS (UNCHANGED) =====================
def run():
    interface_key = get_interface_key()
    # handler = {
    #     ("stacked-neuron-images-with-noise",): interf0_handler,
    # }[interface_key]
    handler = interf0_handler
    return handler()


def interf0_handler():
    # Read the input (unchanged)
    input_stacked_neuron_images_with_noise = load_image_file_as_array(
        location=INPUT_PATH / "images/stacked-neuron-images-with-noise",
    )

    _show_torch_cuda_info()

    # --- Load N2V model ---
    n2v_model = None
    try:
        n2v_model = UNet3D().to(device)
        n2v_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        n2v_model.eval()
        print("N2V model loaded.")
    except Exception as e:
        print(f"CRITICAL: Model load failed: {e}")
        traceback.print_exc()

    img = input_stacked_neuron_images_with_noise.astype(np.float32)

    if n2v_model is not None:
        # Stage 1: N2V 3D denoising
        print("Stage 1: N2V denoising...")
        denoised = predict_sliding_window(n2v_model, img)

        # Stage 2: FM2S slice-by-slice refinement
        print("Stage 2: FM2S refinement...")
        output_stacked_neuron_images_with_reduced_noise = apply_fm2s_to_volume(denoised, FM2S_CONFIG)
    else:
        print("FALLBACK: No model loaded, passing input through.")
        output_stacked_neuron_images_with_reduced_noise = img

    # Save your output (unchanged)
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/stacked-neuron-images-with-reduced-noise",
        array=output_stacked_neuron_images_with_reduced_noise,
    )

    return 0


def get_interface_key():
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])
    return SimpleITK.GetArrayFromImage(result)


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".tif"
    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
