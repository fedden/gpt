import copy
import os
import shutil

import cv2
import numpy as np
import PIL
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler

CACHE_DIR: str = "/shared/huggingface/"
GLOBAL_SEED: int = 42


def get_canny_image(
    url: str,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> PIL.Image.Image:
    """Get image."""
    image = load_image(url)
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return PIL.Image.fromarray(image)


def create_image_grid(imgs: list, rows: int, cols: int) -> PIL.Image.Image:
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = PIL.Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i // rows * w, i % rows * h))
    return grid


def inference(
    positive_prompts,
    negative_prompts,
    canny_image,
    n_steps,
):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    generator = [
        torch.Generator(device="cpu").manual_seed(seed) for _ in range(n_prompts)
    ]
    output = pipe(
        positive_prompts,
        canny_image,
        negative_prompt=negative_prompts,
        num_inference_steps=n_steps,
        generator=generator,
    )
    return copy.deepcopy(output.images)


n_seeds: int = 4
n_steps: int = 20
root_save_dir: str = "output/"
image_urls = [
    # Square AZ logo only.
    "https://e7.pngegg.com/pngimages/374/967/png-clipart-astrazeneca-pharmaceutical-industry-medical-science-liaison-biologic-waltham-others-miscellaneous-angle.png",
    # Wide AZ logo with text image.
    "https://assets.stickpng.com/images/5847ebf8cef1014c0b5e4859.png",
]
subjects: list = [
    "A panel of marble with a marble text carved, lots of flowers carved in marble",
    "straight line PCB. gps navigation inside a pcb with installed components. A circuit board made of silicon, with processors and circuits arranged as text, lots of silicon",
    "Multiple layers of silhouette mountains, with silhouette of big rocket in sky, sharp edges, at sunset, with heavy fog in air, vector style, horizon silhouette Landscape wallpaper by Alena Aenami, firewatch game style, vector style background",
    "gold dia de los muertos pendant, intricate 2d vector geometric, cutout shape pendant, blueprint frame lines sharp edges, svg vector style, product studio shoot",
    #  "(retro volkswagen van:1.1) (attached to hot air balloon:1.2), (dark colors:1.2), snthwve style wallpaper",
    "Die-cut sticker, Cute kawaii flower character sticker, white background, illustration minimalism, vector, pastel colors",
]
negative_prompt = "monochrome, all black, all one color, lowres, bad anatomy, worst quality, low quality"
additional_prompts = [
    "best quality",
    "extremely detailed",
    "edge to edge",
    "fullscreen",
    "highest quality",
    "render",
    "HD",
    "UHD",
    "4K",
    "pro",
    "professional",
    "amazing",
]
positive_prompts = []
for subject in subjects:
    positive_prompts += [subject, ", ".join([subject, *additional_prompts])]
n_prompts: int = len(positive_prompts)
negative_prompts = [negative_prompt] * n_prompts
shutil.rmtree(root_save_dir, ignore_errors=True)
for url_i, url in enumerate(image_urls):
    save_dir: str = os.path.join(root_save_dir, f"image-{url_i}")
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(GLOBAL_SEED)
    canny_image: PIL.Image.Image = get_canny_image(
        url=url,
        low_threshold=100,
        high_threshold=200,
    )
    images = []
    for seed in np.random.randint(
        low=np.iinfo(np.int32).min,
        high=np.iinfo(np.int32).max,
        size=n_seeds,
        dtype=np.int32,
    ).tolist():
        output_images = inference(
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            canny_image=canny_image,
            n_steps=n_steps,
        )
        images += output_images
        for i, image in enumerate(output_images):
            image.save(os.path.join(save_dir, f"prompt-{i}-seed-{seed}.png"))
    image_grid = create_image_grid(images, n_prompts, n_seeds)
    image_grid.save(os.path.join(save_dir, "output-images.png"))
    canny_image.save(os.path.join(save_dir, "canny-image.png"))
    print(f"saved data to: '{os.path.abspath(save_dir)}'")
