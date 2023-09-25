import copy
import os
import shutil

import click
import cv2
import joblib
import numpy as np
import PIL
import torch
import tqdm
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
    seed,
    n_prompts,
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


@click.command()
@click.option("--seed", default=GLOBAL_SEED, help="Global seed.")
@click.option("--image_url", required=True, help="Image url")
@click.option("--output_dir", default="/shared/controlnet_output_images/")
@click.option(
    "--n_prompts_per_inference", default=5, help="Number of images to feed model."
)
def main(
    seed: int, image_url: str, output_dir: str, n_prompts_per_inference: int
) -> None:
    md5: str = joblib.hash((seed, image_url))
    n_seeds: int = 4
    n_steps: int = 40
    save_dir: str = os.path.join(output_dir, md5)
    subjects: list = [
        "style of Andrew Ferez",
        "style of Hugh Ferriss",
        "style of Filip Hodas",
        "style of Atelier Olschinsky",
        "style of Hieronymus Bosch",
        "style of Leonardo da Vinci",
        "style of Hishikawa Moronobu",
        "style of Hokusai",
        "style of John Constable",
        "style of Alexander Nasmyth",
        "style of Utagawa Kunisada",
        "style of William Morris",
        "style of Edvard Munch",
        "style of Nicholas Roerich",
        "style of Vincent van Gogh",
        "style of Winsor McCay",
        "style of Gary Larson",
        "style of Simon Stalenhag",
        "style of Angus McKie",
        "style of Wes Anderson",
        "style of Hayao Miyazaki",
        "style of Hannah Yata",
        "style of Hiroshi Yoshida",
        "Keanu Reeves portrait photo of a asia old warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes, 50mm portrait photography, hard rim lighting photography–beta –ar 2:3 –beta –upbeta –beta –upbeta –beta –upbeta",
        "city made out of glass : : close shot : : 3 5 mm, realism, octane render, 8 k, exploration, cinematic, trending on artstation, realistic, 3 5 mm camera, unreal engine, hyper detailed, photo – realistic maximum detail, volumetric light, moody cinematic epic concept art, realistic matte painting, hyper photorealistic, concept art, volumetric light, cinematic epic, octane render, 8 k, corona render, movie concept art, octane render, 8 k, corona render, cinematic, trending on artstation, movie concept art, cinematic composition, ultra – detailed, realistic, hyper – realistic, volumetric lighting, 8 k",
        "Residential home high end futuristic interior, olson kundig::1 Interior Design by Dorothy Draper, maison de verre, axel vervoordt::2 award winning photography of an indoor-outdoor living library space, minimalist modern designs::1 high end indoor/outdoor residential living space, rendered in vray, rendered in octane, rendered in unreal engine, architectural photography, photorealism, featured in dezeen, cristobal palma::2.5 chaparral landscape outside, black surfaces/textures for furnishings in outdoor space::1 –q 2 –ar 4:7",
        "Realistic architectural rendering of a capsule multiple house within concrete giant blocks with moss and tall rounded windows with lights in the interior, human scales, fog like london, in the middle of a contemporary city of Tokyo, stylish, generative design, nest, spiderweb structure, silkworm thread patterns, realistic, Designed based on Kengo Kuma, Sou Fujimoto, cinematic, unreal engine, 8K, HD, volume twilight –ar 9:54",
        "a lone skyscraper landscape vista photography by Carr Clifton & Galen Rowell, 16K resolution, Landscape veduta photo by Dustin Lefevre & tdraw, 8k resolution, detailed landscape painting by Ivan Shishkin, DeviantArt, Flickr, rendered in Enscape, Miyazaki, Nausicaa Ghibli, Breath of The Wild, 4k detailed post processing, atmospheric, hyper realistic, 8k, epic composition, cinematic, artstation –w 1024 –h 1280",
        "Garden+factory,Tall factory,Many red rose,A few roses,clouds, ultra wide shot, atmospheric, hyper realistic, 8k, epic composition, cinematic, octane render, artstation landscape vista photography by Carr Clifton & Galen Rowell, 16K resolution, Landscape veduta photo by Dustin Lefevre & tdraw, 8k resolution, detailed landscape painting by Ivan Shishkin, DeviantArt, Flickr, rendered in Enscape, Miyazaki, Nausicaa Ghibli, Breath of The Wild, 4k detailed post processing, artstation, rendering by octane, unreal –hd –ar 9:16",
        "The Legend of Zelda landscape atmospheric, hyper realistic, 8k, epic composition, cinematic, octane render, artstation landscape vista photography by Carr Clifton & Galen Rowell, 16K resolution, Landscape veduta photo by Dustin Lefevre & tdraw, 8k resolution, detailed landscape painting by Ivan Shishkin, DeviantArt, Flickr, rendered in Enscape, Miyazaki, Nausicaa Ghibli, Breath of The Wild, 4k detailed post processing, artstation, rendering by octane, unreal engine —ar 16:9",
        "rough ocean storm atmospheric, hyper realistic, 8k, epic composition, cinematic, octane render, artstation landscape vista photography by Carr Clifton & Galen Rowell, 16K resolution, Landscape veduta photo by Dustin Lefevre & tdraw, 8k resolution, detailed landscape painting by Ivan Shishkin, DeviantArt, Flickr, rendered in Enscape, Miyazaki, Nausicaa Ghibli, Breath of The Wild, 4k detailed post processing, artstation, rendering by octane, unreal engine —ar 16:9",
        "Simplified technical drawing, Leonardo da Vinci, Mechanical Dinosaur Skeleton, Minimalistic annotations, Hand-drawn illustrations, Basic design and engineering, Wonder and curiosity",
        "a landscape from the Moon with the Earth setting on the horizon, realistic, detailed",
        "Isometric Atlantis city,great architecture with columns, great details, ornaments,seaweed, blue ambiance, 3D cartoon style, soft light, 45° view",
        "darth vadar",
        "jedi vs sith",
        "indiana jones",
        "A panel of marble with a marble text carved, lots of flowers carved in marble",
        "straight line PCB. gps navigation inside a pcb with installed components. A circuit board made of silicon, with processors and circuits arranged as text, lots of silicon",
        "Multiple layers of silhouette mountains, with silhouette of big rocket in sky, sharp edges, at sunset, with heavy fog in air, vector style, horizon silhouette Landscape wallpaper by Alena Aenami, firewatch game style, vector style background",
        "gold dia de los muertos pendant, intricate 2d vector geometric, cutout shape pendant, blueprint frame lines sharp edges, svg vector style, product studio shoot",
        #  "(retro volkswagen van:1.1) (attached to hot air balloon:1.2), (dark colors:1.2), snthwve style wallpaper",
        "Die-cut sticker, Cute kawaii flower character sticker, white background, illustration minimalism, vector, pastel colors",
        "cloudy sky background lush landscape house and green trees, RAW photo (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
        "cloudy sky background lush landscape house and trees illustration concept art anime key visual trending pixiv fanbox by wlop and greg rutkowski and makoto shinkai and studio ghibli",
        "masterpice, A beautiful jungle with lots of flowers ranbow landscape form zelda breath of the wild, soft sunlight, trees, high detailed grass",
        "style of Ansel Adams",
        "style of Neal Adams",
        "style of Giuseppe Arcimboldo",
        "a closeup macro photo of tiny cute Rainbow Dewdrops spider, (crystal:0. 1) spider (symmetry:0. 1), perfect spider forms",
        "MRI CT Scan",
        "WSI H&E slide",
    ]
    negative_prompt = "((bad artist)) ((low quality)) ((low details)), monochrome, all black, all one color, lowres, bad anatomy, worst quality, low quality"
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
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(GLOBAL_SEED)
    canny_image: PIL.Image.Image = get_canny_image(
        url=image_url,
        low_threshold=100,
        high_threshold=200,
    )
    images = []
    for start_i in tqdm.trange(
        0, len(subjects), n_prompts_per_inference, desc="prompt"
    ):
        end_i = start_i + n_prompts_per_inference
        for use_additional_prompts in tqdm.tqdm(
            [False, True], desc="use additional prompts"
        ):
            if use_additional_prompts:
                positive_prompts = [
                    ", ".join([subject, *additional_prompts])
                    for subject in subjects[start_i:end_i]
                ]
            else:
                positive_prompts = subjects[start_i:end_i]
            n_prompts: int = len(positive_prompts)
            negative_prompts = [negative_prompt] * n_prompts
            for use_negative in tqdm.tqdm([True, False], desc="use negative"):
                for seed in tqdm.tqdm(
                    np.random.randint(
                        low=np.iinfo(np.int32).min,
                        high=np.iinfo(np.int32).max,
                        size=n_seeds,
                        dtype=np.int32,
                    ).tolist(),
                    desc="seed",
                ):
                    output_images = inference(
                        positive_prompts=positive_prompts,
                        negative_prompts=negative_prompts
                        if use_negative
                        else [""] * n_prompts,
                        canny_image=canny_image,
                        n_steps=n_steps,
                        seed=seed,
                        n_prompts=n_prompts,
                    )
                    images += output_images
                    for prompt_i, image in zip(range(start_i, end_i), output_images):
                        image.save(
                            os.path.join(
                                save_dir,
                                (
                                    f"prompt-{prompt_i}-"
                                    f"use-additional-{use_additional_prompts}-"
                                    f"use-negative-{use_negative}-"
                                    f"seed-{seed}"
                                    f".png"
                                ),
                            )
                        )
    image_grid = create_image_grid(images, len(subjects) * 4, n_seeds)
    image_grid.save(os.path.join(save_dir, "output-images.png"))
    canny_image.save(os.path.join(save_dir, "canny-image.png"))
    print(f"saved data to: '{os.path.abspath(save_dir)}'")


if __name__ == "__main__":
    main()
