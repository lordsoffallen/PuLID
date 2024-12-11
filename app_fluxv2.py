import time

import torch
from einops import rearrange
from PIL import Image
from dataclasses import dataclass

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.sampling import prepare_fill
from flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    configs,
)
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    lora_scale: float | None

    # Extras for flux tools
    img_cond_path: str = None
    img_mask_path: str = None


def get_models(name: str, device: torch.device, offload: bool, max_length: int = 128):
    t5 = load_t5(device, max_length=max_length)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    model.eval()
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip


class FluxGenerator:
    def __init__(self, model_name: str, device: str, offload: bool, aggressive_offload: bool):
        self.device = torch.device(device)
        self.offload = offload
        self.aggressive_offload = aggressive_offload
        self.model_name = model_name
        self.model, self.ae, self.t5, self.clip = get_models(
            model_name,
            device=self.device,
            offload=self.offload,
        )
        self.pulid_model = PuLIDPipeline(
            self.model, 
            device="cpu" if offload else device, 
            weight_dtype=torch.bfloat16, 
            onnx_provider="gpu"
        )
        if offload:
            self.pulid_model.face_helper.face_det.mean_tensor = self.pulid_model.face_helper.face_det.mean_tensor.to(torch.device("cuda"))
            self.pulid_model.face_helper.face_det.device = torch.device("cuda")
            self.pulid_model.face_helper.device = torch.device("cuda")
            self.pulid_model.device = torch.device("cuda")
        self.pulid_model.load_pretrain(pretrain_path=None, version="v0.9.1")

    @torch.inference_mode()
    def generate_image(
        self,
        width,
        height,
        num_steps,
        start_step,
        guidance,
        seed,
        prompt,
        id_image=None,
        id_weight=1.0,
        neg_prompt="",
        true_cfg=1.0,
        timestep_to_start_cfg=1,
        max_sequence_length=512,
    ):
        self.t5.max_length = max_sequence_length

        seed = int(seed)
        if seed == -1:
            seed = None

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{opts.prompt}' with seed {opts.seed}")
        t0 = time.perf_counter()

        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=True,
        )

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        inp_neg = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt) if use_true_cfg else None

        # offload TEs to CPU, load processor models and id encoder to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.pulid_model.components_to_device(torch.device("cuda"))

        if id_image is not None:
            id_image = resize_numpy_image_long(id_image, 1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(id_image, cal_uncond=use_true_cfg)
        else:
            id_embeddings = None
            uncond_id_embeddings = None

        # offload processor models and id encoder to CPU, load dit model to gpu
        if self.offload:
            self.pulid_model.components_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()
            if self.aggressive_offload:
                self.model.components_to_gpu()
            else:
                self.model = self.model.to(self.device)

        # denoise initial noise
        x = denoise(
            self.model, **inp, timesteps=timesteps, guidance=opts.guidance, id=id_embeddings, id_weight=id_weight,
            start_step=start_step, uncond_id=uncond_id_embeddings, true_cfg=true_cfg,
            timestep_to_start_cfg=timestep_to_start_cfg,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
            aggressive_offload=self.aggressive_offload,
        )

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")
        # bring into PIL format
        x = x.clamp(-1, 1)
        # x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img, str(opts.seed), self.pulid_model.debug_img_list


class FluxFillGenerator:
    def __init__(
        self, 
        model_name: str = "flux-dev-fill", 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        offload: bool = False, 
        aggressive_offload: bool = False
    ):
        self.device = torch.device(device)
        self.offload = offload
        self.aggressive_offload = aggressive_offload
        self.model_name = model_name

        if model_name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Got unknown model name: {model_name}, chose from {available}")

        self.model, self.ae, self.t5, self.clip = get_models(
            model_name,
            device=self.device,
            offload=self.offload,
            max_length=128,
        )
        self.pulid_model = PuLIDPipeline(
            self.model, 
            device="cpu" if offload else device, 
            weight_dtype=torch.bfloat16, 
            onnx_provider="gpu"
        )
        if offload:
            self.pulid_model.face_helper.face_det.mean_tensor = self.pulid_model.face_helper.face_det.mean_tensor.to(torch.device("cuda"))
            self.pulid_model.face_helper.face_det.device = torch.device("cuda")
            self.pulid_model.face_helper.device = torch.device("cuda")
            self.pulid_model.device = torch.device("cuda")
        self.pulid_model.load_pretrain(pretrain_path=None, version="v0.9.1")

    @torch.inference_mode()
    def fill_image(
        self,
        img_cond_path,
        img_mask_path,
        seed,
        num_steps=50,
        start_step=2,
        guidance=30.0,
        prompt=None,
        id_image=None,
        id_weight=1.0,
        neg_prompt="",
        true_cfg=1.0,
        timestep_to_start_cfg=1,
        max_sequence_length=512,
    ):
        self.t5.max_length = max_sequence_length

        seed = int(seed)
        if seed == -1:
            seed = None
        
        with Image.open(img_cond_path) as img:
            width, height = img.size

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
            img_cond_path=img_cond_path,
            img_mask_path=img_mask_path,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{opts.prompt}' with seed {opts.seed}")
        t0 = time.perf_counter()

        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        
        inp = prepare_fill(
            self.t5,
            self.clip,
            x,
            prompt=opts.prompt,
            ae=self.ae,
            img_cond_path=opts.img_cond_path,
            mask_path=opts.img_mask_path,
        )
        inp_neg = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt) if use_true_cfg else None

        timesteps = get_schedule(
            opts.num_steps, 
            inp["img"].shape[1], 
            shift=(name != "flux-schnell")
        )

        # offload TEs to CPU, load processor models and id encoder to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.pulid_model.components_to_device(torch.device("cuda"))

        if id_image is not None:
            id_image = resize_numpy_image_long(id_image, 1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(id_image, cal_uncond=use_true_cfg)
        else:
            id_embeddings = None
            uncond_id_embeddings = None

        # offload processor models and id encoder to CPU, load dit model to gpu
        if self.offload:
            self.pulid_model.components_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()
            if self.aggressive_offload:
                self.model.components_to_gpu()
            else:
                self.model = self.model.to(self.device)

        # denoise initial noise
        x = denoise(
            self.model, 
            **inp, 
            timesteps=timesteps, 
            guidance=opts.guidance, 
            id=id_embeddings, 
            id_weight=id_weight,
            start_step=start_step, 
            uncond_id=uncond_id_embeddings, 
            true_cfg=true_cfg,
            timestep_to_start_cfg=timestep_to_start_cfg,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
            aggressive_offload=self.aggressive_offload,
        )

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")
        # bring into PIL format
        x = x.clamp(-1, 1)
        # x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img, str(opts.seed), self.pulid_model.debug_img_list


# def create_demo(
#     model_name: str, 
#     device: str = "cuda" if torch.cuda.is_available() else "cpu",
#     offload: bool = False, 
#     aggressive_offload: bool = False
# ):
#     generator = FluxGenerator(model_name, device, offload, aggressive_offload)

#     output_image, seed_output, intermediate_output = generator.generate_image(
#         width,
#         height,
#         num_steps,
#         start_step,
#         guidance,
#         seed,
#         prompt,
#         id_image=None,
#         id_weight=1.0,
#         neg_prompt="",
#         true_cfg=1.0,
#         timestep_to_start_cfg=1,
#         max_sequence_length=512,
#     )

# def run_fill():
#     generator = FluxFillGenerator()
#     output_image, seed_output, intermediate_output = generator.fill_image(
#         img_cond_path=,
#         img_mask_path=,
#         seed=239128774,
#         num_steps=50,
#         start_step,
#         guidance,
#         prompt,
#         id_image=None,
#         id_weight=1.0,
#         neg_prompt="",
#         true_cfg=1.0,
#         timestep_to_start_cfg=1,
#         max_sequence_length=128,
#     )
