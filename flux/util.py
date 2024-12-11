import json
import os
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft

from flux.model import Flux, FluxLoraWrapper, FluxParams
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.modules.conditioner import HFEmbedder



@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    lora_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        lora_path=None,
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        lora_path=None,
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-canny": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Canny-dev",
        repo_flow="flux1-canny-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_CANNY"),
        lora_path=None,
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-canny-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        lora_path=os.getenv("FLUX_DEV_CANNY_LORA"),
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-depth": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Depth-dev",
        repo_flow="flux1-depth-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_DEPTH"),
        lora_path=None,
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-depth-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        lora_path=os.getenv("FLUX_DEV_DEPTH_LORA"),
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fill": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        repo_flow="flux1-fill-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_FILL"),
        lora_path=None,
        params=FluxParams(
            in_channels=384,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_flow_model(name: str, device: str = "cuda", hf_download: bool = True, verbose: bool = False):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    lora_path = configs[name].lora_path

    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow, local_dir='models')

    with torch.device(device):
        if lora_path is not None:
            model = FluxLoraWrapper(params=configs[name].params).to(torch.bfloat16)
        else:
            model = Flux(configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        sd = optionally_expand_state_dict(model, sd)

        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if verbose:
            print_load_warning(missing, unexpected)
        
    if configs[name].lora_path is not None:
        print("Loading LoRA")
        lora_sd = load_sft(configs[name].lora_path, device=str(device))

        # loading the lora params + overwriting scale values in the norms
        missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)

        if verbose:
            print_load_warning(missing, unexpected)
    return model

# from XLabs-AI https://github.com/XLabs-AI/x-flux/blob/1f8ef54972105ad9062be69fe6b7f841bce02a08/src/flux/util.py#L330
def load_flow_model_quintized(name: str, device: str = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = 'models/flux-dev-fp8.safetensors'
    if (
        not os.path.exists(ckpt_path)
        and hf_download
    ):
        ckpt_path = hf_hub_download("XLabs-AI/flux-dev-fp8", "flux-dev-fp8.safetensors")
    json_path = hf_hub_download("XLabs-AI/flux-dev-fp8", 'flux_dev_quantization_map.json')

    model = Flux(configs[name].params).to(torch.bfloat16)

    print("Loading checkpoint")
    # load_sft doesn't support torch.device
    sd = load_sft(ckpt_path, device='cpu')
    with open(json_path) as f:
        quantization_map = json.load(f)
    print("Start a quantization process...")
    from optimum.quanto import requantize
    requantize(model, sd, quantization_map, device=device)
    print("Model is quantized!")
    return model


def load_t5(device: str = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("xlabs-ai/xflux_text_encoders", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str = "cuda") -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str = "cuda", hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if (
        not os.path.exists(ckpt_path)
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae, local_dir='models')

    # Loading the autoencoder
    print("Init AE")
    with torch.device(device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False)
        print_load_warning(missing, unexpected)
    return ae


def optionally_expand_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    """
    Optionally expand the state dict to match the model's parameters shapes.
    """
    for name, param in model.named_parameters():
        if name in state_dict:
            if state_dict[name].shape != param.shape:
                print(
                    f"Expanding '{name}' with shape {state_dict[name].shape} to model parameter with shape {param.shape}."
                )
                # expand with zeros:
                expanded_state_dict_weight = torch.zeros_like(param, device=state_dict[name].device)
                slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
                expanded_state_dict_weight[slices] = state_dict[name]
                state_dict[name] = expanded_state_dict_weight

    return state_dict
