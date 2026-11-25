#!/usr/bin/env python3
"""Run one pinned AnomalyCLIP crop and save its map as a portable NPZ artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types

import numpy as np
import packaging
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch


def select_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(args: argparse.Namespace) -> dict[str, object]:
    repository = Path(args.repository).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    image_path = Path(args.input).resolve()
    if not repository.is_dir() or not checkpoint.is_file() or not image_path.is_file():
        raise FileNotFoundError("repository, checkpoint and input image must exist")
    sys.path.insert(0, str(repository))
    # Upstream imports the removed setuptools pkg_resources.packaging alias.
    compatibility = types.ModuleType("pkg_resources")
    compatibility.packaging = packaging
    sys.modules.setdefault("pkg_resources", compatibility)
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    device = select_device(args.device)
    parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }
    model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px",
        device=device,
        design_details=parameters,
        download_root=str(Path(args.clip_cache).resolve()),
    )
    model.eval()
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), parameters)
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    prompt_learner.load_state_dict(checkpoint_data["prompt_learner"])
    prompt_learner.to(device).eval()
    model.to(device).eval()
    model.visual.DAPM_replace(DPAM_layer=args.dpam_layer)
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    preprocess, _ = get_transform(args)
    with Image.open(image_path) as source:
        image = preprocess(source.convert("RGB")).reshape(1, 3, args.image_size, args.image_size).to(device)
    with torch.inference_mode():
        image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=args.dpam_layer)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        probabilities = (image_features @ text_features.permute(0, 2, 1) / 0.07).softmax(-1)[:, 0, 1]
        maps = []
        for index, patch_feature in enumerate(patch_features):
            if index >= args.feature_map_layer:
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                maps.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0)
        if not maps:
            raise RuntimeError("AnomalyCLIP produced no selected patch maps")
        anomaly_map = torch.stack(maps).sum(dim=0)[0].detach().float().cpu().numpy()
    anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigma).astype(np.float32)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "repository": str(repository),
        "checkpoint": str(checkpoint),
        "device": device,
        "image_size": args.image_size,
        "features_list": args.features_list,
        "feature_map_layer": args.feature_map_layer,
        "dpam_layer": args.dpam_layer,
    }
    np.savez_compressed(output, heatmap=anomaly_map, anomaly_score=float(probabilities.item()), metadata=json.dumps(metadata))
    return {"output": str(output), "anomaly_score": float(probabilities.item()), **metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip-cache", default="artifacts/external/clip")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--features-list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n-ctx", type=int, default=12)
    parser.add_argument("--t-n-ctx", type=int, default=4)
    parser.add_argument("--feature-map-layer", type=int, default=0)
    parser.add_argument("--dpam-layer", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=4.0)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
