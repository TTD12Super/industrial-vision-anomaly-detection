"""Pinned AnomalyCLIP single-process session used by the inspection localizer."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
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


class AnomalyClipSession:
    """Load upstream code/checkpoint once, then infer each routed crop in memory."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self.repository = Path(settings["repository"])
        self.checkpoint = Path(settings["checkpoint"])
        self.device = select_device(str(settings.get("device", "auto")))
        self.image_size = int(settings.get("image_size", 518))
        self.features_list = [int(value) for value in settings.get("features_list", [6, 12, 18, 24])]
        self.feature_map_layer = int(settings.get("feature_map_layer", 0))
        self.dpam_layer = int(settings.get("dpam_layer", 20))
        self.sigma = float(settings.get("sigma", 4.0))
        self.model: Any | None = None
        self.preprocess: Any | None = None
        self.text_features: torch.Tensor | None = None
        self._attention_hooks: list[Any] = []
        self._captured_attention: dict[int, torch.Tensor] = {}

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.repository.is_dir() or not self.checkpoint.is_file():
            raise FileNotFoundError("AnomalyCLIP repository or checkpoint is missing")
        sys.path.insert(0, str(self.repository))
        compatibility = types.ModuleType("pkg_resources")
        import packaging

        compatibility.packaging = packaging
        sys.modules.setdefault("pkg_resources", compatibility)
        import AnomalyCLIP_lib
        from prompt_ensemble import AnomalyCLIP_PromptLearner
        from utils import get_transform

        parameters = {
            "Prompt_length": int(self.settings.get("n_ctx", 12)),
            "learnabel_text_embedding_depth": int(self.settings.get("depth", 9)),
            "learnabel_text_embedding_length": int(self.settings.get("t_n_ctx", 4)),
        }
        model, _ = AnomalyCLIP_lib.load(
            "ViT-L/14@336px",
            device=self.device,
            design_details=parameters,
            download_root=str(Path(self.settings.get("clip_cache", "artifacts/external/clip"))),
        )
        model.eval()
        prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), parameters)
        checkpoint = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        prompt_learner.to(self.device).eval()
        model.to(self.device).eval()
        model.visual.DAPM_replace(DPAM_layer=self.dpam_layer)
        for index, block in enumerate(model.visual.transformer.resblocks):
            attention = getattr(block, "attn", None)
            if not hasattr(attention, "qkv"):
                continue
            self._attention_hooks.append(attention.register_forward_pre_hook(self._make_attention_capture(index)))
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        args = types.SimpleNamespace(image_size=self.image_size)
        self.preprocess, _ = get_transform(args)
        self.model = model
        self._library = AnomalyCLIP_lib

    def _make_attention_capture(self, index: int):
        def capture(module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0]
            batch, tokens, channels = x.shape
            qkv = module.qkv(x).reshape(batch, tokens, 3, module.num_heads, channels // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            self._captured_attention[index] = ((q @ k.transpose(-2, -1)) * module.scale).softmax(dim=-1).detach()

        return capture

    def infer(self, crop: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Return baseline map plus the validated raw tensors needed by Relation V2."""

        self._load()
        assert self.model is not None and self.preprocess is not None and self.text_features is not None
        self._captured_attention.clear()
        with Image.fromarray(crop.astype(np.uint8), mode="RGB") as image:
            pixels = self.preprocess(image).reshape(1, 3, self.image_size, self.image_size).to(self.device)
        with torch.inference_mode():
            image_features, patch_features = self.model.encode_image(pixels, self.features_list, DPAM_layer=self.dpam_layer)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            probabilities = (image_features @ self.text_features.permute(0, 2, 1) / 0.07).softmax(-1)[:, 0, 1]
            maps = []
            for index, patch_feature in enumerate(patch_features):
                if index >= self.feature_map_layer:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = self._library.compute_similarity(patch_feature, self.text_features[0])
                    similarity_map = self._library.get_similarity_map(similarity[:, 1:, :], self.image_size)
                    maps.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0)
            if not maps:
                raise RuntimeError("AnomalyCLIP produced no selected patch maps")
            heatmap = torch.stack(maps).sum(dim=0)[0].detach().float().cpu().numpy()
            if not self._captured_attention:
                raise RuntimeError("AnomalyCLIP did not expose q-k attention tensors")
            attention_layers = [self._captured_attention[index] for index in sorted(self._captured_attention)][-5:]
            attention = torch.stack([layer.mean(dim=1)[0] for layer in attention_layers]).mean(dim=0)
            features = patch_features[-1][0]
            if features.shape[0] != attention.shape[0]:
                raise RuntimeError(f"Relation token mismatch: features={tuple(features.shape)} attention={tuple(attention.shape)}")
            attention = attention[1:, 1:].float().cpu().numpy()
            features = features[1:].float().cpu().numpy()
        return gaussian_filter(heatmap, sigma=self.sigma).astype(np.float32), float(probabilities.item()), attention, features

    def localize(self, crop: np.ndarray) -> tuple[np.ndarray, float]:
        heatmap, score, _, _ = self.infer(crop)
        return heatmap, score
