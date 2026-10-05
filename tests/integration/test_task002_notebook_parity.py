"""Parity checks between the refactored encoder and notebook patch logic."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from mvtec_system import SharedDINOv2, image_tensor, select_device


def notebook_patch_features(
    backbone: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    padding: tuple[int, int],
    feature_key: str,
) -> torch.Tensor:
    """Mirror cells 30 and 34 in notebooks/02_first_model.ipynb."""

    images = images.to(device)
    pad_bottom, pad_right = padding
    images = F.pad(images, (0, pad_right, 0, pad_bottom), mode="replicate")
    backbone.eval()
    with torch.inference_mode():
        outputs = backbone.forward_features(images)
    if feature_key not in outputs:
        raise KeyError(f"The backbone did not return {feature_key}.")
    features = outputs[feature_key]
    return F.normalize(features.float(), p=2, dim=-1)


class NotebookParityTests(unittest.TestCase):
    def test_normalized_patch_features_match_notebook(self):
        artifact_root = Path("artifacts/dinov2_knn")
        checkpoint = artifact_root / "torch_hub/checkpoints/dinov2_vitb14_pretrain.pth"
        repository = artifact_root / "torch_hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8"
        if not checkpoint.is_file() or not repository.is_dir():
            self.skipTest("local pinned DINOv2 cache is not available")

        paths = [
            Path("datasets/mvtec_ad_processed/bottle/train/good/000.png"),
            Path("datasets/mvtec_ad_processed/bottle/train/good/001.png"),
        ]
        batch = torch.stack([image_tensor(path) for path in paths])
        encoder = SharedDINOv2(device=select_device())

        notebook_features = notebook_patch_features(
            encoder.backbone,
            batch,
            encoder.device,
            encoder.padding,
            encoder.config.patch_feature_key,
        )
        module_features = encoder.forward(batch).patch_embeddings

        self.assertEqual(tuple(module_features.shape), (2, 361, 768))
        self.assertEqual(tuple(notebook_features.shape), (2, 361, 768))
        self.assertTrue(torch.isfinite(module_features).all().item())
        self.assertTrue(torch.isfinite(notebook_features).all().item())
        self.assertTrue(
            torch.allclose(module_features, notebook_features, rtol=1e-5, atol=1e-6)
        )

        max_difference = torch.max(torch.abs(module_features - notebook_features)).item()
        print(f"Notebook parity max absolute difference: {max_difference:.3e}")


if __name__ == "__main__":
    unittest.main()
