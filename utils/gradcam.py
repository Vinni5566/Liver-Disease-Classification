"""Architecture-agnostic, hook-based Grad-CAM utilities for PyTorch models."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the final spatial Conv2d layer without architecture-specific names."""
    conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No Conv2d layer was found in the model.")
    return conv_layers[-1]


class GradCAM:
    """Generate a single-image Grad-CAM explanation without third-party packages."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or find_last_conv_layer(model)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle = None
        self._backward_handle = None

    def _save_activations(self, _module, _inputs, output) -> None:
        # Grad-CAM operates on convolutional feature maps A^k with shape [B, C, H, W].
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            shape = getattr(output, "shape", None)
            raise RuntimeError(
                "The Grad-CAM target layer must produce a 4D [B, C, H, W] activation tensor; "
                f"got {shape!r}."
            )
        self.activations = output

    def _save_gradients(self, _module, _grad_inputs, grad_outputs) -> None:
        gradients = grad_outputs[0] if grad_outputs else None
        if not isinstance(gradients, torch.Tensor) or gradients.ndim != 4:
            shape = getattr(gradients, "shape", None)
            raise RuntimeError(
                "The Grad-CAM target layer must receive 4D [B, C, H, W] output gradients; "
                f"got {shape!r}."
            )
        self.gradients = gradients

    def _register_hooks(self) -> None:
        self.close()
        self.activations = None
        self.gradients = None
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def __call__(self, image_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """Return normalized heatmap, target class index, and target-class confidence.

        With ``target_class=None``, the target is the model's predicted class. This
        implementation accepts one image at a time so its returned heatmap is unambiguous.
        """
        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError("GradCAM expects one input image with shape [1, C, H, W].")

        self._register_hooks()
        was_training = self.model.training
        try:
            # eval() fixes dropout/BatchNorm inference behaviour; autograd still works in eval mode.
            self.model.eval()
            self.model.zero_grad(set_to_none=True)
            input_for_cam = image_tensor.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                logits = self.model(input_for_cam)
                if logits.ndim != 2 or logits.shape[0] != 1:
                    raise RuntimeError("GradCAM expects model logits with shape [1, num_classes].")
                probabilities = torch.softmax(logits, dim=1)
                if target_class is None:
                    target_class = int(probabilities.argmax(dim=1).item())
                if not 0 <= target_class < logits.shape[1]:
                    raise ValueError(f"target_class must be in [0, {logits.shape[1] - 1}].")
                confidence = float(probabilities[0, target_class].item())

                # Differentiate the target-class score y^c with respect to every feature-map pixel.
                logits[0, target_class].backward()

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

            # Global-average-pool gradients spatially: alpha_k^c = mean_ij(dy^c / dA_ij^k).
            channel_weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            # Form the weighted activation map sum_k(alpha_k^c * A^k), then keep positive evidence.
            cam = torch.relu((channel_weights * self.activations).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(
                cam, size=input_for_cam.shape[-2:], mode="bilinear", align_corners=False
            )
            heatmap = np.nan_to_num(cam[0, 0].detach().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
            # Normalize to [0, 1]; a zero-valued CAM remains an all-zero heatmap safely.
            maximum = float(heatmap.max())
            heatmap = heatmap / maximum if maximum > 0.0 else np.zeros_like(heatmap)
            return heatmap, target_class, confidence
        finally:
            self.close()
            self.model.train(was_training)

    def close(self) -> None:
        """Remove any active hook handles; safe to call repeatedly."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an RGB image and return RGB output."""
    heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, colored, alpha, 0)
