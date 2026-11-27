"""CLIP backbone and prompt-based classification utilities for CLIP-CoMA."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import open_clip


class CLIPClassifier:
    """A CLIP model that classifies images by image-text similarity.

    The text tower is frozen. The visual tower is trainable. No linear
    classification head is added; class names are represented by text prompts.
    """

    def __init__(self, device: torch.device | str, model_name: str = "ViT-B-32-quickgelu",
                 pretrained: str = "openai", prompt_template: str = "a photo of a {}."):
        self.device = torch.device(device)
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.visual.parameters():
            parameter.requires_grad_(True)

        self.model.eval()

    def train_mode(self):
        self.model.eval()
        self.model.visual.train()

    @torch.no_grad()
    def make_text_features(self, class_names: list[str]) -> torch.Tensor:
        prompts = [self.prompt_template.format(name.replace("_", " ")) for name in class_names]
        tokens = self.tokenizer(prompts).to(self.device)
        features = self.model.encode_text(tokens)
        return F.normalize(features.float(), dim=-1)

    def image_text_logits(self, images: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1)
        scale = self.model.logit_scale.exp().float()
        return scale * image_features @ text_features.T

    def trainable_parameters(self):
        return [p for p in self.model.visual.parameters() if p.requires_grad]
