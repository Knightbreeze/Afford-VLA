# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import numpy as np
import PIL
import torch
from torchvision.transforms import v2

from sam3.model.data_misc import FindStage


class Sam3AffProcessor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = v2.functional.to_image(image).to(self.device)
        image = self.transform(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                sam2_backbone_out["backbone_fpn"][0]
            )
            sam2_backbone_out["backbone_fpn"][1] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                sam2_backbone_out["backbone_fpn"][1]
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: list[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(images[0], PIL.Image.Image), "Images must be a list of PIL images"

        batch_size = len(images)
        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [self.transform(v2.functional.to_image(image).to(self.device)) for image in images]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                sam2_backbone_out["backbone_fpn"][0]
            )
            sam2_backbone_out["backbone_fpn"][1] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                sam2_backbone_out["backbone_fpn"][1]
            )
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: dict):
        """Sets the text prompt and run the inference"""

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def set_text_prompt_batch(self, prompts: list[str], states: dict):
        """Sets the text prompts for batch processing and run the inference"""

        if "backbone_out" not in states:
            raise ValueError("You must call set_image_batch before set_text_prompt_batch")

        text_outputs = self.model.backbone.forward_text(prompts, device=self.device)
        states["backbone_out"].update(text_outputs)

        if "geometric_prompt" not in states:
            states["geometric_prompt"] = self.model._get_dummy_prompt(num_prompts=len(prompts))

        return self._forward_grounding_batch(states)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: list, label: bool, state: dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(["visual"], device=self.device)
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: dict):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        return outputs

    @torch.inference_mode()
    def _forward_grounding_batch(self, state: dict):
        batch_size = len(state["original_heights"])

        find_stage_batch = FindStage(
            img_ids=torch.arange(batch_size, device=self.device, dtype=torch.long),
            text_ids=torch.arange(batch_size, device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        geometric_prompt = state["geometric_prompt"]

        batch_outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_stage_batch,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

        return batch_outputs
