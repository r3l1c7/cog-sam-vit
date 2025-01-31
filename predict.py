# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, BaseModel, Input, Path
import numpy as np
import cv2
import torch
import tempfile

from segment_anything import sam_model_registry, SamPredictor

class Predictor(BasePredictor):
    def setup(self) -> None:
        """
        Load the SAM model into GPU memory so subsequent predict calls run quickly.
        """
        checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device='cuda')
        self.predictor = SamPredictor(sam)

    def predict(
        self,
        source_image: Path = Input(
            description="Input image file"
        ),
        box_prompt: str = Input(
            description=(
                "Optional bounding box in the format 'x_min,y_min,x_max,y_max' (no spaces). "
                "If not provided, no box prompt is used."
            ),
            default=None
        ),
        point_coords: str = Input(
            description=(
                "Optional list of point coords in the format 'x1,y1;x2,y2' (no spaces). "
                "If not provided, no point coords are used."
            ),
            default=None
        ),
        point_labels: str = Input(
            description=(
                "Optional list of point labels in the format 'l1;l2', each label is 0 or 1. "
                "If not provided, no point labels are used."
            ),
            default=None
        ),
        multimask_output: bool = Input(
            description="If True, SAM returns multiple masks. We pick the best one by highest score.",
            default=True
        ),
    ) -> Path:
        """
        Produce a mask from the specified bounding box or point prompts (or both).
        Return the best mask as a PNG file path.
        """
        # --- Load the image ---
        image_bgr = cv2.imread(str(source_image))
        if image_bgr is None:
            raise ValueError("Could not open or read the image file.")

        # Convert the image from BGR (OpenCV) to RGB (Segment Anything expects RGB).
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # --- Set the image in SAM Predictor ---
        self.predictor.set_image(image_rgb)

        # --- Parse the bounding box prompt if provided ---
        box = None
        if box_prompt:
            # Expecting format: "x_min,y_min,x_max,y_max"
            coords = list(map(int, box_prompt.split(",")))
            if len(coords) != 4:
                raise ValueError("box_prompt must have exactly 4 integers (x_min,y_min,x_max,y_max).")
            box = np.array(coords, dtype=np.float32)

        # --- Parse the point coords/labels if provided ---
        pts_coords = None
        pts_labels = None
        if point_coords and point_labels:
            # Expecting format: "x1,y1;x2,y2" for coords
            # and matching "l1;l2" for labels
            coords_strs = point_coords.split(";")
            label_strs = point_labels.split(";")
            if len(coords_strs) != len(label_strs):
                raise ValueError("point_coords and point_labels must have the same number of entries.")

            pts_coords = []
            for cs in coords_strs:
                x_str, y_str = cs.split(",")
                pts_coords.append([float(x_str), float(y_str)])
            pts_coords = np.array(pts_coords, dtype=np.float32)

            pts_labels = list(map(int, label_strs))
            pts_labels = np.array(pts_labels, dtype=np.int32)

        # --- Run SAM Predictor with the provided prompts ---
        # `predict` can handle (box=..., point_coords=..., point_labels=..., multimask_output=...).
        masks, scores, logits = self.predictor.predict(
            box=box,
            point_coords=pts_coords,
            point_labels=pts_labels,
            multimask_output=multimask_output,
        )
        
        if masks is None or len(masks) == 0:
            raise ValueError("No masks were returned by Segment Anything.")

        # --- If multiple masks are returned, pick the best by highest score ---
        if multimask_output and len(scores) > 1:
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
        else:
            best_mask = masks[0]

        # best_mask is a boolean or float32 array of shape [H, W].
        # Convert it to 0..255 uint8 image:
        mask_255 = (best_mask * 255).astype(np.uint8)

        # --- Write out to a temporary PNG file ---
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            save_path = tmp_file.name
            cv2.imwrite(save_path, mask_255)

        # Return the path so Cog/Replicate can serve it
        return Path(save_path)
