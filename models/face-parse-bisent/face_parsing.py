import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import requests

class FaceParser:
    def __init__(self, model_path="jonathandinu/face-parsing", device=None):
        """
        Initialize the FaceParser with a specified model path and device.

        Args:
            model_path (str): Path to the pretrained model.
            device (str, optional): Device to run the model on. Default is determined automatically.
        """
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.image_processor = SegformerImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)

    def parse(self, images):
        """
        Parse the input images to extract binary masks of the face regions.

        Args:
            images (List[PIL.Image.Image]): List of input images.

        Returns:
            List[np.ndarray]: List of binary masks for the face regions.
        """
        # Get width and height of images
        w, h = images[0].size
        
        # Run inference on images
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

        # Resize output to match input image dimensions
        upsampled_logits = nn.functional.interpolate(logits,
                                                     size=(h, w),
                                                     mode='bilinear',
                                                     align_corners=False)

        # Get label masks
        labels = upsampled_logits.argmax(dim=1)

        # Process each label mask
        face_masks = []
        for label in labels:
            # Set elements larger than 12 to 0
            label[label > 12] = 0

            # Set everything else to 1
            label[label != 0] = 1

            # Convert to numpy array
            face_mask = label.byte().cpu().numpy()
            face_masks.append(face_mask)

        return face_masks

if __name__ == '__main__':
    urls = [
        "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6",
        "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6",
        "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6",
        "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6"
    ]
    images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
    
    parser = FaceParser()
    face_masks = parser.parse(images)
    
    # Save and show the masks for verification
    for i, face_mask in enumerate(face_masks):
        Image.fromarray(face_mask * 255).save(f"face_parsing_{i}.png")