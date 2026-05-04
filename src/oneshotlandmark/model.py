from __future__ import annotations
from transformers import AutoImageProcessor, AutoModel
import torch
import time
import logging
from collections import defaultdict
from tqdm import tqdm
from oneshotlandmark.utils import login_huggingface

logger = logging.getLogger(__name__)

class ViTModel:
    """
    Wrapper around a Vision Transformer model for generating patch-level embeddings.

    Args:
        model_id (str): HuggingFace model identifier.
        device_str (str, optional): Device to run the model on ('cuda', 'mps', 'cpu').
            If None, auto-detects the best available device.
        verbose (bool): If True, enables progress bars and timing logs.
    """

    DEFAULT_MODEL_ID="facebook/dinov3-vitb16-pretrain-lvd1689m"
    def __init__(self, model_id=DEFAULT_MODEL_ID, device_str=None, verbose=False):
        self.model_id = model_id
        self.verbose = verbose
        if device_str is None:
            self.device_str = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu").type
        else:
            self.device_str = device_str

        login_huggingface()
        self.model = self.__load_model()
        
        self.processor = self.__load_processor()
    
    def __load_model(self):
        """Load the pretrained model onto the target device in eval mode."""
        device = torch.device(self.device_str)
        model = AutoModel.from_pretrained(self.model_id, attn_implementation="sdpa")
        model = model.to(device)
        model.eval()
        logger.info(f"Loaded model '{self.model_id}' on {self.device_str}")
        return model

    def __load_processor(self):
        """Load the image processor associated with the model."""
        return AutoImageProcessor.from_pretrained(self.model_id)

    def generate_embedding(self, image, output_attentions=False):
        """
        Generate token-level embeddings (and optionally attention maps) for a single image.
 
        The image processor is applied without resizing or center-cropping, so the
        input image dimensions directly determine the number of patch tokens.
 
        Args:
            image (PIL.Image): Input image.
            output_attentions (bool): If True, also return the last-layer attention map.
 
        Returns:
            torch.Tensor: Hidden states of shape (num_tokens, hidden_dim) when
                output_attentions is False.
            tuple[torch.Tensor, torch.Tensor]: (hidden_states, attention_map) when
                output_attentions is True. Attention map is from the last transformer layer.
        """
        start = time.perf_counter()
        # Apply image processor
        inputs = self.processor( images=image,
                              do_resize=False,
                              do_center_crop=False,
                              return_tensors="pt",
                        )
        pixel_values = inputs["pixel_values"].to(self.device_str)
        
        with torch.no_grad():
            out = self.model(pixel_values=pixel_values, output_attentions=output_attentions)

        hidden_states = out.last_hidden_state.squeeze(0).detach().cpu()  # (T, D)
        del pixel_values

        if self.verbose:
            elapsed = time.perf_counter() - start
            logger.info(f"generate_embedding: {elapsed:.3f}s")
        
        if output_attentions:
            return hidden_states, out.attentions[-1]
        return hidden_states
    
    def generate_embedding_batch(self, images, batch_size=4, output_attentions=False):
        """
        Generate embeddings for a batch of images, grouped by dimensions.
 
        Images with identical (width, height) are batched together for efficient
        GPU inference. Images of different sizes are processed in separate batches,
        effectively falling back to sequential processing if all inputs are of different sizes.
 
        Args:
            images (list[PIL.Image]): Input images.
            batch_size (int): Maximum number of same-size images per forward pass.
            output_attentions (bool): If True, also return last-layer attention maps.
 
        Returns:
            list[torch.Tensor]: Per-image hidden states, each of shape (num_tokens, hidden_dim),
                when output_attentions is False.
            list[tuple[torch.Tensor, torch.Tensor]]: Per-image (hidden_states, attention_map)
                when output_attentions is True.
        """
        start = time.perf_counter()

        # Group by (width, height)
        size_groups = defaultdict(list)
        for idx, img in enumerate(images):
            size_groups[img.size].append((idx, img))

        # Process each group in batches
        results = [None] * len(images)

        all_batches = []
        for size, group in size_groups.items():
            for i in range(0, len(group), batch_size):
                all_batches.append(group[i:i + batch_size])
        
        iterator = tqdm(all_batches, desc="Generating embeddings", leave=False) if self.verbose else all_batches

        for batch in iterator:
                indices = [b[0] for b in batch]
                imgs = [b[1] for b in batch]
                
                inputs = self.processor(images=imgs, do_resize=False,
                                       do_center_crop=False, return_tensors="pt")
                
                pixel_values = inputs["pixel_values"].to(self.device_str)

                with torch.no_grad():
                    out = self.model(pixel_values=pixel_values, output_attentions=output_attentions)

                for j, orig_idx in enumerate(indices):
                    hidden = out.last_hidden_state[j].cpu()
                    if output_attentions:
                        attn = out.attentions[-1][j].cpu()
                        results[orig_idx] = (hidden, attn)
                    else:
                        results[orig_idx] = hidden
                
                del pixel_values, out
        
        if self.verbose:
            elapsed = time.perf_counter() - start
            logger.info(
                f"generate_embedding_batch: {len(images)} images in {elapsed:.3f}s "
                f"({len(size_groups)} size groups, {len(all_batches)} batches)"
            )
 
        return results
    
    def __repr__(self):
        return f"ViTModel(model_id='{self.model_id}', device='{self.device_str}')"