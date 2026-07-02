import torch
from typing import Dict, Tuple

class MCUncertaintyAnalyser:
    """
    Monte Carlo Dropout uncertainty estimation: runs `n_passes` stochastic
    forward passes with dropout kept active (model.train(), but since this
    UNet's norm layers are InstanceNorm without running stats, train()-mode
    norm behaves identically to eval()-mode norm, so only dropout becomes
    stochastic) and reports the mean prediction, voxelwise variance, and
    predictive entropy.
    """

    def __init__(self, model: torch.nn.Module, n_passes: int = 15,
                roi_size: Tuple[int, int, int] = (128, 128, 128),
                sw_batch_size: int = 4, overlap: float = 0.5):
        self.model = model
        self.n_passes = n_passes
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap

    @torch.no_grad()
    def analyse(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        from monai.inferers import sliding_window_inference

        # Keep dropout stochastic; InstanceNorm (no running stats) is
        # unaffected by train()/eval(), so this only randomizes dropout masks.
        self.model.train()
        probs_list = []
        for _ in range(self.n_passes):
            logits = sliding_window_inference(
                x, self.roi_size, self.sw_batch_size, self.model, overlap=self.overlap)
            probs_list.append(torch.sigmoid(logits))
        self.model.eval()

        stack = torch.stack(probs_list)            # (N, B, C, H, W, D)
        mean_pred = stack.mean(0)
        variance = stack.var(0)
        eps = 1e-6
        entropy = -(mean_pred * torch.log(mean_pred + eps)
                   + (1 - mean_pred) * torch.log(1 - mean_pred + eps))
        per_pass_entropy = -(stack * torch.log(stack + eps)
                            + (1 - stack) * torch.log(1 - stack + eps))
        mutual_info = (entropy - per_pass_entropy.mean(0)).clamp(min=0.0)
        return {"mean_pred": mean_pred, "variance": variance,
               "entropy": entropy, "mutual_info": mutual_info}

class TTAUncertaintyAnalyser:
    """
    Test-Time Augmentation (TTA) uncertainty estimation for 3D segmentation.
    
    Instead of using stochastic network weights (like MC Dropout), this locks the 
    model and applies 8 deterministic spatial flip combinations (X, Y, Z axes) 
    to the input volume. It runs inference, inverts the flips on the predictions, 
    and computes the variance/entropy across the augmented passes.
    """
    def __init__(self, model: torch.nn.Module, roi_size: tuple = (128, 128, 128),
                 sw_batch_size: int = 4, overlap: float = 0.5):
        self.model = model
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        
        # PyTorch 3D Image tensors are formatted as (Batch, Channel, X, Y, Z)
        # Therefore, the spatial dimensions are indices 2, 3, and 4.
        # These 8 tuples represent all possible flip combinations across the 3 spatial axes.
        self.flip_combinations = [
            (),               # 1. Original (No flips)
            (2,),             # 2. Flip X
            (3,),             # 3. Flip Y
            (4,),             # 4. Flip Z
            (2, 3),           # 5. Flip X, Y
            (2, 4),           # 6. Flip X, Z
            (3, 4),           # 7. Flip Y, Z
            (2, 3, 4)         # 8. Flip X, Y, Z
        ]

    @torch.no_grad()
    def analyse(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        from monai.inferers import sliding_window_inference # 1. Lock the model: No dropout, strictly deterministic
        self.model.eval()
        
        probs_list = []
        
        # 2. Iterate through all 8 augmentation states
        for dims in self.flip_combinations:
            
            # A. Apply spatial flip to the input image
            if len(dims) > 0:
                x_aug = torch.flip(x, dims)
            else:
                x_aug = x
                
            # B. Run inference on the augmented image
            logits = sliding_window_inference(
                x_aug, self.roi_size, self.sw_batch_size, self.model, overlap=self.overlap
            )
            
            # C. Invert the flip on the output mask
            # (A flip is its own inverse, so flipping the same dims restores original orientation)
            if len(dims) > 0:
                logits = torch.flip(logits, dims)
                
            # D. Convert logits to probabilities and store
            probs_list.append(torch.sigmoid(logits))

        # 3. Stack all 8 predictions: shape (8, B, C, X, Y, Z)
        stack = torch.stack(probs_list)
        
        # 4. Calculate Uncertainty Metrics (Variance & Entropy)
        mean_pred = stack.mean(0)
        variance = stack.var(0)
        
        eps = 1e-6
        # Total Predictive Entropy (Uncertainty of the mean prediction)
        entropy = -(mean_pred * torch.log(mean_pred + eps) + 
                   (1 - mean_pred) * torch.log(1 - mean_pred + eps))
                   
        # Aleatoric Entropy (Mean of the individual prediction entropies)
        per_pass_entropy = -(stack * torch.log(stack + eps) + 
                            (1 - stack) * torch.log(1 - stack + eps))
                            
        # Epistemic Uncertainty (Mutual Information = Total Entropy - Aleatoric Entropy)
        mutual_info = (entropy - per_pass_entropy.mean(0)).clamp(min=0.0)
        
        return {
            "mean_pred": mean_pred, 
            "variance": variance, 
            "entropy": entropy, 
            "mutual_info": mutual_info
        }
