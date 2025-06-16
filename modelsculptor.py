import torch
import numpy as np
import folder_paths

class ModelSculptorFlux:
    GRADIENT_SHAPES = [
        "Linear (Ascending)",
        "Linear (Descending)",
        "Ease In (Quadratic)",
        "Ease Out (Quadratic)",
        "Ease In/Out (Sine)",
        "Spike (Gaussian)",
        "Dip (Inverse Gaussian)",
        "Steps (Ascending)",
        "Steps (Descending)",
        "Random (Noise)"
    ]
    
    TARGET_BLOCKS = [
        "all",
        "in_layers",
        "double_blocks",
        "single_blocks",
        "Double & Single (Synced Shape)"
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "gradient_shape": (s.GRADIENT_SHAPES,),
                "strength": ("FLOAT", {"default": 0.1, "min": -2.0, "max": 2.0, "step": 0.01}),
                "target_blocks": (s.TARGET_BLOCKS,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "sculpt_model"
    CATEGORY = "models/advanced"

    def _generate_gradient(self, shape, num_steps):
        if num_steps <= 1:
            return np.ones(num_steps) if num_steps == 1 else np.array([])
        x = np.linspace(0, 1, num_steps)
        if shape == "Linear (Ascending)":
            return x
        elif shape == "Linear (Descending)":
            return 1 - x
        elif shape == "Ease In (Quadratic)":
            return x * x
        elif shape == "Ease Out (Quadratic)":
            return 1 - (1 - x) * (1 - x)
        elif shape == "Ease In/Out (Sine)":
            return (np.sin((x - 0.5) * np.pi) + 1) / 2
        elif shape == "Spike (Gaussian)":
            return np.exp(-((x - 0.5) ** 2) / (2 * (0.15 ** 2)))
        elif shape == "Dip (Inverse Gaussian)":
            return 1 - np.exp(-((x - 0.5) ** 2) / (2 * (0.15 ** 2)))
        elif shape == "Steps (Ascending)":
            return np.floor(x * 10) / 10
        elif shape == "Steps (Descending)":
            return np.floor((1 - x) * 10) / 10
        elif shape == "Random (Noise)":
            return np.random.rand(num_steps)
        return np.zeros(num_steps)

    def sculpt_model(self, model, gradient_shape, strength, target_blocks):
        sculpted_model = model.clone()
        key_patches = model.get_key_patches("diffusion_model.")

        flux_architecture = {
            "in_layers": [f"diffusion_model.{p}" for p in ["img_in.", "time_in.", "guidance_in.", "vector_in.", "txt_in."]],
            "double_blocks": [f"diffusion_model.double_blocks.{i}." for i in range(19)],
            "single_blocks": [f"diffusion_model.single_blocks.{i}." for i in range(38)],
        }
        
        modified_patch_count = 0
        
        if target_blocks == "Double & Single (Synced Shape)":
            print(f"[Model Sculptor Flux] Applying synced shape '{gradient_shape}' with strength {strength}.")
            
            for block_key in ["double_blocks", "single_blocks"]:
                prefixes = flux_architecture[block_key]
                gradient = self._generate_gradient(gradient_shape, len(prefixes))
                print(f"  - Sculpting {len(gradient)} {block_key}.")
                
                for i, prefix in enumerate(prefixes):
                    gradient_ratio = gradient[i] * strength
                    
                    for patch_key in key_patches:
                        if patch_key.startswith(prefix):
                            sculpted_model.add_patches({patch_key: key_patches[patch_key]}, 1.0, gradient_ratio)
                            modified_patch_count += 1
        else:
            target_prefixes = []
            if target_blocks == "all":
                sorted_keys = ["in_layers", "double_blocks", "single_blocks"]
                for block_type in sorted_keys:
                    target_prefixes.extend(flux_architecture[block_type])
            elif target_blocks in flux_architecture:
                target_prefixes = flux_architecture[target_blocks]
            else:
                print(f"Warning: [Model Sculptor Flux] Unknown target block type '{target_blocks}'.")
                return (model,)

            gradient = self._generate_gradient(gradient_shape, len(target_prefixes))
            print(f"[Model Sculptor Flux] Sculpting model with shape '{gradient_shape}', strength {strength}, on {len(gradient)} '{target_blocks}' blocks.")

            for i, prefix in enumerate(target_prefixes):
                gradient_ratio = gradient[i] * strength
                
                for patch_key in key_patches:
                    if patch_key.startswith(prefix):
                        sculpted_model.add_patches({patch_key: key_patches[patch_key]}, 1.0, gradient_ratio)
                        modified_patch_count += 1

        if modified_patch_count == 0:
            print(f"ERROR: [Model Sculptor Flux] Found 0 patches matching target prefixes for '{target_blocks}'. Please check model compatibility.")
            return (model,)

        print(f"[Model Sculptor Flux] Successfully applied gradient patches to {modified_patch_count} model components.")
        
        return (sculpted_model,)


class ModelSculptorSDXL:
    GRADIENT_SHAPES = [
        "Linear (Ascending)",
        "Linear (Descending)",
        "Ease In (Quadratic)",
        "Ease Out (Quadratic)",
        "Ease In/Out (Sine)",
        "Spike (Gaussian)",
        "Dip (Inverse Gaussian)",
        "Steps (Ascending)",
        "Steps (Descending)",
        "Random (Noise)"
    ]
    
    TARGET_BLOCKS = [
        "all",
        "input_blocks",
        "middle_block",
        "output_blocks",
        "time_embed",
        "label_emb",
        "Input & Output (Synced Shape)"
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "gradient_shape": (s.GRADIENT_SHAPES,),
                "strength": ("FLOAT", {"default": 0.1, "min": -2.0, "max": 2.0, "step": 0.01}),
                "target_blocks": (s.TARGET_BLOCKS,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "sculpt_model"
    CATEGORY = "models/advanced"

    def _generate_gradient(self, shape, num_steps):
        if num_steps <= 1:
            return np.ones(num_steps) if num_steps == 1 else np.array([])
        x = np.linspace(0, 1, num_steps)
        if shape == "Linear (Ascending)":
            return x
        elif shape == "Linear (Descending)":
            return 1 - x
        elif shape == "Ease In (Quadratic)":
            return x * x
        elif shape == "Ease Out (Quadratic)":
            return 1 - (1 - x) * (1 - x)
        elif shape == "Ease In/Out (Sine)":
            return (np.sin((x - 0.5) * np.pi) + 1) / 2
        elif shape == "Spike (Gaussian)":
            return np.exp(-((x - 0.5) ** 2) / (2 * (0.15 ** 2)))
        elif shape == "Dip (Inverse Gaussian)":
            return 1 - np.exp(-((x - 0.5) ** 2) / (2 * (0.15 ** 2)))
        elif shape == "Steps (Ascending)":
            return np.floor(x * 10) / 10
        elif shape == "Steps (Descending)":
            return np.floor((1 - x) * 10) / 10
        elif shape == "Random (Noise)":
            return np.random.rand(num_steps)
        return np.zeros(num_steps)

    def sculpt_model(self, model, gradient_shape, strength, target_blocks):
        sculpted_model = model.clone()
        key_patches = model.get_key_patches("diffusion_model.")

        sdxl_architecture = {
            "input_blocks": [f"diffusion_model.input_blocks.{i}." for i in range(12)],
            "middle_block": [f"diffusion_model.middle_block."],
            "output_blocks": [f"diffusion_model.output_blocks.{i}." for i in range(12)],
            "time_embed": [f"diffusion_model.time_embed."],
            "label_emb": [f"diffusion_model.label_emb."],
        }
        
        modified_patch_count = 0
        
        if target_blocks == "Input & Output (Synced Shape)":
            print(f"[Model Sculptor SDXL] Applying synced shape '{gradient_shape}' with strength {strength}.")
            
            for block_key in ["input_blocks", "output_blocks"]:
                prefixes = sdxl_architecture[block_key]
                gradient = self._generate_gradient(gradient_shape, len(prefixes))
                print(f"  - Sculpting {len(gradient)} {block_key}.")
                
                for i, prefix in enumerate(prefixes):
                    gradient_ratio = gradient[i] * strength
                    
                    for patch_key in key_patches:
                        if patch_key.startswith(prefix):
                            sculpted_model.add_patches({patch_key: key_patches[patch_key]}, 1.0, gradient_ratio)
                            modified_patch_count += 1
        else:
            target_prefixes = []
            if target_blocks == "all":
                sorted_keys = ["time_embed", "label_emb", "input_blocks", "middle_block", "output_blocks"]
                for block_type in sorted_keys:
                    target_prefixes.extend(sdxl_architecture[block_type])
            elif target_blocks in sdxl_architecture:
                target_prefixes = sdxl_architecture[target_blocks]
            else:
                print(f"Warning: [Model Sculptor SDXL] Unknown target block type '{target_blocks}'.")
                return (model,)

            gradient = self._generate_gradient(gradient_shape, len(target_prefixes))
            print(f"[Model Sculptor SDXL] Sculpting model with shape '{gradient_shape}', strength {strength}, on {len(gradient)} '{target_blocks}' blocks.")

            for i, prefix in enumerate(target_prefixes):
                gradient_ratio = gradient[i] * strength
                
                for patch_key in key_patches:
                    if patch_key.startswith(prefix):
                        sculpted_model.add_patches({patch_key: key_patches[patch_key]}, 1.0, gradient_ratio)
                        modified_patch_count += 1

        if modified_patch_count == 0:
            print(f"ERROR: [Model Sculptor SDXL] Found 0 patches matching target prefixes for '{target_blocks}'. Please check model compatibility.")
            return (model,)

        print(f"[Model Sculptor SDXL] Successfully applied gradient patches to {modified_patch_count} model components.")
        
        return (sculpted_model,)


class ModelSculptorSD3:
    GRADIENT_SHAPES = [
        "Linear (Ascending)",
        "Linear (Descending)",
        "Ease In (Quadratic)",
        "Ease Out (Quadratic)",
        "Ease In/Out (Sine)",
        "Spike (Gaussian)",
        "Dip (Inverse Gaussian)",
        "Steps (Ascending)",
        "Steps (Descending)",
        "Random (Noise)"
    ]
    
    TARGET_BLOCKS = [
        "all",
        "joint_blocks",
        "x_embedder",
        "y_embedder", 
        "t_embedder",
        "pos_embed",
        "final_layer",
        "Joint & Final (Synced Shape)"
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "gradient_shape": (s.GRADIENT_SHAPES,),
                "strength": ("FLOAT", {"default": 0.1, "min": -2.0, "max": 2.0, "step": 0.01}),
                "target_blocks": (s.TARGET_BLOCKS,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "sculpt_model"
    CATEGORY = "models/advanced"

    def _generate_gradient(self, shape, num_steps):
        if num_steps <= 1:
            return np.ones(num_steps) if num_steps == 1 else np.array([])
        x = np.linspace(0, 1, num_steps)
        if shape == "Linear (Ascending)":
            return x
        elif shape == "Linear (Descending)":
            return 1 - x
        elif shape == "Ease In (Quadratic)":
            return x * x
        elif shape == "Ease Out (Quadratic)":
            return 1 - (1 - x) * (1 - x)
        elif shape == "Ease In/Out (Sine)":
            return (np.sin((x - 0.5) * np.pi) + 1) / 2
        elif shape == "Spike (Gaussian)":
            return np.exp(-((x - 0.5) ** 2) / (2 * (0.15 ** 2)))
        elif shape == "Dip (Inverse Gaussian)":
            return 1 - np.exp(-((x - 0.5) ** 2) / (2 * (0.15 ** 2)))
        elif shape == "Steps (Ascending)":
            return np.floor(x * 10) / 10
        elif shape == "Steps (Descending)":
            return np.floor((1 - x) * 10) / 10
        elif shape == "Random (Noise)":
            return np.random.rand(num_steps)
        return np.zeros(num_steps)

    def sculpt_model(self, model, gradient_shape, strength, target_blocks):
        sculpted_model = model.clone()
        key_patches = model.get_key_patches("diffusion_model.")

        sd3_architecture = {
            "joint_blocks": [f"diffusion_model.joint_blocks.{i}." for i in range(24)],
            "x_embedder": [f"diffusion_model.x_embedder."],
            "y_embedder": [f"diffusion_model.y_embedder."],
            "t_embedder": [f"diffusion_model.t_embedder."],
            "pos_embed": [f"diffusion_model.pos_embed."],
            "final_layer": [f"diffusion_model.final_layer."],
        }
        
        modified_patch_count = 0
        
        if target_blocks == "Joint & Final (Synced Shape)":
            print(f"[Model Sculptor SD3] Applying synced shape '{gradient_shape}' with strength {strength}.")
            
            for block_key in ["joint_blocks", "final_layer"]:
                prefixes = sd3_architecture[block_key]
                gradient = self._generate_gradient(gradient_shape, len(prefixes))
                print(f"  - Sculpting {len(gradient)} {block_key}.")
                
                for i, prefix in enumerate(prefixes):
                    gradient_ratio = gradient[i] * strength
                    
                    for patch_key in key_patches:
                        if patch_key.startswith(prefix):
                            sculpted_model.add_patches({patch_key: key_patches[patch_key]}, 1.0, gradient_ratio)
                            modified_patch_count += 1
        else:
            target_prefixes = []
            if target_blocks == "all":
                sorted_keys = ["x_embedder", "y_embedder", "t_embedder", "pos_embed", "joint_blocks", "final_layer"]
                for block_type in sorted_keys:
                    target_prefixes.extend(sd3_architecture[block_type])
            elif target_blocks in sd3_architecture:
                target_prefixes = sd3_architecture[target_blocks]
            else:
                print(f"Warning: [Model Sculptor SD3] Unknown target block type '{target_blocks}'.")
                return (model,)

            gradient = self._generate_gradient(gradient_shape, len(target_prefixes))
            print(f"[Model Sculptor SD3] Sculpting model with shape '{gradient_shape}', strength {strength}, on {len(gradient)} '{target_blocks}' blocks.")

            for i, prefix in enumerate(target_prefixes):
                gradient_ratio = gradient[i] * strength
                
                for patch_key in key_patches:
                    if patch_key.startswith(prefix):
                        sculpted_model.add_patches({patch_key: key_patches[patch_key]}, 1.0, gradient_ratio)
                        modified_patch_count += 1

        if modified_patch_count == 0:
            print(f"ERROR: [Model Sculptor SD3] Found 0 patches matching target prefixes for '{target_blocks}'. Please check model compatibility.")
            return (model,)

        print(f"[Model Sculptor SD3] Successfully applied gradient patches to {modified_patch_count} model components.")
        
        return (sculpted_model,)


NODE_CLASS_MAPPINGS = { 
    "ModelSculptorFlux": ModelSculptorFlux,
    "ModelSculptorSDXL": ModelSculptorSDXL,
    "ModelSculptorSD3": ModelSculptorSD3,
}
NODE_DISPLAY_NAME_MAPPINGS = { 
    "ModelSculptorFlux": "Jurdn's Model Sculptor (Flux)",
    "ModelSculptorSDXL": "Jurdn's Model Sculptor (SDXL)",
    "ModelSculptorSD3": "Jurdn's Model Sculptor (SD3)",
}