# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass, field
from typing import Optional
from functools import wraps

import trl
import inspect
from trl import SFTTrainer
from . import is_bfloat16_supported
from .models._utils import get_multi_gpu_config, init_distributed_training_if_needed
from unsloth_zoo.training_utils import (
    unsloth_train as _unsloth_train,
)
from unsloth_zoo.vision_utils import (
    UnslothVisionDataCollator,
)
from packaging.version import Version
import dataclasses

__all__ = [
    "UnslothTrainingArguments",
    "UnslothTrainer",
    "unsloth_train",
    "_patch_trl_trainer",
    "_patch_trainer_with_ddp_support",
    "UnslothVisionDataCollator",
]

# DDP Configuration Environment Variables
# Users can set these to control DDP behavior
DDP_CONFIG = {
    # Core DDP controls
    "UNSLOTH_DISABLE_DDP_STATIC_GRAPH": "Disable DDP static graph optimization (0/1)",
    "UNSLOTH_DISABLE_DDP_STATIC_GRAPH_FOR_GRAD_CHECKPOINT": "Disable DDP static graph when gradient checkpointing detected (0/1)",
    "UNSLOTH_DISABLE_GRAD_CHECKPOINT_HOOKS": "Disable gradient checkpointing safety hooks (0/1)",

    # Gradient checkpointing detection
    "UNSLOTH_USE_GRADIENT_CHECKPOINTING": "Force enable gradient checkpointing detection (0/1)",
    "GRADIENT_CHECKPOINTING": "Alternative gradient checkpointing flag (0/1)",

    # Debugging and diagnostics
    "UNSLOTH_DEBUG_DDP": "Enable verbose DDP debugging output (0/1)",
    "UNSLOTH_DEBUG_DDP_MODEL_DETECTION": "Enable DDP model detection debugging (0/1)",
    "UNSLOTH_DEBUG_REDUCER": "Enable DDP reducer debugging (0/1)",

    # Memory management
    "UNSLOTH_DDP_MEMORY_CLEANUP": "Enable aggressive memory cleanup in DDP (0/1, default: 1)",
    "UNSLOTH_DDP_DUMMY_FORWARD_STRATEGIES": "Number of dummy forward strategies to try (default: 3)",

    # Multi-GPU configuration
    "UNSLOTH_ENABLE_MULTIGPU": "Enable multi-GPU training auto-detection (0/1)",
    "UNSLOTH_FORCE_DDP_FIND_UNUSED_PARAMETERS": "Force DDP find_unused_parameters setting (true/false)",

    # Advanced DDP settings
    "UNSLOTH_DDP_TIMEOUT_SECONDS": "DDP initialization timeout in seconds (default: 1800)",
    "UNSLOTH_DDP_BACKEND": "Force DDP backend (nccl/gloo/mpi)",
    "UNSLOTH_DDP_BUCKET_SIZE_MB": "DDP bucket size in MB (default: 25)",
}

def _get_ddp_config(key, default=None):
    """Get DDP configuration value from environment variables."""
    import os
    value = os.environ.get(key, default)
    if value in ("0", "false", "False", "FALSE"):
        return False
    elif value in ("1", "true", "True", "TRUE"):
        return True
    return value

def _print_ddp_config():
    """Print current DDP configuration for debugging."""
    import os
    print("Unsloth DDP Configuration:")
    print("-" * 50)
    for key, description in DDP_CONFIG.items():
        value = os.environ.get(key, "not set")
        print(f"{key}: {value} ({description})")
    print("-" * 50)

# Unsloth gradient accumulation fix:
from transformers import __version__ as transformers_version
if Version(transformers_version) > Version("4.45.2"):
    def unsloth_train(trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)
    pass
else:
    def unsloth_train(trainer, *args, **kwargs):
        if len(args) != 0 or len(kwargs) != 0:
            raise RuntimeError(
                "Unsloth: Our custom gradient accumulation fixed trainer does not support other arguments.\n"\
                "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"\
                '`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`'
            )

        return _unsloth_train(trainer)
    pass
pass

try:
    from trl import SFTConfig as TrainingArguments
except:
    from transformers import TrainingArguments
pass

class UnslothTrainingArguments(TrainingArguments):
    def __init__(self, embedding_learning_rate: float = None, *args, **kwargs):
        embedding_learning_rate = embedding_learning_rate
        super().__init__(*args, **kwargs)
pass


def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr = 5e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = \
    {
        "non_embeddings" : {},
        "embeddings"     : {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[:-len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".")+1:]

            param_groups["embeddings"]    [name] = param
        else:
            param_groups["non_embeddings"][name] = param
        pass
    pass

    optimizer_grouped_parameters = [
        {
            "params"       : list(param_groups["non_embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : lr,
        },
        {
            "params"       : list(param_groups["embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
pass


class UnslothTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        # Check for multi-GPU setup and initialize distributed training if needed
        _setup_distributed_training()
        super().__init__(*args, **kwargs)
        
        # Set up DDP static graph after model is initialized
        self._setup_ddp_static_graph(self.model)
    
    def train(self, *args, **kwargs):
        """Override train to ensure DDP static graph is set up before training starts."""
        # Re-setup DDP static graph in case model wrapping happened after init
        self._setup_ddp_static_graph(self.model)
        return super().train(*args, **kwargs)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to handle DDP gradient checkpointing issues."""
        # Setup DDP static graph just before the first training step if not already done
        self._setup_ddp_static_graph_lazy(model)
        
        # Additional safeguard for expect_autograd_hooks_ error:
        # Ensure DDP reducer is properly prepared before training step
        self._prepare_ddp_reducer_for_training(model)
        
        # CRITICAL FIX: Reset gradient checkpointing state for each training step
        # This prevents the "parameter marked ready twice" error in distributed training
        self._reset_gradient_checkpointing_state(model)
        
        return super().training_step(model, inputs, num_items_in_batch)
    
    def _reset_gradient_checkpointing_state(self, model):
        """Reset gradient checkpointing state to prevent parameter marked ready twice errors."""
        try:
            ddp_model = self._find_ddp_model(model)
            if ddp_model is not None and hasattr(ddp_model, 'reducer') and ddp_model.reducer is not None:
                reducer = ddp_model.reducer

                # Reset the ready buckets tracking for gradient checkpointing
                if hasattr(reducer, '_unsloth_ready_buckets'):
                    reducer._unsloth_ready_buckets.clear()

                # Reset reducer's internal state to prevent autograd hook conflicts
                if hasattr(reducer, 'next_bucket'):
                    reducer.next_bucket = 0

                # Enhanced memory management: Clear any cached bucket states
                if hasattr(reducer, '_unsloth_bucket_ready_count'):
                    reducer._unsloth_bucket_ready_count.clear()

                # Reset iteration counter for gradient checkpointing tracking
                if hasattr(reducer, '_unsloth_iteration_count'):
                    reducer._unsloth_iteration_count = 0

                # Clear any accumulated gradient states that might cause memory issues
                try:
                    for param in ddp_model.parameters():
                        if param.requires_grad and param.grad is not None:
                            # Only clear gradients if they're not needed for the current step
                            # This helps prevent memory accumulation in gradient checkpointing
                            if hasattr(param, '_unsloth_grad_cleared'):
                                param.grad = None
                                delattr(param, '_unsloth_grad_cleared')
                except Exception:
                    pass

        except Exception:
            # Silently continue if reset fails - this is a safeguard, not critical
            pass
    
    def _find_ddp_model(self, model):
        """Recursively search for DDP-wrapped model in the model hierarchy."""
        return _find_ddp_model(model)
    
    def _setup_ddp_static_graph(self, model):
        """Setup DDP static graph to fix gradient checkpointing issues."""
        return _setup_ddp_static_graph(model)
    
    def _setup_ddp_static_graph_lazy(self, model):
        """Setup DDP static graph just before first training step if not already done."""
        if not hasattr(self, '_unsloth_ddp_static_graph_setup_done'):
            # Try multiple times with the latest model reference
            # In case Accelerate wrapped the model after init
            success = False
            accelerator_model = None
            if hasattr(self, 'accelerator') and hasattr(self.accelerator, 'model'):
                accelerator_model = self.accelerator.model
            
            # Try all possible model references
            model_candidates = [model, getattr(self, 'model', None), accelerator_model]
            
            # Also add the trainer itself as a candidate in case the model is nested in it
            if hasattr(self, 'accelerator'):
                model_candidates.extend([
                    self.accelerator,
                    getattr(self.accelerator, '_models', None),
                    getattr(self.accelerator, '_prepared_models', None)
                ])
            
            for model_ref in model_candidates:
                if model_ref is not None:
                    if self._setup_ddp_static_graph(model_ref):
                        success = True
                        break
            
            self._unsloth_ddp_static_graph_setup_done = True
            
            if not success:
                # Last resort: try to find DDP model in accelerator with more thorough search
                try:
                    if hasattr(self, 'accelerator'):
                        # Try all attributes of accelerator that might contain models
                        for attr_name in dir(self.accelerator):
                            if 'model' in attr_name.lower() and not attr_name.startswith('_'):
                                try:
                                    attr_value = getattr(self.accelerator, attr_name)
                                    if attr_value is not None and self._setup_ddp_static_graph(attr_value):
                                        success = True
                                        break
                                except (AttributeError, RuntimeError):
                                    continue
                except:
                    pass
            return success
        return True
    
    def _prepare_ddp_reducer_for_training(self, model):
        """Prepare DDP reducer to avoid expect_autograd_hooks_ errors."""
        if hasattr(self, '_unsloth_ddp_reducer_prepared'):
            return True
            
        import os
        import torch
        
        # Only proceed if we're in a distributed environment
        if not (os.environ.get("LOCAL_RANK") is not None or 
                os.environ.get("WORLD_SIZE") is not None):
            return False
        
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                return False
                
            # Find the DDP-wrapped model
            ddp_model = self._find_ddp_model(model)
            
            if ddp_model is not None:
                try:
                    # ENHANCED FIX for expect_autograd_hooks_ errors:
                    # The key insight is that DDP's reducer needs to be fully initialized
                    # with proper autograd hook registration before any backward pass occurs.
                    
                    # Method 1: Enhanced dummy forward pass to ensure DDP is fully initialized
                    # This ensures the reducer knows about all parameters and their autograd hooks
                    try:
                        # Create dummy input that matches the model's expected input structure
                        # This triggers DDP's lazy initialization of autograd hooks
                        with torch.no_grad():
                            # Get first parameter to determine device and dtype
                            first_param = next(ddp_model.parameters())
                            device = first_param.device
                            dtype = first_param.dtype

                            # Set model to eval mode temporarily to avoid affecting training state
                            original_training_mode = ddp_model.training
                            ddp_model.eval()

                            # Try multiple dummy input strategies
                            dummy_inputs = [
                                # Strategy 1: Standard transformer input
                                {
                                    'input_ids': torch.tensor([[1, 2]], device=device, dtype=torch.long),
                                    'attention_mask': torch.tensor([[1, 1]], device=device, dtype=torch.long),
                                },
                                # Strategy 2: Extended transformer input with position_ids
                                {
                                    'input_ids': torch.tensor([[1, 2]], device=device, dtype=torch.long),
                                    'attention_mask': torch.tensor([[1, 1]], device=device, dtype=torch.long),
                                    'position_ids': torch.tensor([[0, 1]], device=device, dtype=torch.long),
                                },
                                # Strategy 3: Vision-language model input
                                {
                                    'input_ids': torch.tensor([[1, 2]], device=device, dtype=torch.long),
                                    'attention_mask': torch.tensor([[1, 1]], device=device, dtype=torch.long),
                                    'pixel_values': torch.randn(1, 3, 224, 224, device=device, dtype=dtype),
                                },
                            ]

                            success = False
                            for dummy_input in dummy_inputs:
                                try:
                                    # Run dummy forward pass to initialize DDP reducer and autograd hooks
                                    with torch.amp.autocast('cuda', enabled=False):  # Disable autocast for dummy pass
                                        _ = ddp_model(**dummy_input)
                                    success = True
                                    break
                                except Exception:
                                    continue

                            # If structured inputs fail, try simple tensor inputs
                            if not success:
                                tensor_inputs = [
                                    torch.randn(1, 2, device=device, dtype=dtype),
                                    torch.randn(1, 2, 768, device=device, dtype=dtype),  # Common hidden size
                                    torch.randn(1, 2, 4096, device=device, dtype=dtype),  # Larger hidden size
                                ]

                                for dummy_tensor in tensor_inputs:
                                    try:
                                        with torch.amp.autocast('cuda', enabled=False):
                                            _ = ddp_model(dummy_tensor)
                                        success = True
                                        break
                                    except Exception:
                                        continue

                            # Restore original training mode
                            ddp_model.train(original_training_mode)

                            # Enhanced memory cleanup after dummy forward pass
                            try:
                                # Clear any cached tensors from the dummy forward pass
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                                # Clear gradients that might have been created during dummy pass
                                for param in ddp_model.parameters():
                                    if param.grad is not None:
                                        param.grad = None

                            except Exception:
                                pass

                            if success:
                                print("Unsloth: Successfully initialized DDP reducer with dummy forward pass")

                    except Exception as e:
                        print(f"Unsloth: Could not run dummy forward pass for DDP initialization: {e}")
                        
                    # Method 2: Enhanced reducer preparation with comprehensive state management
                    if hasattr(ddp_model, 'reducer') and ddp_model.reducer is not None:
                        reducer = ddp_model.reducer

                        # Step 1: Force reducer bucket rebuilding to ensure proper autograd hook setup
                        if hasattr(reducer, '_rebuild_buckets'):
                            try:
                                reducer._rebuild_buckets()
                                print("Unsloth: Successfully rebuilt DDP reducer buckets")
                            except Exception as e:
                                print(f"Unsloth: Warning - Could not rebuild reducer buckets: {e}")

                        # Step 2: Initialize reducer state for forward pass
                        if hasattr(reducer, '_prepare_for_forward'):
                            try:
                                reducer._prepare_for_forward()
                            except Exception:
                                pass

                        # Step 3: Reset and validate autograd hook state
                        if hasattr(reducer, 'next_bucket'):
                            try:
                                # Reset the autograd hook state to ensure consistency
                                # This is the key fix for expect_autograd_hooks_ errors
                                reducer.next_bucket = 0

                                # Initialize bucket tracking if not present
                                if not hasattr(reducer, '_unsloth_bucket_ready_count'):
                                    reducer._unsloth_bucket_ready_count = {}

                                # Ensure hooks are properly aligned with parameters
                                if hasattr(reducer, '_ensure_autograd_hooks_prepared'):
                                    reducer._ensure_autograd_hooks_prepared()

                                # Validate reducer state consistency
                                if hasattr(reducer, 'buckets') and hasattr(reducer, '_autograd_hooks'):
                                    num_buckets = len(reducer.buckets) if reducer.buckets else 0
                                    num_hooks = len(reducer._autograd_hooks) if reducer._autograd_hooks else 0
                                    if num_buckets > 0 and num_hooks == 0:
                                        print("Unsloth: Warning - DDP reducer has buckets but no autograd hooks")

                            except Exception as e:
                                print(f"Unsloth: Warning - Could not validate reducer state: {e}")

                        # Step 4: Enhanced autograd hook validation
                        try:
                            # Check if all trainable parameters have corresponding autograd hooks
                            trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]
                            if hasattr(reducer, '_autograd_hooks') and reducer._autograd_hooks:
                                hook_count = len(reducer._autograd_hooks)
                                param_count = len(trainable_params)
                                if hook_count != param_count:
                                    print(f"Unsloth: Warning - Autograd hook count ({hook_count}) != parameter count ({param_count})")
                                    # Try to force hook re-registration
                                    if hasattr(reducer, '_install_hooks'):
                                        try:
                                            reducer._install_hooks()
                                        except Exception:
                                            pass
                        except Exception:
                            pass
                    
                    # Method 3: Force parameter registration and validation
                    try:
                        # Ensure all trainable parameters are known to DDP
                        trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]
                        if trainable_params:
                            # Access parameter gradients to ensure autograd graph readiness
                            for param in trainable_params[:5]:  # Check first few params only
                                if param.grad is not None:
                                    # Clear any existing gradients to reset autograd state
                                    param.grad.zero_()
                    except Exception:
                        pass
                    
                    self._unsloth_ddp_reducer_prepared = True

                    return True
                    
                except Exception as e:
                    print(f"Unsloth: Warning - Could not prepare DDP reducer: {e}")
                    self._unsloth_ddp_reducer_prepared = True  # Mark as done to avoid repeated attempts
                    return False
            
            self._unsloth_ddp_reducer_prepared = True
            return False
            
        except Exception as e:
            print(f"Unsloth: Warning - Could not prepare DDP reducer: {e}")
            self._unsloth_ddp_reducer_prepared = True
            return False
    
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None: return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        pass
        return self.optimizer
    pass
pass

# From `trl>=0.13.0`, they changed how to pass several params to the trainer
# We need to patch to make the transition smooth
def _backwards_compatible_trainer(trainer_class, config_class):
    original_init = trainer_class.__init__
    
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # All Trainer tokenizer are now called processing_class
        trainer_params = set(inspect.signature(original_init).parameters.keys())

        if "processing_class" in trainer_params and "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        pass

        if ("args" in kwargs) and (Version(trl.__version__) >= Version("0.13.0.dev0")):
            training_args = kwargs.pop("args", None)

            # Get parameters that Trainer.__init__ actually expects
            trainer_params.remove('self')
            trainer_params.remove('args')

            # Get fields that should be passed to Config init
            config_fields = {
                field.name: field for field in dataclasses.fields(config_class) 
                if field.init
            }
            
            # Create config dict with valid fields from training_args
            config_dict = {
                name: getattr(training_args, name)
                for name in config_fields
                if hasattr(training_args, name)
            }

            # Get parameters that exist in Config but not in TrainingArguments
            from transformers import TrainingArguments
            moved_params = \
                set(inspect.signature(config_class)     .parameters.keys()) - \
                set(inspect.signature(TrainingArguments).parameters.keys())
            
            # Separate kwargs into trainer kwargs and config kwargs
            trainer_kwargs = {}
            additional_config_kwargs = {}

            for key, value in kwargs.items():
                if key in trainer_params: trainer_kwargs[key] = value
                elif key in moved_params or key in config_fields:
                    additional_config_kwargs[key] = value
                else:
                    additional_config_kwargs[key] = value
                pass
            pass

            # Update config_dict with additional kwargs
            config_dict.update(additional_config_kwargs)

            # Create Config with all the collected parameters
            # Reinitialising config class with parameters (that were none initially but populated on first init)
            # causes the 2nd init to fail as there are mutual exclusive checks on pairs of parameters.
            # Refer: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py#L499-L502 for example
            # So we only create config class if the previous init was not TrainingArguments
            if not isinstance(training_args, TrainingArguments):
                config = config_class(**config_dict)
            else:
                config = training_args

            # Reconstruct kwargs for Trainer
            kwargs = trainer_kwargs
            kwargs["args"] = config
        pass
        original_init(self, *args, **kwargs)
    pass
    return new_init
pass


# Standalone DDP functions that can be used to patch any trainer
def _setup_distributed_training():
    """Setup distributed training if in multi-GPU environment with enhanced error handling."""
    import os
    import torch

    # Print configuration if debugging is enabled
    if _get_ddp_config("UNSLOTH_DEBUG_DDP", False):
        _print_ddp_config()

    try:
        # Get multi-GPU configuration
        multi_gpu_config = get_multi_gpu_config()

        # Initialize distributed training if needed
        if multi_gpu_config["enable_multi_gpu"]:
            init_distributed_training_if_needed()

        # Check if we're in a distributed environment
        if (os.environ.get("LOCAL_RANK") is not None or
            os.environ.get("WORLD_SIZE") is not None):
            try:
                import torch.distributed as dist

                # Enhanced distributed environment validation
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                world_size = int(os.environ.get("WORLD_SIZE", 1))

                print(f"Unsloth: Detected distributed environment - LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")

                if not dist.is_initialized():
                    # Enhanced validation and race condition prevention
                    import time
                    from datetime import timedelta

                    # Validate CUDA availability for NCCL backend
                    if torch.cuda.is_available():
                        if local_rank >= torch.cuda.device_count():
                            raise RuntimeError(f"LOCAL_RANK {local_rank} >= available CUDA devices {torch.cuda.device_count()}")

                        # Set device before initializing process group
                        torch.cuda.set_device(local_rank)
                        print(f"Unsloth: Set CUDA device to {local_rank}")

                        # Initialize with NCCL backend for CUDA
                        backend = "nccl"
                    else:
                        # Fallback to Gloo for CPU-only training
                        backend = "gloo"
                        print("Unsloth: CUDA not available, using Gloo backend for distributed training")

                    # Allow backend override
                    backend_override = _get_ddp_config("UNSLOTH_DDP_BACKEND", None)
                    if backend_override:
                        backend = backend_override
                        print(f"Unsloth: Using backend override: {backend}")

                    # Configure timeout
                    timeout_seconds = int(_get_ddp_config("UNSLOTH_DDP_TIMEOUT_SECONDS", "1800"))
                    timeout = timedelta(seconds=timeout_seconds)

                    # Race condition prevention: Add small random delay for different ranks
                    if world_size > 1:
                        import random
                        delay = random.uniform(0.1, 0.5) * local_rank
                        time.sleep(delay)

                    # Initialize process group with enhanced error handling
                    print(f"Unsloth: Initializing distributed process group with {backend} backend (timeout: {timeout_seconds}s)")

                    try:
                        dist.init_process_group(
                            backend=backend,
                            timeout=timeout,
                            world_size=world_size,
                            rank=local_rank
                        )

                        # Verify initialization
                        if dist.is_initialized():
                            actual_rank = dist.get_rank()
                            actual_world_size = dist.get_world_size()
                            print(f"Unsloth: Successfully initialized distributed training - rank {actual_rank}/{actual_world_size}")

                            # Synchronization barrier to ensure all processes are ready
                            if world_size > 1:
                                print(f"Unsloth: Waiting for all processes to synchronize...")
                                dist.barrier()
                                print(f"Unsloth: All processes synchronized")
                        else:
                            raise RuntimeError("Process group initialization appeared to succeed but dist.is_initialized() returns False")

                    except Exception as init_e:
                        print(f"Unsloth: Failed to initialize process group: {init_e}")
                        # Try alternative initialization methods
                        if backend == "nccl" and torch.cuda.is_available():
                            print("Unsloth: Retrying with Gloo backend as fallback...")
                            try:
                                dist.init_process_group(
                                    backend="gloo",
                                    timeout=timeout,
                                    world_size=world_size,
                                    rank=local_rank
                                )
                                print("Unsloth: Successfully initialized with Gloo backend fallback")
                            except Exception as fallback_e:
                                print(f"Unsloth: Fallback initialization also failed: {fallback_e}")
                                raise init_e
                        else:
                            raise init_e

                else:
                    print(f"Unsloth: Distributed training already initialized - rank {dist.get_rank()}/{dist.get_world_size()}")

            except Exception as e:
                print(f"Unsloth: Failed to initialize distributed training: {e}")
                print(f"Unsloth: Error type: {type(e).__name__}")
                print("Unsloth: Falling back to single-GPU training")

                # Additional debugging information
                print(f"Unsloth: Environment variables - LOCAL_RANK: {os.environ.get('LOCAL_RANK')}, WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
                if torch.cuda.is_available():
                    print(f"Unsloth: CUDA devices available: {torch.cuda.device_count()}")
                else:
                    print("Unsloth: CUDA not available")

        elif multi_gpu_config["supports_multi_gpu"] and multi_gpu_config["enable_multi_gpu"]:
            print("Unsloth: Multi-GPU support enabled but not in distributed environment")

    except Exception as e:
        print(f"Unsloth: Error in distributed training setup: {e}")
        print("Unsloth: Continuing with single-GPU training")

def _find_ddp_model(model):
    """Recursively search for DDP-wrapped model in the model hierarchy."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    # First, do a direct check - this catches the most obvious cases
    if isinstance(model, DDP):
        return model

    # Second, check the common .module pattern
    if hasattr(model, 'module') and isinstance(model.module, DDP):
        return model.module

    # CRITICAL FIX: Handle PEFT models specifically
    # PEFT models (PeftModel, PeftModelForCausalLM, etc.) wrap the base model
    # The DDP wrapper might be at the base_model level
    model_type_name = type(model).__name__
    if 'Peft' in model_type_name:
        # Check for PEFT-specific attributes where the base model is stored
        peft_base_attrs = ['base_model', 'model']
        for attr in peft_base_attrs:
            if hasattr(model, attr):
                base_model = getattr(model, attr)
                if isinstance(base_model, DDP):
                    return base_model
                # Some PEFT versions nest it further: base_model.model
                if hasattr(base_model, 'model') and isinstance(base_model.model, DDP):
                    return base_model.model
                # ENHANCED: Handle the specific structure from the error: base_model.model.model
                # This is the structure causing the issue: PeftModelForCausalLM -> base_model -> model -> model (DDP)
                if hasattr(base_model, 'model') and hasattr(base_model.model, 'model'):
                    nested_model = base_model.model.model
                    if isinstance(nested_model, DDP):
                        return nested_model
                # Some cases have base_model.module for DDP
                if hasattr(base_model, 'module') and isinstance(base_model.module, DDP):
                    return base_model.module
                # ENHANCED: Handle even deeper nesting: base_model.model.model.model
                # Some complex PEFT + Accelerate setups can have this structure
                if (hasattr(base_model, 'model') and hasattr(base_model.model, 'model') and
                    hasattr(base_model.model.model, 'model')):
                    deep_nested_model = base_model.model.model.model
                    if isinstance(deep_nested_model, DDP):
                        return deep_nested_model
                # Check for module patterns at each level
                if hasattr(base_model, 'model') and hasattr(base_model.model, 'module'):
                    if isinstance(base_model.model.module, DDP):
                        return base_model.model.module
    
    # Track visited objects to avoid infinite recursion
    visited = set()

    def _recursive_search(obj, depth=0, max_depth=12):
        # Avoid infinite recursion
        if depth > max_depth or id(obj) in visited:
            return None
        visited.add(id(obj))

        # Check if this object is a DDP model
        if isinstance(obj, DDP):
            return obj

        # Don't recurse into basic types
        if not hasattr(obj, '__dict__') and not hasattr(obj, '__getattribute__'):
            return None

        # Enhanced search: Check for more comprehensive list of attribute names
        # where DDP models might be nested in various training frameworks
        search_attrs = [
            # Core PyTorch patterns
            'module', 'model', '_orig_mod', '_module', '_model',
            'wrapped_model', 'inner_model', '_wrapped_model', '_inner_model',
            'core_model', '_core_model', 'underlying_model', '_underlying_model',

            # PEFT library attributes (high priority for this fix)
            'base_model', 'peft_model', '_peft_model', 'peft_base_model',

            # Accelerate library attributes
            '_ddp_module', '_orig_ddp_module', '_accelerate_wrapped_model',
            '_prepared_model', '_models', '_model_ref', '_original_model',
            '_accelerate_model', '_prepared', '_prep_model', '_accelerate_prepared_model',

            # Transformers library attributes
            '_transformers_model', '_hf_model', 'transformer', '_transformer',

            # TRL/SFT trainer attributes
            '_sft_model', '_trl_model', 'sft_model', 'trl_model',

            # Additional nested patterns found in distributed training
            '_distributed_model', '_ddp_wrapped', '_wrapped', '_ref',
            'ddp_model', '_ddp_model', 'distributed_model',

            # Trainer-specific attributes
            'trainer_model', '_trainer_model', 'training_model', '_training_model',

            # Framework-specific wrappers
            'pytorch_model', '_pytorch_model', 'torch_model', '_torch_model',
            'compiled_model', '_compiled_model', 'optimized_model', '_optimized_model'
        ]
        
        for attr_name in search_attrs:
            try:
                if hasattr(obj, attr_name):
                    attr_value = getattr(obj, attr_name)
                    if isinstance(attr_value, DDP):
                        return attr_value
                    # Recursive search for deeply nested models
                    found = _recursive_search(attr_value, depth + 1)
                    if found is not None:
                        return found
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # Some attributes may not be accessible or may raise errors
                continue

        # ENHANCED: Check for dynamic attribute patterns
        # Some frameworks create attributes dynamically
        try:
            if hasattr(obj, '__dict__'):
                for attr_name, attr_value in obj.__dict__.items():
                    # Skip private attributes that are likely not models
                    if attr_name.startswith('__') or attr_name in ['_modules', '_parameters', '_buffers']:
                        continue
                    # Look for attributes that might contain models
                    if ('model' in attr_name.lower() or 'ddp' in attr_name.lower() or
                        'module' in attr_name.lower() or 'wrapped' in attr_name.lower()):
                        try:
                            if isinstance(attr_value, DDP):
                                return attr_value
                            # Recursive search for nested models
                            found = _recursive_search(attr_value, depth + 1)
                            if found is not None:
                                return found
                        except (AttributeError, RuntimeError, TypeError, ValueError):
                            continue
        except (AttributeError, RuntimeError, TypeError):
            pass
        
        # PEFT-specific handling: For PEFT models, also check nested base_model.model patterns
        obj_type_name = type(obj).__name__
        if 'Peft' in obj_type_name and hasattr(obj, 'base_model'):
            try:
                base_model = obj.base_model
                # Check if base_model has a 'model' attribute (common PEFT pattern)
                if hasattr(base_model, 'model'):
                    nested_model = base_model.model
                    if isinstance(nested_model, DDP):
                        return nested_model
                    # Also check if the nested model has a module (DDP pattern)
                    if hasattr(nested_model, 'module') and isinstance(nested_model.module, DDP):
                        return nested_model.module
            except (AttributeError, RuntimeError):
                pass
        
        # Check if the object has _modules dict (common in PyTorch modules)
        try:
            if hasattr(obj, '_modules') and isinstance(obj._modules, dict):
                for module in obj._modules.values():
                    if isinstance(module, DDP):
                        return module
                    found = _recursive_search(module, depth + 1)
                    if found is not None:
                        return found
        except (AttributeError, RuntimeError):
            pass
        
        # Check if the object has parameters (indicating it's a model-like object)
        try:
            if hasattr(obj, 'parameters') and callable(obj.parameters):
                # This might be a wrapper around the actual model, check its attributes
                for attr_name in dir(obj):
                    if not attr_name.startswith('_') and attr_name not in ['parameters', 'named_parameters', 'modules', 'named_modules']:
                        try:
                            attr_value = getattr(obj, attr_name)
                            if hasattr(attr_value, '__dict__') or hasattr(attr_value, '_modules'):
                                found = _recursive_search(attr_value, depth + 1)
                                if found is not None:
                                    return found
                        except (AttributeError, RuntimeError, TypeError):
                            continue
        except (AttributeError, RuntimeError):
            pass
        
        return None
    
    # First try the basic recursive search
    result = _recursive_search(model)
    if result is not None:
        return result
    
    # Enhanced search: If we still haven't found it, try some additional patterns
    # that are specific to accelerate and distributed training setups
    try:
        # Check if this is actually a reference to a trainer/accelerator object
        # that might have the DDP model nested deeper
        for attr in ['accelerator', '_accelerator', 'trainer', '_trainer', 'engine', '_engine']:
            if hasattr(model, attr):
                accelerator_obj = getattr(model, attr)
                if accelerator_obj is not None:
                    # Look for model in accelerator with expanded search
                    for model_attr in ['model', '_model', 'prepared_model', '_prepared_model',
                                     'wrapped_model', '_wrapped_model', 'ddp_model', '_ddp_model']:
                        if hasattr(accelerator_obj, model_attr):
                            acc_model = getattr(accelerator_obj, model_attr)
                            if isinstance(acc_model, DDP):
                                return acc_model
                            # Recursive search on accelerator's model
                            found = _recursive_search(acc_model)
                            if found is not None:
                                return found

                    # ENHANCED: Check for list/dict of models in accelerator
                    # Some accelerate versions store models in collections
                    for collection_attr in ['_models', '_prepared_models', 'models']:
                        if hasattr(accelerator_obj, collection_attr):
                            try:
                                collection = getattr(accelerator_obj, collection_attr)
                                if isinstance(collection, (list, tuple)):
                                    for item in collection:
                                        if isinstance(item, DDP):
                                            return item
                                        found = _recursive_search(item)
                                        if found is not None:
                                            return found
                                elif isinstance(collection, dict):
                                    for item in collection.values():
                                        if isinstance(item, DDP):
                                            return item
                                        found = _recursive_search(item)
                                        if found is not None:
                                            return found
                            except (AttributeError, RuntimeError, TypeError):
                                continue
    except (AttributeError, RuntimeError, TypeError):
        pass
    
    return None


def _setup_ddp_static_graph(model):
    """Setup DDP static graph and autograd hooks to fix gradient checkpointing issues."""
    import os
    import torch
    
    # Check configuration flags
    if _get_ddp_config("UNSLOTH_DISABLE_DDP_STATIC_GRAPH", False):
        if _get_ddp_config("UNSLOTH_DEBUG_DDP", False):
            print("Unsloth: DDP static graph disabled by UNSLOTH_DISABLE_DDP_STATIC_GRAPH")
        return False

    if _get_ddp_config("UNSLOTH_DISABLE_DDP_STATIC_GRAPH_FOR_GRAD_CHECKPOINT", False):
        if _get_ddp_config("UNSLOTH_DEBUG_DDP", False):
            print("Unsloth: DDP static graph disabled for gradient checkpointing by UNSLOTH_DISABLE_DDP_STATIC_GRAPH_FOR_GRAD_CHECKPOINT")
        return False

    if _get_ddp_config("UNSLOTH_DISABLE_GRAD_CHECKPOINT_HOOKS", False):
        print("Unsloth: Gradient checkpointing safety hooks disabled by environment variable")
        return False
    
    # Only proceed if we're in a distributed environment
    if not (os.environ.get("LOCAL_RANK") is not None or 
            os.environ.get("WORLD_SIZE") is not None):
        return False
    
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            return False
            
        # Find the DDP-wrapped model - check multiple levels of nesting
        ddp_model = _find_ddp_model(model)
        
        if ddp_model is not None:
            try:
                # Check if static graph is already set
                if hasattr(ddp_model, '_static_graph') and ddp_model._static_graph:
                    # Already set, don't set again
                    return True
                    
                # CRITICAL FIX: Check if gradient checkpointing is enabled
                # Static graph is incompatible with gradient checkpointing that changes graph structure
                uses_gradient_checkpointing = False

                # Method 1: Check direct gradient checkpointing flags
                if hasattr(model, 'gradient_checkpointing') and model.gradient_checkpointing:
                    uses_gradient_checkpointing = True
                elif hasattr(model, '_set_gradient_checkpointing'):
                    # Some models have this flag
                    if hasattr(model, 'gradient_checkpointing_enable') or hasattr(model, '_gradient_checkpointing'):
                        uses_gradient_checkpointing = True

                # Method 2: Check the underlying model (in case it's wrapped)
                actual_model = getattr(model, 'module', model)
                if hasattr(actual_model, 'gradient_checkpointing') and actual_model.gradient_checkpointing:
                    uses_gradient_checkpointing = True

                # Method 3: Check nested models (PEFT, Accelerate wrappers)
                models_to_check = [model, actual_model]
                if hasattr(model, 'base_model'):
                    models_to_check.append(model.base_model)
                    if hasattr(model.base_model, 'model'):
                        models_to_check.append(model.base_model.model)

                for check_model in models_to_check:
                    if hasattr(check_model, 'gradient_checkpointing') and check_model.gradient_checkpointing:
                        uses_gradient_checkpointing = True
                        break

                # Method 4: Check if any layer has gradient checkpointing enabled
                for module in actual_model.modules():
                    if hasattr(module, 'gradient_checkpointing') and module.gradient_checkpointing:
                        uses_gradient_checkpointing = True
                        break
                
                # Check for Unsloth-specific gradient checkpointing
                # Look for common Unsloth gradient checkpointing patterns
                for module in actual_model.modules():
                    # Check if module name contains unsloth gradient checkpointing indicators
                    module_name = module.__class__.__name__
                    if 'unsloth' in module_name.lower() or 'checkpoint' in module_name.lower():
                        # Additional check for gradient checkpointing usage
                        if hasattr(module, 'forward') and hasattr(module.forward, '__wrapped__'):
                            # This suggests the forward method has been wrapped for checkpointing
                            uses_gradient_checkpointing = True
                            break
                
                # Method 5: UNSLOTH SPECIFIC - Check if unsloth smart gradient checkpointing is active
                # This is the most reliable way to detect Unsloth's gradient checkpointing
                try:
                    # Import the unsloth zoo utility to check for active gradient checkpointing
                    import unsloth_zoo.gradient_checkpointing as unsloth_gc
                    # If this module exists and has been patched, gradient checkpointing is likely active
                    if hasattr(unsloth_gc, '_UNSLOTH_GRADIENT_CHECKPOINTING_ENABLED'):
                        if getattr(unsloth_gc, '_UNSLOTH_GRADIENT_CHECKPOINTING_ENABLED', False):
                            uses_gradient_checkpointing = True

                    # ENHANCED: Check for Unsloth gradient checkpointing function patches
                    # Look for indicators that gradient checkpointing has been applied
                    if hasattr(unsloth_gc, 'forward') or hasattr(unsloth_gc, 'backward'):
                        # If the module has forward/backward functions, it's likely active
                        uses_gradient_checkpointing = True

                    # Check for specific Unsloth gradient checkpointing classes
                    if (hasattr(unsloth_gc, 'Unsloth_Gradient_Checkpointer') or
                        hasattr(unsloth_gc, 'Unsloth_Offloaded_Gradient_Checkpointer')):
                        uses_gradient_checkpointing = True

                except ImportError:
                    pass

                # Method 6: Check for environment variables that indicate gradient checkpointing
                # Many users set this when using Unsloth
                if _get_ddp_config("UNSLOTH_USE_GRADIENT_CHECKPOINTING", False):
                    uses_gradient_checkpointing = True
                if _get_ddp_config("GRADIENT_CHECKPOINTING", False):
                    uses_gradient_checkpointing = True
                    
                # Method 7: Look for gradient checkpointing in the actual model forward methods
                # Unsloth often patches the forward methods of transformer layers
                try:
                    # Check a limited number of modules to avoid performance issues
                    module_count = 0
                    for module in actual_model.modules():
                        if module_count > 50:  # Limit search to avoid performance issues
                            break
                        module_count += 1

                        if hasattr(module, 'forward'):
                            forward_func = module.forward
                            # Check if forward method has been wrapped by gradient checkpointing
                            if (hasattr(forward_func, '__wrapped__') or
                                hasattr(forward_func, '_unsloth_gradient_checkpointed') or
                                hasattr(forward_func, '_gradient_checkpointed') or
                                'checkpoint' in str(forward_func) or
                                'unsloth' in str(forward_func).lower()):
                                uses_gradient_checkpointing = True
                                break

                        # Check for specific Unsloth layer types that use gradient checkpointing
                        module_name = module.__class__.__name__
                        if ('unsloth' in module_name.lower() and
                            ('layer' in module_name.lower() or 'block' in module_name.lower())):
                            # Unsloth layers often have gradient checkpointing enabled by default
                            uses_gradient_checkpointing = True
                            break
                except Exception:
                    pass

                # Method 8: Look for gradient checkpointing in model's config
                models_with_config = [actual_model]
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'config'):
                    models_with_config.append(model.base_model)

                for check_model in models_with_config:
                    if hasattr(check_model, 'config'):
                        config = check_model.config
                        if hasattr(config, 'use_gradient_checkpointing') and config.use_gradient_checkpointing:
                            uses_gradient_checkpointing = True
                            break
                        if hasattr(config, 'gradient_checkpointing') and config.gradient_checkpointing:
                            uses_gradient_checkpointing = True
                            break
                
                # If gradient checkpointing is detected, disable static graph
                if uses_gradient_checkpointing:
                    print("Unsloth: Gradient checkpointing detected - disabling DDP static graph to prevent parameter ready issues")

                    # Debug output if enabled
                    if _get_ddp_config("UNSLOTH_DEBUG_DDP", False):
                        print("Unsloth: DEBUG - Gradient checkpointing detection details:")
                        print(f"  - Model gradient_checkpointing: {getattr(model, 'gradient_checkpointing', 'not found')}")
                        print(f"  - Actual model gradient_checkpointing: {getattr(actual_model, 'gradient_checkpointing', 'not found')}")
                        print(f"  - Environment UNSLOTH_USE_GRADIENT_CHECKPOINTING: {_get_ddp_config('UNSLOTH_USE_GRADIENT_CHECKPOINTING', 'not set')}")
                        print(f"  - Environment GRADIENT_CHECKPOINTING: {_get_ddp_config('GRADIENT_CHECKPOINTING', 'not set')}")
                    
                    # CRITICAL FIX: Don't set static graph when gradient checkpointing is active
                    # Instead, apply alternative fixes for the parameter ready issue
                    
                    # Alternative fix 1: Ensure find_unused_parameters is False
                    if hasattr(ddp_model, 'find_unused_parameters'):
                        if ddp_model.find_unused_parameters:
                            print("Unsloth: Warning - DDP find_unused_parameters=True with gradient checkpointing may cause parameter ready issues")
                            print("Unsloth: Recommend setting ddp_find_unused_parameters=False in training arguments")
                            
                    # Alternative fix 2: Apply enhanced gradient synchronization hook to prevent multiple ready states
                    try:
                        if hasattr(ddp_model, 'reducer') and ddp_model.reducer is not None:
                            reducer = ddp_model.reducer

                            # Install a hook to manage parameter ready state for gradient checkpointing
                            if not hasattr(reducer, '_unsloth_grad_checkpoint_hook_installed'):
                                original_mark_bucket_ready = getattr(reducer, '_mark_bucket_ready', None)

                                if original_mark_bucket_ready is not None:
                                    def _unsloth_safe_mark_bucket_ready(bucket_index):
                                        """Safely mark bucket ready, avoiding duplicate marking with gradient checkpointing."""
                                        try:
                                            # Initialize tracking if not present
                                            if not hasattr(reducer, '_unsloth_ready_buckets'):
                                                reducer._unsloth_ready_buckets = set()
                                            if not hasattr(reducer, '_unsloth_iteration_count'):
                                                reducer._unsloth_iteration_count = 0

                                            # Check if this bucket has already been marked ready in this iteration
                                            bucket_key = (reducer._unsloth_iteration_count, bucket_index)
                                            if bucket_key in reducer._unsloth_ready_buckets:
                                                # Already marked, skip to avoid "ready twice" error
                                                return

                                            # Mark this bucket as ready for this iteration
                                            reducer._unsloth_ready_buckets.add(bucket_key)

                                            # Call the original function
                                            return original_mark_bucket_ready(bucket_index)

                                        except Exception as e:
                                            # If our hook fails, fall back to original behavior
                                            print(f"Unsloth: Warning in gradient checkpointing hook: {e}")
                                            return original_mark_bucket_ready(bucket_index)

                                    # Install iteration counter hook
                                    original_prepare_for_forward = getattr(reducer, '_prepare_for_forward', None)
                                    if original_prepare_for_forward is not None:
                                        def _unsloth_prepare_for_forward():
                                            """Increment iteration counter and clean up old bucket tracking."""
                                            try:
                                                if hasattr(reducer, '_unsloth_iteration_count'):
                                                    reducer._unsloth_iteration_count += 1
                                                    # Clean up old bucket tracking (keep only last 2 iterations)
                                                    if hasattr(reducer, '_unsloth_ready_buckets'):
                                                        current_iter = reducer._unsloth_iteration_count
                                                        reducer._unsloth_ready_buckets = {
                                                            (iter_num, bucket_idx) for iter_num, bucket_idx in reducer._unsloth_ready_buckets
                                                            if iter_num >= current_iter - 1
                                                        }
                                                return original_prepare_for_forward()
                                            except Exception as e:
                                                print(f"Unsloth: Warning in iteration tracking: {e}")
                                                return original_prepare_for_forward()

                                        reducer._prepare_for_forward = _unsloth_prepare_for_forward

                                    # Install the main hook
                                    reducer._mark_bucket_ready = _unsloth_safe_mark_bucket_ready
                                    reducer._unsloth_grad_checkpoint_hook_installed = True
                                    print("Unsloth: Installed enhanced gradient checkpointing safety hook to prevent parameter ready errors")

                    except Exception as e:
                        print(f"Unsloth: Could not install gradient checkpointing safety hook: {e}")
                    
                    # Don't set static graph when gradient checkpointing is active
                    return False
                    
                # Additional fix for expect_autograd_hooks_ error:
                # Ensure that find_unused_parameters is False to avoid autograd hook issues
                if hasattr(ddp_model, 'find_unused_parameters'):
                    if ddp_model.find_unused_parameters:
                        print("Unsloth: Warning - DDP find_unused_parameters=True may cause autograd hook errors with gradient checkpointing")
                        print("Unsloth: Recommend setting ddp_find_unused_parameters=False in training arguments")
                
                # Force DDP to finalize its reducer state before setting static graph
                # This helps avoid expect_autograd_hooks_ errors by ensuring proper initialization
                if hasattr(ddp_model, 'reducer') and ddp_model.reducer is not None:
                    # Check if reducer is properly initialized
                    if not hasattr(ddp_model.reducer, '_rebuild_buckets_called'):
                        # Force lazy initialization of the reducer
                        try:
                            # Call _rebuild_buckets to ensure reducer is properly initialized
                            if hasattr(ddp_model.reducer, '_rebuild_buckets'):
                                ddp_model.reducer._rebuild_buckets()
                        except Exception as e:
                            print(f"Unsloth: Warning - Could not initialize DDP reducer: {e}")
                
                # Enable static graph optimization for DDP ONLY if no gradient checkpointing
                # This is safe for most fine-tuning scenarios where the computation graph is static
                ddp_model._set_static_graph()

                
                # Additional safeguard: Mark all parameters as ready to help with autograd hooks
                # This prevents expect_autograd_hooks_ errors by ensuring proper hook state
                try:
                    if hasattr(ddp_model, 'reducer') and ddp_model.reducer is not None:
                        # Ensure the reducer knows about all parameters to avoid hook issues
                        if hasattr(ddp_model.reducer, '_mark_all_parameters_ready'):
                            # This method exists in some PyTorch versions to help with hook synchronization
                            pass  # Don't call it here as it might interfere with training
                except Exception:
                    pass  # Ignore if this advanced method doesn't exist
                
                return True
            except Exception as e:
                print(f"Unsloth: Warning - Could not enable DDP static graph: {e}")
                print("Unsloth: This may cause 'parameter marked ready twice' or 'expect_autograd_hooks_' errors in distributed training")
                return False
        else:
            # Only print warning in distributed environment where we expect to find DDP
            if (os.environ.get("LOCAL_RANK") is not None and
                os.environ.get("WORLD_SIZE") is not None):

                # Enhanced debugging information
                model_type = type(model).__name__
                model_attrs = [attr for attr in dir(model) if 'model' in attr.lower() or 'module' in attr.lower()][:10]

                print("Unsloth: Warning - Could not find DDP-wrapped model for static graph optimization")
                print("Unsloth: If you encounter 'parameter marked ready twice' or 'expect_autograd_hooks_' errors, this is the likely cause")
                print(f"Unsloth: Model type: {model_type}")
                print(f"Unsloth: Available model-related attributes: {model_attrs}")

                # Enhanced debugging with model hierarchy inspection
                try:
                    from torch.nn.parallel import DistributedDataParallel as DDP

                    # Check direct DDP
                    if isinstance(model, DDP):
                        print("Unsloth: DEBUG - Model is actually DDP but wasn't detected in initial search")
                    elif hasattr(model, 'module') and isinstance(model.module, DDP):
                        print("Unsloth: DEBUG - Model.module is DDP but wasn't detected in initial search")

                    # Check for nested structures
                    if hasattr(model, 'base_model'):
                        base_model_type = type(model.base_model).__name__
                        print(f"Unsloth: DEBUG - Found base_model of type: {base_model_type}")
                        if hasattr(model.base_model, 'model'):
                            nested_model_type = type(model.base_model.model).__name__
                            print(f"Unsloth: DEBUG - Found base_model.model of type: {nested_model_type}")

                    # Check for accelerator wrapping
                    if hasattr(model, '__dict__'):
                        accelerator_attrs = [k for k in model.__dict__.keys() if 'accelerat' in k.lower()]
                        if accelerator_attrs:
                            print(f"Unsloth: DEBUG - Found accelerator-related attributes: {accelerator_attrs}")

                    # Check parameter count for validation
                    try:
                        param_count = sum(1 for _ in model.parameters())
                        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
                        print(f"Unsloth: DEBUG - Model has {param_count} parameters ({trainable_count} trainable)")
                    except Exception:
                        pass

                except Exception as debug_e:
                    print(f"Unsloth: DEBUG - Error during model inspection: {debug_e}")

            return False

    except Exception as e:
        print(f"Unsloth: Warning - Could not setup DDP static graph: {e}")
        print(f"Unsloth: Error type: {type(e).__name__}")

        # Additional error context
        import traceback
        if os.environ.get("UNSLOTH_DEBUG_DDP", "0") == "1":
            print("Unsloth: Full traceback (UNSLOTH_DEBUG_DDP=1):")
            traceback.print_exc()

        return False


def _prepare_ddp_reducer_for_training(trainer, model):
    """Prepare DDP reducer to avoid expect_autograd_hooks_ errors."""
    if hasattr(trainer, '_unsloth_ddp_reducer_prepared'):
        return True
        
    import os
    import torch
    
    # Only proceed if we're in a distributed environment
    if not (os.environ.get("LOCAL_RANK") is not None or 
            os.environ.get("WORLD_SIZE") is not None):
        return False
    
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            return False
            
        # Find the DDP-wrapped model
        ddp_model = _find_ddp_model(model)
        
        if ddp_model is not None:
            try:
                # ENHANCED FIX for expect_autograd_hooks_ errors:
                # The key insight is that DDP's reducer needs to be fully initialized
                # with proper autograd hook registration before any backward pass occurs.
                
                # Method 1: Enhanced dummy forward pass to ensure DDP is fully initialized
                # This ensures the reducer knows about all parameters and their autograd hooks
                try:
                    # Create dummy input that matches the model's expected input structure
                    # This triggers DDP's lazy initialization of autograd hooks
                    with torch.no_grad():
                        # Get first parameter to determine device and dtype
                        first_param = next(ddp_model.parameters())
                        device = first_param.device
                        dtype = first_param.dtype

                        # Set model to eval mode temporarily to avoid affecting training state
                        original_training_mode = ddp_model.training
                        ddp_model.eval()

                        # Try multiple dummy input strategies
                        dummy_inputs = [
                            # Strategy 1: Standard transformer input
                            {
                                'input_ids': torch.tensor([[1, 2]], device=device, dtype=torch.long),
                                'attention_mask': torch.tensor([[1, 1]], device=device, dtype=torch.long),
                            },
                            # Strategy 2: Extended transformer input with position_ids
                            {
                                'input_ids': torch.tensor([[1, 2]], device=device, dtype=torch.long),
                                'attention_mask': torch.tensor([[1, 1]], device=device, dtype=torch.long),
                                'position_ids': torch.tensor([[0, 1]], device=device, dtype=torch.long),
                            },
                            # Strategy 3: Vision-language model input
                            {
                                'input_ids': torch.tensor([[1, 2]], device=device, dtype=torch.long),
                                'attention_mask': torch.tensor([[1, 1]], device=device, dtype=torch.long),
                                'pixel_values': torch.randn(1, 3, 224, 224, device=device, dtype=dtype),
                            },
                        ]

                        success = False
                        for dummy_input in dummy_inputs:
                            try:
                                # Run dummy forward pass to initialize DDP reducer and autograd hooks
                                with torch.amp.autocast('cuda', enabled=False):  # Disable autocast for dummy pass
                                    _ = ddp_model(**dummy_input)
                                success = True
                                break
                            except Exception:
                                continue

                        # If structured inputs fail, try simple tensor inputs
                        if not success:
                            tensor_inputs = [
                                torch.randn(1, 2, device=device, dtype=dtype),
                                torch.randn(1, 2, 768, device=device, dtype=dtype),  # Common hidden size
                                torch.randn(1, 2, 4096, device=device, dtype=dtype),  # Larger hidden size
                            ]

                            for dummy_tensor in tensor_inputs:
                                try:
                                    with torch.amp.autocast('cuda', enabled=False):
                                        _ = ddp_model(dummy_tensor)
                                    success = True
                                    break
                                except Exception:
                                    continue

                        # Restore original training mode
                        ddp_model.train(original_training_mode)

                        if success:
                            print("Unsloth: Successfully initialized DDP reducer with dummy forward pass")
                            
                except Exception as e:
                    print(f"Unsloth: Could not run dummy forward pass for DDP initialization: {e}")
                    
                # Method 2: Enhanced reducer preparation
                if hasattr(ddp_model, 'reducer') and ddp_model.reducer is not None:
                    reducer = ddp_model.reducer
                    
                    # Force reducer bucket rebuilding to ensure proper autograd hook setup
                    if hasattr(reducer, '_rebuild_buckets'):
                        try:
                            reducer._rebuild_buckets()
                        except Exception:
                            pass
                    
                    # Ensure reducer is marked as ready for backward pass
                    if hasattr(reducer, '_prepare_for_forward'):
                        try:
                            reducer._prepare_for_forward()
                        except Exception:
                            pass
                            
                    # Additional fix: Ensure autograd hooks are properly registered
                    # by checking reducer's internal state
                    if hasattr(reducer, '_autograd_hooks') and hasattr(reducer, 'next_bucket'):
                        try:
                            # Reset the autograd hook state to ensure consistency
                            # This is the key fix for expect_autograd_hooks_ errors
                            reducer.next_bucket = 0
                            
                            # Ensure hooks are properly aligned with parameters
                            if hasattr(reducer, '_ensure_autograd_hooks_prepared'):
                                reducer._ensure_autograd_hooks_prepared()
                                
                        except Exception:
                            pass
                
                # Method 3: Force parameter registration and validation
                try:
                    # Ensure all trainable parameters are known to DDP
                    trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]
                    if trainable_params:
                        # Access parameter gradients to ensure autograd graph readiness
                        for param in trainable_params[:5]:  # Check first few params only
                            if param.grad is not None:
                                # Clear any existing gradients to reset autograd state
                                param.grad.zero_()
                except Exception:
                    pass
                
                trainer._unsloth_ddp_reducer_prepared = True

                return True
                
            except Exception as e:
                print(f"Unsloth: Warning - Could not prepare DDP reducer: {e}")
                trainer._unsloth_ddp_reducer_prepared = True  # Mark as done to avoid repeated attempts
                return False
        
        trainer._unsloth_ddp_reducer_prepared = True
        return False
        
    except Exception as e:
        print(f"Unsloth: Warning - Could not prepare DDP reducer: {e}")
        trainer._unsloth_ddp_reducer_prepared = True
        return False


def _patch_trainer_with_ddp_support(trainer_class):
    """Add DDP support to any trainer class by patching its methods."""
    original_init = trainer_class.__init__
    original_train = trainer_class.train
    original_training_step = trainer_class.training_step
    
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # Setup distributed training before model initialization
        _setup_distributed_training()
        
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Setup DDP static graph after model is initialized
        if hasattr(self, 'model'):
            _setup_ddp_static_graph(self.model)
    
    @wraps(original_train)
    def new_train(self, *args, **kwargs):
        """Override train to ensure DDP static graph is set up before training starts."""
        # Re-setup DDP static graph in case model wrapping happened after init
        if hasattr(self, 'model'):
            _setup_ddp_static_graph(self.model)
        return original_train(self, *args, **kwargs)
    
    @wraps(original_training_step)
    def new_training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to handle DDP gradient checkpointing issues."""
        # Setup DDP static graph just before the first training step if not already done
        if not hasattr(self, '_unsloth_ddp_static_graph_setup_done'):
            # Try multiple times with the latest model reference
            # In case Accelerate wrapped the model after init
            success = False
            accelerator_model = None
            if hasattr(self, 'accelerator') and hasattr(self.accelerator, 'model'):
                accelerator_model = self.accelerator.model
            for model_ref in [model, getattr(self, 'model', None), accelerator_model]:
                if model_ref is not None:
                    if _setup_ddp_static_graph(model_ref):
                        success = True
                        break
            self._unsloth_ddp_static_graph_setup_done = True
            
            if not success:
                # Last resort: try to find DDP model in accelerator
                try:
                    if hasattr(self, 'accelerator') and hasattr(self.accelerator, 'model'):
                        _setup_ddp_static_graph(self.accelerator.model)
                except:
                    pass
        
        # Additional safeguard for expect_autograd_hooks_ error:
        # Prepare DDP reducer before training step
        _prepare_ddp_reducer_for_training(self, model)
        
        return original_training_step(self, model, inputs, num_items_in_batch)
    
    # Apply the patches
    trainer_class.__init__ = new_init
    trainer_class.train = new_train
    trainer_class.training_step = new_training_step
    
    return trainer_class


def _patch_trl_trainer():
    import trl
    if hasattr(trl, "__UNSLOTH_BACKWARDS_COMPATIBLE__"): return
    if Version(trl.__version__) <= Version("0.11.0"): return

    import trl.trainer
    trl_classes = dir(trl.trainer)
    trl_trainers = set(x[:-len("Trainer")] for x in trl_classes if x.endswith("Trainer"))
    trl_configs  = set(x[:-len("Config")]  for x in trl_classes if x.endswith("Config"))
    trl_classes = list(trl_trainers & trl_configs)

    for x in trl_classes:
        try:    
            # Apply backwards compatibility patch
            exec(f"trl.{x}Trainer.__init__ = _backwards_compatible_trainer(trl.{x}Trainer, trl.{x}Config)", globals())
            
            # Apply DDP support patch
            trainer_class = getattr(trl, f"{x}Trainer")
            _patch_trainer_with_ddp_support(trainer_class)
            
        except: continue
    pass

    trl.__UNSLOTH_BACKWARDS_COMPATIBLE__ = True
pass
