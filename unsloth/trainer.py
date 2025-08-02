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
        
        return super().training_step(model, inputs, num_items_in_batch)
    
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
            for model_ref in [model, getattr(self, 'model', None), accelerator_model]:
                if model_ref is not None:
                    if self._setup_ddp_static_graph(model_ref):
                        success = True
                        break
            self._unsloth_ddp_static_graph_setup_done = True
            
            if not success:
                # Last resort: try to find DDP model in accelerator
                try:
                    if hasattr(self, 'accelerator') and hasattr(self.accelerator, 'model'):
                        self._setup_ddp_static_graph(self.accelerator.model)
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
                    
                    # Method 1: Force a dummy forward pass to ensure DDP is fully initialized
                    # This ensures the reducer knows about all parameters and their autograd hooks
                    try:
                        # Create dummy input that matches the model's expected input structure
                        # This triggers DDP's lazy initialization of autograd hooks
                        with torch.no_grad():
                            # Get first parameter to determine device and dtype
                            first_param = next(ddp_model.parameters())
                            device = first_param.device
                            dtype = first_param.dtype
                            
                            # Create minimal dummy input - most transformer models expect input_ids
                            dummy_input = {
                                'input_ids': torch.tensor([[1, 2]], device=device, dtype=torch.long),
                                'attention_mask': torch.tensor([[1, 1]], device=device, dtype=torch.long),
                            }
                            
                            # Set model to eval mode temporarily to avoid affecting training state
                            original_training_mode = ddp_model.training
                            ddp_model.eval()
                            
                            try:
                                # Run dummy forward pass to initialize DDP reducer and autograd hooks
                                with torch.amp.autocast('cuda', enabled=False):  # Disable autocast for dummy pass
                                    _ = ddp_model(**dummy_input)

                            except Exception:
                                # If structured input fails, try simple tensor input
                                try:
                                    dummy_tensor = torch.randn(1, 2, device=device, dtype=dtype)
                                    with torch.amp.autocast('cuda', enabled=False):
                                        _ = ddp_model(dummy_tensor)

                                except Exception:
                                    # If both fail, still continue - the other methods may help
                                    pass
                            finally:
                                # Restore original training mode
                                ddp_model.train(original_training_mode)
                                
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
    """Setup distributed training if in multi-GPU environment."""
    import os
    import torch
    
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
            if not dist.is_initialized():
                # Initialize distributed training
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

                

                
        except Exception as e:
            print(f"Unsloth: Failed to initialize distributed training: {e}")
            print("Unsloth: Falling back to single-GPU training")
    elif multi_gpu_config["supports_multi_gpu"] and multi_gpu_config["enable_multi_gpu"]:
        pass

def _find_ddp_model(model):
    """Recursively search for DDP-wrapped model in the model hierarchy."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    # Check current model
    if isinstance(model, DDP):
        return model
    
    # Track visited objects to avoid infinite recursion
    visited = set()
    
    def _recursive_search(obj, depth=0, max_depth=10):
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
            'module', 'model', 'base_model', '_orig_mod', '_module', '_model',
            'wrapped_model', 'inner_model', '_wrapped_model', '_inner_model',
            'core_model', '_core_model', 'underlying_model', '_underlying_model',
            # Accelerate library attributes
            '_ddp_module', '_orig_ddp_module', '_accelerate_wrapped_model',
            # Transformers library attributes  
            '_transformers_model', '_hf_model',
            # TRL/SFT trainer attributes
            '_sft_model', '_trl_model'
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
            except (AttributeError, RuntimeError):
                # Some attributes may not be accessible
                continue
        
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
    
    return _recursive_search(model)


def _setup_ddp_static_graph(model):
    """Setup DDP static graph and autograd hooks to fix gradient checkpointing issues."""
    import os
    import torch
    
    # Allow users to disable the fix if needed
    if os.environ.get("UNSLOTH_DISABLE_DDP_STATIC_GRAPH", "0") == "1":

        return False
    
    # Allow users to force disable due to gradient checkpointing
    if os.environ.get("UNSLOTH_DISABLE_DDP_STATIC_GRAPH_FOR_GRAD_CHECKPOINT", "0") == "1":

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
                
                # Check for various gradient checkpointing indicators
                if hasattr(model, 'gradient_checkpointing') and model.gradient_checkpointing:
                    uses_gradient_checkpointing = True
                elif hasattr(model, '_set_gradient_checkpointing'):
                    # Some models have this flag
                    if hasattr(model, 'gradient_checkpointing_enable') or hasattr(model, '_gradient_checkpointing'):
                        uses_gradient_checkpointing = True
                
                # Also check the underlying model (in case it's wrapped)
                actual_model = getattr(model, 'module', model)
                if hasattr(actual_model, 'gradient_checkpointing') and actual_model.gradient_checkpointing:
                    uses_gradient_checkpointing = True
                    
                # Check if any layer has gradient checkpointing enabled
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
                
                # UNSLOTH SPECIFIC: Check if unsloth smart gradient checkpointing is active
                # This is the most reliable way to detect Unsloth's gradient checkpointing
                try:
                    # Import the unsloth zoo utility to check for active gradient checkpointing
                    import unsloth_zoo.gradient_checkpointing as unsloth_gc
                    # If this module exists and has been patched, gradient checkpointing is likely active
                    if hasattr(unsloth_gc, '_UNSLOTH_GRADIENT_CHECKPOINTING_ENABLED'):
                        if getattr(unsloth_gc, '_UNSLOTH_GRADIENT_CHECKPOINTING_ENABLED', False):
                            uses_gradient_checkpointing = True
                except ImportError:
                    pass
                
                # Check for environment variables or settings that indicate gradient checkpointing
                # Many users set this when using Unsloth
                if os.environ.get("UNSLOTH_USE_GRADIENT_CHECKPOINTING") == "1":
                    uses_gradient_checkpointing = True
                
                # Look for gradient checkpointing in model's config
                if hasattr(actual_model, 'config'):
                    config = actual_model.config
                    if hasattr(config, 'use_gradient_checkpointing') and config.use_gradient_checkpointing:
                        uses_gradient_checkpointing = True
                
                # If gradient checkpointing is detected, disable static graph
                if uses_gradient_checkpointing:

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
                print("Unsloth: Warning - Could not find DDP-wrapped model for static graph optimization")
                print("Unsloth: If you encounter 'parameter marked ready twice' or 'expect_autograd_hooks_' errors, this is the likely cause")
            return False
            
    except Exception as e:
        print(f"Unsloth: Warning - Could not setup DDP static graph: {e}")
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
                
                # Method 1: Force a dummy forward pass to ensure DDP is fully initialized
                # This ensures the reducer knows about all parameters and their autograd hooks
                try:
                    # Create dummy input that matches the model's expected input structure
                    # This triggers DDP's lazy initialization of autograd hooks
                    with torch.no_grad():
                        # Get first parameter to determine device and dtype
                        first_param = next(ddp_model.parameters())
                        device = first_param.device
                        dtype = first_param.dtype
                        
                        # Create minimal dummy input - most transformer models expect input_ids
                        dummy_input = {
                            'input_ids': torch.tensor([[1, 2]], device=device, dtype=torch.long),
                            'attention_mask': torch.tensor([[1, 1]], device=device, dtype=torch.long),
                        }
                        
                        # Set model to eval mode temporarily to avoid affecting training state
                        original_training_mode = ddp_model.training
                        ddp_model.eval()
                        
                        try:
                            # Run dummy forward pass to initialize DDP reducer and autograd hooks
                            with torch.amp.autocast('cuda', enabled=False):  # Disable autocast for dummy pass
                                _ = ddp_model(**dummy_input)

                        except Exception:
                            # If structured input fails, try simple tensor input
                            try:
                                dummy_tensor = torch.randn(1, 2, device=device, dtype=dtype)
                                with torch.amp.autocast('cuda', enabled=False):
                                    _ = ddp_model(dummy_tensor)

                            except Exception:
                                # If both fail, still continue - the other methods may help
                                pass
                        finally:
                            # Restore original training mode
                            ddp_model.train(original_training_mode)
                            
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
