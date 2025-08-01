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
        print(
            "Unsloth: Using our custom gradient accumulation fixed trainer, which is not feature complete.\n"\
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
            print(f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}.")
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
            for model_ref in [model, getattr(self, 'model', None), getattr(self, 'accelerator', {}).get('model', None)]:
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
                print(f"Unsloth: Initialized distributed training on rank {local_rank}")
                
                # Set up proper device mapping for this rank
                if torch.cuda.is_available():
                    device = torch.device(f"cuda:{local_rank}")
                    print(f"Unsloth: Using device {device} for rank {local_rank}")
                
        except Exception as e:
            print(f"Unsloth: Failed to initialize distributed training: {e}")
            print("Unsloth: Falling back to single-GPU training")
    elif multi_gpu_config["supports_multi_gpu"] and multi_gpu_config["enable_multi_gpu"]:
        print(f"Unsloth: Multi-GPU setup detected ({multi_gpu_config['device_count']} GPUs) but not using distributed training")
        print("Unsloth: For true distributed training, launch with: torchrun --nproc_per_node={} your_script.py".format(multi_gpu_config['device_count']))


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
            
        # Check common attribute names where DDP models might be nested
        for attr_name in ['module', 'model', 'base_model', '_orig_mod', '_module', '_model']:
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
    """Setup DDP static graph to fix gradient checkpointing issues."""
    import os
    import torch
    
    # Allow users to disable the fix if needed
    if os.environ.get("UNSLOTH_DISABLE_DDP_STATIC_GRAPH", "0") == "1":
        print("Unsloth: DDP static graph optimization disabled by environment variable")
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
                    
                # Enable static graph optimization for DDP
                # This is safe for most fine-tuning scenarios where the computation graph is static
                ddp_model._set_static_graph()
                print("Unsloth: Enabled DDP static graph optimization to fix gradient checkpointing issues")
                return True
            except Exception as e:
                print(f"Unsloth: Warning - Could not enable DDP static graph: {e}")
                print("Unsloth: This may cause 'parameter marked ready twice' errors in distributed training")
                return False
        else:
            # Only print warning in distributed environment where we expect to find DDP
            if (os.environ.get("LOCAL_RANK") is not None and 
                os.environ.get("WORLD_SIZE") is not None):
                print("Unsloth: Warning - Could not find DDP-wrapped model for static graph optimization")
                print("Unsloth: If you encounter 'parameter marked ready twice' errors, this is the likely cause")
            return False
            
    except Exception as e:
        print(f"Unsloth: Warning - Could not setup DDP static graph: {e}")
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
            for model_ref in [model, getattr(self, 'model', None), getattr(self, 'accelerator', {}).get('model', None)]:
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
