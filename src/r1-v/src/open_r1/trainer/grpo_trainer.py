# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import random

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from accelerate.utils.other import is_compiled_module

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from trl.import_utils import is_vllm_available

from qwen_vl_utils import process_vision_info
from experiment_config import ExperimentConfig, task_type_from_problem_type
from welford import BayesianWelford
from temporal_mask import build_temporal_frame_mask  # [FIX-3] 帧级部分 mask

import copy

# vLLM colocate mode imports (requires vllm >= 0.8.0 for sleep/wake API)
if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.distributed import parallel_state as vllm_parallel_state


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            # [OOM-FIX] Load ref_model to CPU first to avoid the 14GB spike when
            # from_pretrained places the full 7B model on one GPU before DeepSpeed
            # shards it. Do NOT use device_map (incompatible with ZeRO-3).
            # prepare_deepspeed() below will properly shard and manage it.
            ref_init_kwargs = dict(model_init_kwargs)
            ref_init_kwargs["low_cpu_mem_usage"] = False
            ref_init_kwargs.pop("device_map", None)
            ref_init_kwargs.pop("use_cache", None)
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **ref_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **ref_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **ref_init_kwargs)
            else:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **ref_init_kwargs)
            self.ref_model = self.ref_model.cpu()  # ensure CPU before DeepSpeed takes over
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or True:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temporal = script_args.temporal
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1,
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,  
            temperature=1,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.len_control = script_args.len_control
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # ---- Experiment Config (4D Regularization) ----
        if script_args is not None and getattr(script_args, 'experiment_config', None):
            self.exp_config = ExperimentConfig.from_yaml(script_args.experiment_config)
        else:
            self.exp_config = ExperimentConfig()  # default = baseline, all flags OFF

        if self.exp_config.enable_length_penalty:
            self.welford = BayesianWelford(
                sigma_prior=self.exp_config.welford_sigma_prior,
                k_prior=self.exp_config.welford_k_prior,
            )
        else:
            self.welford = None

        # [FIX-9] Override len_control from experiment config if specified.
        # When D3 (length_penalty) is ON, the original hardcoded len_control
        # (320-512 range, +0.2 bonus) must be OFF to avoid double-penalty.
        if self.exp_config.override_len_control is not None:
            if self.len_control != self.exp_config.override_len_control:
                print(
                    f"[FIX-9] Experiment config overrides --len_control from {self.len_control} "
                    f"to {self.exp_config.override_len_control} "
                    f"(experiment: {self.exp_config.experiment_name})"
                )
                self.len_control = self.exp_config.override_len_control

        if self.accelerator.is_main_process:
            print(self.exp_config.summary())

        # ============================================================== #
        # vLLM Colocate Mode Initialization                              #
        # When use_vllm=True, init a vLLM LLM engine that shares the     #
        # same GPUs as training (TP=4). Memory handoff is managed by     #
        # sleep/wake API (requires vllm >= 0.8.0).                      #
        # ============================================================== #
        self.use_vllm = getattr(args, 'use_vllm', False)
        # vllm_tensor_parallel_size may be on script_args (our custom arg)
        # or on args (GRPOConfig). Try both.
        self.vllm_tp = getattr(args, 'vllm_tensor_parallel_size', None)
        if self.vllm_tp is None and script_args is not None:
            self.vllm_tp = getattr(script_args, 'vllm_tensor_parallel_size', 1)
        if self.vllm_tp is None:
            self.vllm_tp = 1
        self.vllm_gpu_mem_util = getattr(args, 'vllm_gpu_memory_utilization', 0.3)
        self._vllm_engine = None
        self._last_vllm_weight_step = -1  # track if weights need re-sync

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available but --use_vllm is True. "
                    "Install vllm>=0.8.0: pip install 'vllm>=0.8.0'"
                )
            self._init_vllm(model_id=model_id, max_pixels=max_pixels, min_pixels=min_pixels)

    # ------------------------------------------------------------------ #
    # [FIX-5] Welford checkpoint save / restore                          #
    # ------------------------------------------------------------------ #
    def _save_checkpoint(self, model, trial, metrics=None):
        """Save Welford state alongside the normal checkpoint."""
        super()._save_checkpoint(model, trial, metrics=metrics)
        if self.welford is not None and self.accelerator.is_main_process:
            ckpt_dir = self._get_output_dir(trial)
            welford_path = os.path.join(ckpt_dir, "welford_state.pt")
            torch.save(self.welford.state_dict(), welford_path)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore Welford state when resuming."""
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        if self.welford is not None:
            welford_path = os.path.join(resume_from_checkpoint, "welford_state.pt")
            if os.path.isfile(welford_path):
                state = torch.load(welford_path, map_location="cpu")
                self.welford.load_state_dict(state)
                if self.accelerator.is_main_process:
                    print(f"[Welford] Restored state from {welford_path} "
                          f"(count={self.welford.count}, mean={self.welford.mean:.1f})")

    # ------------------------------------------------------------------ #
    # vLLM Colocate Mode: Engine Lifecycle                               #
    # Requires vllm >= 0.8.0 for sleep()/wake_up() memory handoff       #
    # ------------------------------------------------------------------ #
    def _init_vllm(self, model_id, max_pixels=12845056, min_pixels=3136):
        """Initialize the vLLM LLM engine in colocate mode (TP across training GPUs).

        In colocate mode, vLLM shares the same GPUs as the training process.
        DeepSpeed ZeRO-3 controls the distributed environment, so vLLM uses
        'external_launcher' to avoid re-initializing NCCL/torch.distributed.

        IMPORTANT: With external_launcher, ALL ranks must create their own LLM
        instance. Rank 0 becomes the driver, other ranks become TP workers.
        Only rank 0 calls generate(); the others participate via torch.distributed.

        Memory budget:
          - vllm_gpu_memory_utilization=0.3 -> 30% of VRAM for KV cache
          - Remaining 70% for ZeRO-3 model states, optimizer, gradients
        """
        # ALL ranks create LLM - external_launcher uses the torchrun distributed env
        self._vllm_engine = LLM(
            model=model_id,
            tensor_parallel_size=self.vllm_tp,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=self.vllm_gpu_mem_util,
            dtype=torch.bfloat16,
            enforce_eager=True,  # disable CUDA graphs for sleep/wake compatibility
            enable_prefix_caching=True,
            seed=42,  # required by external_launcher for reproducible sampling
            max_model_len=self.max_prompt_length + self.max_completion_length,
            mm_processor_kwargs={
                "max_pixels": max_pixels,
                "min_pixels": min_pixels,
            },
        )
        self._vllm_sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            max_tokens=self.max_completion_length,
        )

        if self.accelerator.is_main_process:
            print(f"[vLLM Colocate] Engine initialized: TP={self.vllm_tp}, "
                  f"gpu_mem_util={self.vllm_gpu_mem_util}, "
                  f"max_model_len={self.max_prompt_length + self.max_completion_length}")

        self.accelerator.wait_for_everyone()

    def _vllm_sleep(self):
        """Put vLLM engine to sleep, releasing GPU memory for training.

        After calling this, the KV cache and internal buffers are freed,
        allowing ZeRO-3 to use the full GPU memory for the training step.
        Only the main process drives the sleep command; TP workers follow.
        """
        if self._vllm_engine is not None and self.accelerator.is_main_process:
            # vLLM >= 0.8.0: sleep releases KV cache and intermediate tensors
            self._vllm_engine.sleep(level=1)
            if os.getenv("DEBUG_MODE") == "true":
                print("[vLLM Colocate] Engine sleeping -- GPU memory released for training")

    def _vllm_wake_up(self):
        """Wake up vLLM engine, re-allocating GPU memory for generation.

        Call this before vLLM generation. After wake_up, the engine
        re-allocates KV cache space from the gpu_memory_utilization budget.
        """
        if self._vllm_engine is not None and self.accelerator.is_main_process:
            self._vllm_engine.wake_up()
            if os.getenv("DEBUG_MODE") == "true":
                print("[vLLM Colocate] Engine woke up -- GPU memory reserved for generation")

    def _sync_vllm_weights(self):
        """Synchronize training model weights to the vLLM engine.

        In ZeRO-3, parameters are sharded across GPUs. We must gather them
        first using unwrap_model_for_generation, then load into vLLM.
        Only the main process does the actual weight loading; TP workers
        pick up the weights through their own all-gather during forward.
        """
        if self._vllm_engine is None:
            return
        # Only sync when weights have changed (new training step)
        if self.state.global_step == self._last_vllm_weight_step:
            return

        # Gather full weights from ZeRO-3 shards
        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            gather_deepspeed3_params=True,
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                state_dict = unwrapped_model._orig_mod.state_dict()
            else:
                state_dict = unwrapped_model.state_dict()

        if self.accelerator.is_main_process:
            llm_model = (
                self._vllm_engine.llm_engine.model_executor
                .driver_worker.model_runner.model
            )
            llm_model.load_weights(state_dict.items())

        self._last_vllm_weight_step = self.state.global_step

        if self.accelerator.is_main_process and os.getenv("DEBUG_MODE") == "true":
            print(f"[vLLM Colocate] Weights synced at step {self.state.global_step}")

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

                
        
        input_copy = copy.deepcopy(inputs[0]['prompt'])
        
        input_copy = self.remove_none_from_data(input_copy)
        
        if inputs[0]['data_type'] == 'image':
            input_copy[0]['content'][0]['image'] = os.getcwd() + "/Video-R1-data" + inputs[0]['path'][1:] 
        elif inputs[0]['data_type'] == 'video':
            input_copy[0]['content'][0]['video'] = os.getcwd() + "/Video-R1-data" + inputs[0]['path'][1:] 
            
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        except Exception as e:
            print(f"process_vision_info error, using fixed data, {e}")
            if inputs[0]['data_type'] == 'image':
                input_copy[0]['content'][0]['image'] = os.getcwd() + "/Video-R1-data" + '/Math/Multimath-300k/17ff4c7d14c388134de02381b1fc2824.png'
            elif inputs[0]['data_type'] == 'video':
                input_copy[0]['content'][0]['video'] = os.getcwd() + "/Video-R1-data" + '/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4'
                
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        
        
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        
        prompt_inputs = super()._prepare_inputs(prompt_inputs)


        # fix prompt_inputs["input_ids"] length issue
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            
        if self.temporal and video_inputs:
            indices = torch.randperm(video_inputs[0].size(0))
            shuffled_video_inputs = [video_inputs[0][indices]]
            shuffled_prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=shuffled_video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            shuffled_prompt_inputs = super()._prepare_inputs(shuffled_prompt_inputs)
            shuffled_prompt_ids, shuffled_prompt_mask = shuffled_prompt_inputs["input_ids"], shuffled_prompt_inputs["attention_mask"]
            if self.max_prompt_length is not None:
                shuffled_prompt_ids = shuffled_prompt_ids[:, -self.max_prompt_length :]
                shuffled_prompt_mask = shuffled_prompt_mask[:, -self.max_prompt_length :]
        
        
        # Generate completions
        # ------------------------------------------------------------------ #
        # vLLM Colocate Mode: wake up engine, sync weights, generate,      #
        # then sleep to release GPU memory for training.                    #
        # ------------------------------------------------------------------ #
        if self.use_vllm:
            # --- vLLM Colocate Generation ---
            # 1. Wake up vLLM engine (re-allocate KV cache)
            self._vllm_wake_up()
            # 2. Sync latest training weights into vLLM
            self._sync_vllm_weights()

            # Prepare multimodal data for vLLM
            data_type = inputs[0]['data_type']
            mm_data = [[data_type, image_inputs if image_inputs else video_inputs]]

            # Gather prompts across DP ranks for batched vLLM generation
            from accelerate.utils import gather_object
            all_prompts_text = gather_object(prompts_text)
            all_mm_data = gather_object(mm_data)

            # Build vLLM multimodal inputs: {"prompt": text, "multi_modal_data": {type: data}}
            all_multimodal_inputs = []
            for prompt, mm_item in zip(all_prompts_text, all_mm_data):
                all_multimodal_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {mm_item[0]: mm_item[1]}
                })

            # ALL ranks call generate (external_launcher requires all ranks
            # to participate in NCCL forward pass).
            # In external_launcher TP mode, all ranks get identical results,
            # so we don't need broadcast — each rank extracts and slices locally.
            sampling_params = copy.deepcopy(self._vllm_sampling_params)
            sampling_params.n = self.num_generations
            outputs = self._vllm_engine.generate(
                all_multimodal_inputs,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            # All ranks extract results (they're identical across TP ranks)
            completion_ids_list = [
                out.token_ids
                for completion in outputs
                for out in completion.outputs
            ]

            # Slice per-process
            process_slice = slice(
                self.accelerator.process_index * len(prompts) * self.num_generations,
                (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
            )
            completion_ids_list = completion_ids_list[process_slice]

            # Convert to tensors and pad
            from trl.trainer.utils import pad as trl_pad
            device = self.accelerator.device
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
            completion_ids = trl_pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            prompt_length = prompt_ids.size(1)
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

            # Temporal: shuffled generation via vLLM
            if self.temporal and video_inputs:
                shuffled_all_mm_data_raw = gather_object(
                    [[self.accelerator.process_index, data_type, shuffled_video_inputs[0]]]
                    if video_inputs else [None]
                )
                shuffled_all_mm_data = [x for x in shuffled_all_mm_data_raw if x is not None]

                # Build shuffled multimodal inputs referencing the original prompts
                shuffled_all_multimodal_inputs = []
                for mm_item in shuffled_all_mm_data:
                    shuffled_all_multimodal_inputs.append({
                        "prompt": all_prompts_text[mm_item[0]],
                        "multi_modal_data": {mm_item[1]: mm_item[2]}
                    })

                # ALL ranks call generate for temporal (same reason: NCCL sync)
                if shuffled_all_multimodal_inputs:
                    shuffled_sampling_params = copy.deepcopy(self._vllm_sampling_params)
                    shuffled_sampling_params.n = self.shuffled_num_generations
                    shuffled_outputs = self._vllm_engine.generate(
                        shuffled_all_multimodal_inputs,
                        sampling_params=shuffled_sampling_params,
                        use_tqdm=False,
                    )
                    # All ranks extract results (identical across TP ranks)
                    shuffled_completion_ids_list = [
                        out.token_ids
                        for completion in shuffled_outputs
                        for out in completion.outputs
                    ]
                else:
                    shuffled_completion_ids_list = []

                if shuffled_all_multimodal_inputs:
                    # Slice by process index (no broadcast needed — all ranks have same data)
                    process_id_list = []
                    for mm_item in shuffled_all_mm_data:
                        process_id_list += [mm_item[0]] * self.shuffled_num_generations

                    cur_shuffled_ids = []
                    for i, pid in enumerate(process_id_list):
                        if self.accelerator.process_index == pid:
                            cur_shuffled_ids.append(
                                torch.tensor(shuffled_completion_ids_list[i], device=device)
                            )
                    if cur_shuffled_ids:
                        shuffled_completion_ids = trl_pad(
                            cur_shuffled_ids, padding_value=self.processing_class.pad_token_id
                        )
                        shuffled_prompt_ids = shuffled_prompt_ids.repeat_interleave(
                            self.shuffled_num_generations, dim=0
                        )
                        shuffled_prompt_completion_ids = torch.cat(
                            [shuffled_prompt_ids, shuffled_completion_ids], dim=1
                        )
                        shuffled_prompt_length = shuffled_prompt_ids.size(1)
                        shuffled_prompt_mask = prompt_mask[:len(cur_shuffled_ids)]

            # 3. Sleep vLLM to release GPU memory for training
            self._vllm_sleep()

        else:
            # --- HF generate() (original path) ---
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
                prompt_length = prompt_ids.size(1)
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
                prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
                
                if self.temporal:
                    
                    if video_inputs:
            
                        shuffled_prompt_completion_ids = unwrapped_model.generate(**shuffled_prompt_inputs, generation_config=self.shuffled_generation_config)
                        shuffled_prompt_length = shuffled_prompt_ids.size(1)
                        shuffled_prompt_ids = shuffled_prompt_completion_ids[:, :shuffled_prompt_length]
                        shuffled_completion_ids = shuffled_prompt_completion_ids[:, shuffled_prompt_length:]
                        shuffled_prompt_mask = prompt_mask.repeat_interleave(self.shuffled_num_generations, dim=0)
                        
                    else:
                        
                        shuffled_prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.dummy_generation_config)

        # [FIX-7] Debug prints gated behind DEBUG_MODE
        if os.getenv("DEBUG_MODE") == "true":
            print('path:', input_copy[0]['content'][0][inputs[0]['data_type']])   
            print('problem_id:', inputs[0]['problem_id'])       
            print('prompt_length:', prompt_length)
                
        
        
        
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        
        if inputs[0]['data_type'] == 'image':
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)

        if inputs[0]['data_type'] == 'video':
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if 'second_per_grid_ts' in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]
        
        
        
        
        try:
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            per_token_logps = per_token_logps[:, prompt_length - 1 :]
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
        
        # ============================================================== #
        # [FIX-3] D1: partial temporal frame mask (was: full zero-out)   #
        # [FIX-4] D1: model.eval() to disable dropout during causal     #
        #         measurement, then restore model.train()                #
        # ============================================================== #
        masked_per_token_logps = None
        if self.exp_config.enable_causal_reward and video_inputs:
            with torch.no_grad():
                masked_inputs = {}
                for k, v in prompt_inputs.items():
                    masked_inputs[k] = v.clone() if isinstance(v, torch.Tensor) else v

                if "pixel_values_videos" in masked_inputs:
                    # [FIX-3] Partial temporal frame mask via video_grid_thw
                    masked_inputs["pixel_values_videos"] = build_temporal_frame_mask(
                        pixel_values_videos=masked_inputs["pixel_values_videos"],
                        video_grid_thw=masked_inputs["video_grid_thw"],
                        mask_ratio=self.exp_config.causal_mask_ratio,
                        num_generations=self.num_generations,
                    )
                elif "pixel_values" in masked_inputs:
                    # Image: no temporal dim, fall back to full mask
                    masked_inputs["pixel_values"] = torch.zeros_like(
                        masked_inputs["pixel_values"]
                    )

                try:
                    # [FIX-4] Switch to eval mode to disable dropout for clean comparison
                    model.eval()
                    masked_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, **masked_inputs
                    )
                    masked_per_token_logps = masked_per_token_logps[:, prompt_length - 1 :]
                except Exception as e:
                    print(f"[D1] Error computing masked logps: {e}. Skipping causal reward this step.")
                    masked_per_token_logps = None
                finally:
                    # [OOM-FIX] Free masked activation memory before moving on
                    del masked_inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # [FIX-4] Restore training mode
                    model.train()

        with torch.inference_mode():
            try:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        # [FIX-6] Note: x_clamped is the numerical-stability clamp (±10) on the log ratio,
        # NOT the D2 KL truncation. D2 truncation happens below at per_token_kl.clamp(max=kl_d_max).
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1

        # [OOM-FIX] Free large intermediate tensors after KL is computed.
        # Do NOT delete per_token_logps — it is needed later for:
        #   1. D1 causal reward: log_P1 = (per_token_logps[i] * completion_mask[i]).sum()
        #   2. Policy loss: per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        del ref_per_token_logps, x_clamped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # [D2] Token-level clipped KL: cap each token's KL at D_max
        if self.exp_config.enable_token_clipped_kl:
            # [FIX-6] Save pre-clip KL for monitoring before D2 truncation modifies it
            raw_per_token_kl = per_token_kl.detach().clone()
            per_token_kl = per_token_kl.clamp(max=self.exp_config.kl_d_max)
        
        if self.temporal and video_inputs:
            shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                shuffled_completions = [[{"role": "assistant", "content": shuffled_completion}] for shuffled_completion in shuffled_completions]
                
            # Compute the rewards
            shuffled_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                shuffled_reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in shuffled_reward_kwargs:
                    for example in inputs:
                        shuffled_reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
                shuffled_output_reward_func = reward_func(prompts=shuffled_prompts, completions=shuffled_completions, **shuffled_reward_kwargs)
                shuffled_rewards_per_func[:, i] = torch.tensor(shuffled_output_reward_func, dtype=torch.float32, device=device)

        
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            
        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        

        
        
        if self.temporal and video_inputs:
            temporal_rewards_per_func = rewards_per_func.clone()
            
            acc_mean = temporal_rewards_per_func[:, 0].mean()
            shuffled_acc_mean = shuffled_rewards_per_func[:, 0].mean()

            if acc_mean >= 0.8 * shuffled_acc_mean:
                mask = temporal_rewards_per_func[:, 0] > 0.1
                temporal_rewards_per_func[mask, 0] = temporal_rewards_per_func[mask, 0] + 0.3
                temporal_rewards = torch.tensor([1.0]).to('cuda')
            else:
                temporal_rewards = torch.tensor([0.0]).to('cuda')
        else:
            temporal_rewards =  torch.tensor([0.5]).to('cuda')
        
        # Sum the rewards from all reward functions
        if self.temporal and video_inputs:
            rewards = temporal_rewards_per_func.sum(dim=1)
        else:
            rewards = rewards_per_func.sum(dim=1)
    
        
        if self.len_control:
            mem_rewards = [0] * self.num_generations
            mask = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
                    
            if len(selected_indices) > 1:     
                for idx in selected_indices:
                    if 320 <= lenth_list[idx] <= 512:
                        rewards[idx] += 0.2
        
        # [D1] Soft-Truncated Counterfactual Causal Reward
        # For samples where accuracy_reward >= 1.0, measure how much the model
        # relies on visual features by comparing log_P(Y|V,Q) vs log_P(Y|masked_V,Q)
        causal_rewards = torch.zeros_like(rewards)
        d1_num_evaluated = 0
        d1_num_skipped = 0
        if self.exp_config.enable_causal_reward and masked_per_token_logps is not None:
            accuracy_rewards_col = rewards_per_func[:, 0]  # accuracy reward column
            beta_s = self.exp_config.softplus_beta
            for i in range(len(rewards)):
                # Lazy evaluation: only compute for correct answers
                if accuracy_rewards_col[i].item() < 1.0:
                    d1_num_skipped += 1
                    continue
                d1_num_evaluated += 1
                # log_P1: sum of completion token log probs with visual input
                log_P1 = (per_token_logps[i].detach() * completion_mask[i]).sum()
                # log_P2: sum of completion token log probs WITHOUT visual input
                log_P2 = (masked_per_token_logps[i] * completion_mask[i]).sum()
                # Softplus: smooth approximation of max(0, diff)
                diff = log_P1 - log_P2
                causal_rewards[i] = torch.nn.functional.softplus(diff * beta_s) / beta_s

            # Clamp to prevent Advantage polarization from outliers
            causal_rewards = causal_rewards.clamp(max=self.exp_config.causal_reward_clip)
            rewards = rewards + causal_rewards * self.exp_config.causal_reward_scale

        # [D3] Task-Conditioned Piecewise Length Penalty
        # Penalize outputs that are too long or too short for their task type
        length_penalties = torch.zeros_like(rewards)
        if self.exp_config.enable_length_penalty and self.welford is not None:
            completion_lengths = completion_mask.sum(dim=1)  # [batch * num_gen]
            for i in range(len(rewards)):
                # [FIX-8] Safe indexing: clamp sample_idx to valid range of inputs
                sample_idx = min(i // self.num_generations, len(inputs) - 1)
                problem_type = inputs[sample_idx].get('problem_type', '')
                task = task_type_from_problem_type(problem_type) if problem_type else 'open_ended'
                length = completion_lengths[i].item()

                l_min, l_max = self.welford.get_dynamic_bounds(task, self.exp_config)

                if length > l_max:
                    length_penalties[i] = -self.exp_config.length_penalty_alpha * (length - l_max)
                elif length < l_min:
                    length_penalties[i] = -self.exp_config.length_penalty_beta * (l_min - length)

            rewards = rewards + length_penalties
            # Update Welford online statistics with this batch's lengths
            self.welford.update_batch(completion_lengths.tolist())

        # [FIX-7] Debug prints gated behind DEBUG_MODE (was: unconditional print)
        if os.getenv("DEBUG_MODE") == "true":
            print(rewards)
            print(completion_mask.sum(1))

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        
        num_devices = gathered_rewards.size(0) // self.num_generations 
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices
        
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices
        
        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)
        
        if self.temporal:
            temporal_rewards_list = self.accelerator.gather_for_metrics(temporal_rewards)
            self._metrics["temporal_rewards"].append(temporal_rewards_list.mean().item())
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # [D2] KL diagnostics: track truncation ratio when clipping is enabled
        if self.exp_config.enable_token_clipped_kl:
            # [FIX-6] Use the pre-clip snapshot saved earlier (raw_per_token_kl)
            # instead of recomputing from x_clamped to avoid confusion
            trunc_ratio = (raw_per_token_kl > self.exp_config.kl_d_max).float().mean()
            self._metrics["kl_truncation_ratio"].append(trunc_ratio.item())
            self._metrics["kl_raw_max"].append(raw_per_token_kl.max().item())

        # [D1] Causal reward diagnostics
        if self.exp_config.enable_causal_reward:
            self._metrics["causal_reward_mean"].append(causal_rewards.mean().item())
            self._metrics["causal_reward_max"].append(causal_rewards.max().item())
            self._metrics["causal_reward_std"].append(causal_rewards.std().item())  # [FIX-3] track variance
            total_d1 = d1_num_evaluated + d1_num_skipped
            self._metrics["causal_eval_ratio"].append(
                d1_num_evaluated / max(total_d1, 1)
            )

        # [D3] Length penalty diagnostics
        if self.exp_config.enable_length_penalty:
            self._metrics["length_penalty_mean"].append(length_penalties.mean().item())
            nonzero_ratio = (length_penalties != 0).float().mean().item()
            self._metrics["length_penalty_nonzero_ratio"].append(nonzero_ratio)
            self._metrics["welford_mean"].append(self.welford.mean)
            self._metrics["welford_std"].append(self.welford.std)
        

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))