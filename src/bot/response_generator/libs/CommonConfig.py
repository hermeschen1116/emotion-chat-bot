import os
import random
from dataclasses import dataclass
from typing import Any, Optional, Union, Dict, Literal

import huggingface_hub
import numpy as np
import torch.cuda
import wandb
from dotenv import load_dotenv
from transformers import TrainingArguments
from transformers.hf_argparser import HfArg


def value_candidate_check(input_value: Any,
                          possible_values: list,
                          use_default_value: bool,
                          default_value: Optional[Any]) -> Any:
    if input_value not in possible_values:
        error_message: str = \
            f"This parameter should be any of {', '.join(possible_values)}, your input is {input_value}"
        if use_default_value:
            print(error_message)
            return default_value
        raise ValueError(error_message)

    return input_value


def get_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@dataclass
class CommonScriptArguments:
    huggingface_api_token: Optional[str] = (
        HfArg(aliases=["--huggingface-api-token", "--huggingface-token", "--hf-token"], default=""))
    wandb_api_token: Optional[str] = (
        HfArg(aliases=["--wandb-api-token", "--wandb-token"], default=""))

    def __post_init__(self):
        load_dotenv(encoding="utf-8")
        huggingface_hub.login(token=os.environ.get("HF_TOKEN", self.huggingface_api_token), add_to_git_credential=True)
        wandb.login(key=os.environ.get("WANDB_API_KEY", self.wandb_api_token), relogin=True)

        torch.backends.cudnn.deterministic = True
        random.seed(hash("setting random seeds") % 2 ** 32 - 1)
        np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
        torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
        torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)


@dataclass
class CommonWanDBArguments:
    job_type: Optional[str] = HfArg(aliases=["--wandb-job-type", "--job-type"], default=None)
    config: Union[Dict, str, None] = HfArg(aliases="--wandb-type", default_factory=dict)
    project: Optional[str] = HfArg(aliases="--wandb-project", default=None)
    group: Optional[str] = HfArg(aliases=["--wandb-group", "--group"], default=None)
    notes: Optional[str] = HfArg(aliases=["--wandb-notes", "--notes"], default=None)
    mode: Optional[Union[Literal["online", "offline", "disabled"], None]] = HfArg(aliases="--wandb-mode", default=None)
    allow_val_change: Optional[bool] = HfArg(aliases="--allow-val-change", default=False)
    resume: Optional[str] = HfArg(aliases="--wandb-resume", default=None)

    def __post_init__(self):
        module: list = ["Sentiment Analysis",
                        "Candidate Generator",
                        "Emotion Predictor",
                        "Emotion Model",
                        "Similarity Analysis",
                        "Response Generator"]

        self.group = value_candidate_check(self.group,
                                           use_default_value=True,
                                           default_value="",
                                           possible_values=module)


@dataclass
class TrainerArguments:
    output_dir: str = HfArg(default="./checkpoints")
    overwrite_output_dir: bool = HfArg(default=False)
    do_train: bool = HfArg(default=False)
    do_eval: bool = HfArg(default=False)
    do_predict: bool = HfArg(default=False)
    evaluation_strategy: Union = HfArg(default='no')
    prediction_loss_only: bool = HfArg(default=False)
    per_device_train_batch_size: int = HfArg(default=8)
    per_device_eval_batch_size: int = HfArg(default=8)
    per_gpu_train_batch_size: Optional = HfArg(default=None)
    per_gpu_eval_batch_size: Optional = HfArg(default=None)
    gradient_accumulation_steps: int = HfArg(default=1)
    eval_accumulation_steps: Optional = HfArg(default=None)
    eval_delay: Optional = HfArg(default=0)
    learning_rate: float = HfArg(default=5e-05)
    weight_decay: float = HfArg(default=0.0)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    lr_scheduler_type: Union = 'linear'
    lr_scheduler_kwargs: Optional = dict
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    log_level: Optional = 'passive'
    log_level_replica: Optional = 'warning'
    log_on_each_node: bool = True
    logging_dir: Optional = HfArg(default=None)
    logging_strategy: Union = 'steps'
    logging_first_step: bool = HfArg(default=False)
    logging_steps: float = 500
    logging_nan_inf_filter: bool = True
    save_strategy: Union = 'steps'
    save_steps: float = 500
    save_total_limit: Optional = HfArg(default=None)
    save_safetensors: Optional = True
    save_on_each_node: bool = HfArg(default=False)
    save_only_model: bool = HfArg(default=False)
    no_cuda: bool = HfArg(default=False)
    use_cpu: bool = HfArg(default=False)
    use_mps_device: bool = HfArg(default=False)
    seed: int = 42
    data_seed: Optional = HfArg(default=None)
    jit_mode_eval: bool = HfArg(default=False)
    use_ipex: bool = HfArg(default=False)
    bf16: bool = HfArg(default=False)
    fp16: bool = HfArg(default=False)
    fp16_opt_level: str = 'O1'
    half_precision_backend: str = 'auto'
    bf16_full_eval: bool = HfArg(default=False)
    fp16_full_eval: bool = HfArg(default=False)
    tf32: Optional = HfArg(default=None)
    local_rank: int = -1
    ddp_backend: Optional = HfArg(default=None)
    tpu_num_cores: Optional = HfArg(default=None)
    tpu_metrics_debug: bool = HfArg(default=False)
    debug: Union = ''
    dataloader_drop_last: bool = HfArg(default=False)
    eval_steps: Optional = HfArg(default=None)
    dataloader_num_workers: int = 0
    dataloader_prefetch_factor: Optional = HfArg(default=None)
    past_index: int = -1
    run_name: Optional = HfArg(default=None)
    disable_tqdm: Optional = HfArg(default=None)
    remove_unused_columns: Optional = True
    label_names: Optional = HfArg(default=None)
    load_best_model_at_end: Optional = HfArg(default=False)
    metric_for_best_model: Optional = HfArg(default=None)
    greater_is_better: Optional = HfArg(default=None)
    ignore_data_skip: bool = HfArg(default=False)
    fsdp: Union = ''
    fsdp_min_num_params: int = 0
    fsdp_config: Union = HfArg(default=None)
    fsdp_transformer_layer_cls_to_wrap: Optional = HfArg(default=None)
    accelerator_config: Optional = HfArg(default=None)
    deepspeed: Optional = HfArg(default=None)
    label_smoothing_factor: float = 0.0
    optim: Union = 'adamw_torch'
    optim_args: Optional = HfArg(default=None)
    adafactor: bool = HfArg(default=False)
    group_by_length: bool = HfArg(default=False)
    length_column_name: Optional = 'length'
    report_to: Optional = HfArg(default=None)
    ddp_find_unused_parameters: Optional = HfArg(default=None)
    ddp_bucket_cap_mb: Optional = HfArg(default=None)
    ddp_broadcast_buffers: Optional = HfArg(default=None)
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = HfArg(default=False)
    skip_memory_metrics: bool = True
    use_legacy_prediction_loop: bool = HfArg(default=False)
    push_to_hub: bool = HfArg(default=False)
    resume_from_checkpoint: Optional = HfArg(default=None)
    hub_model_id: Optional = HfArg(default=None)
    hub_strategy: Union = 'every_save'
    hub_token: Optional = HfArg(default=None)
    hub_private_repo: bool = HfArg(default=False)
    hub_always_push: bool = HfArg(default=False)
    gradient_checkpointing: bool = HfArg(default=False)
    gradient_checkpointing_kwargs: Optional = HfArg(default=None)
    include_inputs_for_metrics: bool = HfArg(default=False)
    fp16_backend: str = 'auto'
    push_to_hub_model_id: Optional = HfArg(default=None)
    push_to_hub_organization: Optional = HfArg(default=None)
    push_to_hub_token: Optional = HfArg(default=None)
    mp_parameters: str = ''
    auto_find_batch_size: bool = HfArg(default=False)
    full_determinism: bool = HfArg(default=False)
    torchdynamo: Optional = HfArg(default=None)
    ray_scope: Optional = 'last'
    ddp_timeout: Optional = 1800
    torch_compile: bool = HfArg(default=False)
    torch_compile_backend: Optional = HfArg(default=None)
    torch_compile_mode: Optional = HfArg(default=None)
    dispatch_batches: Optional = HfArg(default=None)
    split_batches: Optional = HfArg(default=None)
    include_tokens_per_second: Optional = HfArg(default=False)
    include_num_input_tokens_seen: Optional = HfArg(default=False)
    neftune_noise_alpha: Optional = HfArg(default=None)
    # optim_target_modules: Union = HfArg(default=None)
