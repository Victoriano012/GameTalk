import torch
import torch.nn.functional as F
from transformers.trainer import logger, OptimizerNames, DistributedType, is_sagemaker_mp_enabled, is_torch_hpu_available, is_torch_mlu_available, is_torch_mps_available, is_torch_musa_available, is_torch_npu_available, is_torch_xpu_available
from trl.trainer.utils import selective_log_softmax
from trl import GRPOTrainer, GRPOConfig

from dataclasses import dataclass

from utils import split_dict

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

@dataclass
class CustomGRPOConfig(GRPOConfig):
    eval_samples: int = 32
    entropy_beta: float = 0.01

    full_gradient_accumulation: bool = False


class CustomGRPOTrainer(GRPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        if self.args.entropy_beta > 0:
            per_token_logps, entropy = self._get_per_token_logps_and_entropy(model, input_ids, attention_mask, logits_to_keep)
        else:
            per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        if self.args.entropy_beta > 0:
            per_token_loss = -(per_token_loss - self.beta * per_token_kl + self.args.entropy_beta * entropy)
        else:
            per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        if self.args.entropy_beta > 0:
            mean_entropy = ((entropy * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["entropy"].append(self.accelerator.gather_for_metrics(mean_entropy).mean().item())

        return loss

    def _get_per_token_logps_and_entropy(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        logps = selective_log_softmax(logits, input_ids)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        return logps, entropy  #  compute logprobs for the input tokens
    

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        assert not self.use_apex, "Apex ?"

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        kwargs = {}
        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()
        # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
        # https://github.com/huggingface/transformers/pull/35808
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            kwargs["scale_wrt_gas"] = False
        

        if not self.args.full_gradient_accumulation:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            if self.args.n_gpu > 1: loss = loss.mean()

            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            self.accelerator.backward(loss, **kwargs)
        else:
            losses = []

            for curr_input in split_dict(inputs):
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, curr_input, num_items_in_batch=num_items_in_batch)
                if self.args.n_gpu > 1: loss = loss.mean()

                # Finally we need to normalize the loss for reporting
                if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                    loss = loss / self.args.gradient_accumulation_steps

                self.accelerator.backward(loss, **kwargs)
                losses.append(loss.detach())
            
            loss = torch.stack(losses).mean()
        


        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        return loss.detach()