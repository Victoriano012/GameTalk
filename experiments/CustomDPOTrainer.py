from trl import DPOTrainer, DPOConfig
from dataclasses import dataclass
import torch

@dataclass
class CustomDPOConfig(DPOConfig):
    eval_samples: int = 32
    num_generations: int = 8


class CustomDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        kwargs['judge'] = kwargs.pop('reward_funcs')
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch = None):
        model.train()

        prompts = inputs["prompt"]
        batch_size = len(prompts)

        if self.args.use_vllm:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_vllm(model, prompts)
        else:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate(model, prompts)

        contain_eos_token = torch.any(completion_ids == self.processing_class.eos_token_id, dim=-1)

        logprobs = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        with torch.no_grad():
            if self.ref_model is not None:
                ref_logprobs = self._forward(self.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
            else:  # peft case: we just need to disable the adapter
                with self.model.disable_adapter():
                    ref_logprobs = self._forward(self.model, prompt_ids, prompt_mask, completion_ids, completion_mask)

        # Decode the completions, and format them if the input is conversational
        device = logprobs.device
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Get the reward from the reward model or judge
        if self.judge is not None:
            ranks_of_first_completion = self.judge.judge(
                prompts, list(zip(completions[:batch_size], completions[batch_size:]))
            )

            mask = torch.tensor([rank == 0 for rank in ranks_of_first_completion], device=device)
        else:
            # prompts = 2 * prompts  # repeat the prompt: [prompt0, prompt1] -> [prompt0, prompt1, prompt0, prompt1]
            prompts = self.args.num_generations * prompts

            # Tokenize the prompts
            prompts_ids = self.reward_processing_class(
                prompts, padding=True, return_tensors="pt", padding_side="left"
            )["input_ids"].to(device)
            context_length = prompts_ids.shape[1]

            # Tokenize the completions
            completions_ids = self.reward_processing_class(
                completions, padding=True, return_tensors="pt", padding_side="right"
            )["input_ids"].to(device)

            prompt_completion_ids = torch.cat((prompts_ids, completions_ids), dim=1)
            with torch.inference_mode():
                _, scores, _ = get_reward(
                    self.reward_model, prompt_completion_ids, self.reward_processing_class.pad_token_id, context_length
                )

            # Split the scores in 2 (the prompts of the first half are the same as the second half)
            first_half, second_half = scores.split(batch_size)

            # Get the indices of the chosen and rejected examples
            mask = first_half >= second_half

        batch_range = torch.arange(batch_size, device=device)
        chosen_indices = batch_range + (~mask * batch_size)
        rejected_indices = batch_range + (mask * batch_size)

        # Build tensor so that the first half is the chosen examples and the second half the rejected examples
        cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)  # cr = chosen and rejected
        cr_logprobs = logprobs[cr_indices]
        cr_ref_logprobs = ref_logprobs[cr_indices]

        # mask out the padding tokens
        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask[cr_indices]

        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

        # Split the chosen and rejected examples
        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
        pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
        ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

        logits = pi_logratios - ref_logratios

        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.loss_type}")

        loss = losses.mean()

        # Log everything
        if self.reward_model is not None:
            scores_margin = scores[chosen_indices] - scores[rejected_indices]
            self.stats["objective/scores_margin"].append(
                self.accelerator.gather_for_metrics(scores_margin.mean()).mean().item()
            )
            self.stats["objective/scores"].append(self.accelerator.gather_for_metrics(scores.mean()).mean().item())
        self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())
        self.stats["logps/chosen"].append(self.accelerator.gather_for_metrics(chosen_logprobs_sum).mean().item())
        self.stats["logps/rejected"].append(self.accelerator.gather_for_metrics(rejected_logprobs_sum).mean().item())

        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        self.stats["objective/kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        non_score_reward = (-self.beta * kl).sum(1)
        mean_non_score_reward = non_score_reward.mean()
        self.stats["objective/non_score_reward"].append(
            self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
        )
        if self.reward_model is not None:
            rlhf_reward = scores + non_score_reward
            self.stats["objective/rlhf_reward"].append(self.accelerator.gather_for_metrics(rlhf_reward).mean().item())
        mean_entropy = -logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(self.accelerator.gather_for_metrics(mean_entropy).mean().item())
        chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        gathered_chosen_rewards = self.accelerator.gather_for_metrics(chosen_rewards)
        self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
        rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
        gathered_rejected_rewards = self.accelerator.gather_for_metrics(rejected_rewards)
        self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())
        margin = gathered_chosen_rewards - gathered_rejected_rewards
        self.stats["rewards/margins"].append(margin.mean().item())
        accuracy = margin > 0
        self.stats["rewards/accuracies"].append(accuracy.float().mean().item())
        self.stats["beta"].append(self.beta)

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps
