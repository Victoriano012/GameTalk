from trl import OnlineDPOTrainer, OnlineDPOConfig

from dataclasses import dataclass
import itertools
import torch

from transformers.training_args import OptimizerNames
from trl.trainer.utils import truncate_right, empty_cache
from trl.models.utils import unwrap_model_for_generation
from trl.data_utils import is_conversational, maybe_apply_chat_template



def plackett_luce_logprob(v):
    if not isinstance(v, torch.Tensor):
        v = torch.as_tensor(v, dtype=torch.float32, device='cuda')
    logprob = torch.tensor(0.0, dtype=v.dtype, device=v.device)
    for k in range(len(v)):
        denominator = torch.sum(v[k:])
        logprob += torch.log(v[k]) - torch.log(denominator)
    return logprob


def all_subsets(tensor):
    for r in range(1, len(tensor) + 1):
        for comb in itertools.combinations(range(len(tensor)), r):
            yield tensor[list(comb)]

def tied_plackett_luce_logprob(sets):
    logprob = torch.tensor(0.0, dtype=sets[0].dtype, device=sets[0].device)

    for tie_group in sets:
        logprob += torch.log(tie_group).mean()

    for set in itertools.accumulate(sets[::-1], lambda acc, x: torch.cat([acc,x])):
        denominator = torch.tensor(0.0)
        for subset in all_subsets(set):
            denominator += subset.log().mean().exp() # geometric average
        logprob -= torch.log(denominator)

    return logprob



@dataclass
class CustomDPOConfig(OnlineDPOConfig):
    eval_samples: int = 32
    num_generations: int = 8

    dpo_variant: str = "all_pairs"

    reward_weights: list = None


class CustomDPOTrainer(OnlineDPOTrainer):
    def __init__(self, *args, **kwargs):
        kwargs['judge'] = kwargs.pop('reward_funcs')
        super().__init__(*args, **kwargs)

        self.is_encoder_decoder = kwargs['model'].config.is_encoder_decoder # idk why this is needed
        self.stats = {
            "objective/reward": [],
            "objective/reward_std": [],
            "val/contain_eos_token": [],
            "objective/kl": [],
            "objective/entropy": [],
            "objective/non_score_reward": [],
            "objective/rlhf_reward": [],
            "beta": [],
        }
        if isinstance(self.judge, list):
            if self.args.reward_weights is None: self.args.reward_weights = [1.] * len(self.judge)
            assert len(self.judge) == len(self.args.reward_weights), "Number of reward_funcs, and reward_weights should be the same"

            for reward_func in self.judge:
                self.stats[f"rewards/{reward_func.__name__}"] = []

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

        inputs["prompts"] = inputs.pop("prompt")
        inputs = { k : self.args.num_generations * v for k, v in inputs.items() }
        inputs["completions"] = completions
        if not isinstance(self.judge, list):
            rewards = torch.Tensor(self.judge(**inputs)).to(device)
        else:
            reward_per_func = [torch.Tensor(func(**inputs)) for func in self.judge]
            rewards = sum(
                    self.args.reward_weights[i] * reward_per_func[i]
                    for i in range(len(self.judge))
                )

        ref_logprobs = (ref_logprobs*completion_mask.bool()).sum(1)
        logprobs = (logprobs*completion_mask.bool()).sum(1)
        dpo_weights = torch.exp(self.beta * (logprobs - ref_logprobs))
        
        dpo_weights = dpo_weights.view(self.args.num_generations, batch_size).t()
        rewards_view = rewards.view(self.args.num_generations, batch_size).t()

        print(f"Computing group losses, bs={batch_size}")
        losses = []
        for group_weights, group_rewards in zip(dpo_weights, rewards_view):
            unique_rewards, _ = torch.sort(torch.unique(group_rewards), descending=True)

            while len(unique_rewards) == 1:
                print("Only one unique reward, randomizing this group")
                group_rewards = torch.randint(low=0, high=2, size=group_rewards.shape, device=group_rewards.device)
                unique_rewards, _ = torch.sort(torch.unique(group_rewards), descending=True)
            
            dpo_weight_groupped_by_reward = [group_weights[group_rewards == r] for r in unique_rewards]
            
            if self.args.dpo_variant == 'all_pairs':
                pair_log_probs = []
                for i in range(len(dpo_weight_groupped_by_reward)):
                    for j in range(i + 1, len(dpo_weight_groupped_by_reward)):
                        for x in dpo_weight_groupped_by_reward[i]:
                            for y in dpo_weight_groupped_by_reward[j]:
                                pair_log_probs.append(plackett_luce_logprob([x, y]))
                curr_loss = - torch.stack(pair_log_probs).mean()

            elif self.args.dpo_variant == 'all_perms':
                perm_log_probs = []
                for combination in torch.cartesian_prod(*dpo_weight_groupped_by_reward):
                    perm_log_probs.append(plackett_luce_logprob(combination))
                curr_loss = - torch.stack(perm_log_probs).mean()
            
            elif self.args.dpo_variant == 'with_ties':
                curr_loss = - tied_plackett_luce_logprob(dpo_weight_groupped_by_reward)

            losses.append(curr_loss)
        
        loss = torch.stack(losses).mean()
        print("loss computed")

        # Log everything
        reward_std = rewards.view(batch_size, self.args.num_generations).std(-1)
        self.stats["objective/reward_std"].append(
            self.accelerator.gather_for_metrics(reward_std.mean()).mean().item()
        )
        self.stats["objective/reward"].append(self.accelerator.gather_for_metrics(rewards.mean()).mean().item())
        self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())

        if isinstance(self.judge, list):
            for i, reward_func in enumerate(self.judge):
                self.stats[f"rewards/{reward_func.__name__}"].append(reward_per_func[i].mean().item())

        kl = logprobs - ref_logprobs
        mean_kl = kl.mean()
        self.stats["objective/kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        non_score_reward = (-self.beta * kl)
        mean_non_score_reward = non_score_reward.mean()
        self.stats["objective/non_score_reward"].append(
            self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
        )
        rlhf_reward = rewards.to(device) + non_score_reward.to(device)
        self.stats["objective/rlhf_reward"].append(self.accelerator.gather_for_metrics(rlhf_reward).mean().item())
        mean_entropy = -logprobs.mean()
        self.stats["objective/entropy"].append(self.accelerator.gather_for_metrics(mean_entropy).mean().item())
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
            assert False, "Apex is not supported in CustomDPOTrainer"
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps




    def _generate_vllm(self, model, prompts):
        eos_token_id = self.processing_class.eos_token_id
        pad_token_id = self.processing_class.pad_token_id

        # Load the latest weights
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(model.state_dict().items())

        if is_conversational({"prompt": prompts[0]}):
            outputs = self.llm.chat(prompts, self.generation_config, use_tqdm=False)
        else:
            outputs = self.llm.generate(prompts, self.generation_config, use_tqdm=False)

        ################ ONLY LINES CHANGED ################
        completion_ids = [list(output.outputs[i].token_ids) for i in range(self.args.num_generations) for output in outputs]
        prompt_ids = [list(output.prompt_token_ids) for _ in range(self.args.num_generations) for output in outputs]
        ####################################################

        # Create mask and pad the prompt and completion
        max_prompt_length = max(len(ids) for ids in prompt_ids)
        prompt_mask = [[0] * (max_prompt_length - len(ids)) + [1] * len(ids) for ids in prompt_ids]
        prompt_ids = [[pad_token_id] * (max_prompt_length - len(ids)) + ids for ids in prompt_ids]
        max_tokens = self.generation_config.max_tokens
        completion_mask = [[1] * len(ids) + [0] * (max_tokens - len(ids)) for ids in completion_ids]
        completion_ids = [
            ids + [eos_token_id] if ids[-1] != eos_token_id and len(ids) < max_tokens else ids
            for ids in completion_ids
        ]
        completion_ids = [ids + [pad_token_id] * (max_tokens - len(ids)) for ids in completion_ids]

        # Convert to tensors
        prompt_ids = torch.tensor(prompt_ids, device=self.accelerator.device)
        prompt_mask = torch.tensor(prompt_mask, device=self.accelerator.device)
        completion_ids = torch.tensor(completion_ids, device=self.accelerator.device)
        completion_mask = torch.tensor(completion_mask, device=self.accelerator.device)

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _generate(self, model, prompts):
        eos_token_id = self.processing_class.eos_token_id
        pad_token_id = self.processing_class.pad_token_id

        inputs = [{"prompt": prompt} for prompt in prompts]
        inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
        inputs = [self.tokenize_row(x, self.is_encoder_decoder, self.processing_class) for x in inputs]
        inputs = self.data_collator(inputs)

        # Sample n completions per prompt of size `max_new_tokens` from the model
        inputs = self._prepare_inputs(inputs)

        ################ ONLY LINES CHANGED ################
        prompt_ids = inputs["prompt_input_ids"].repeat(self.args.num_generations, 1)
        prompt_mask = inputs["prompt_attention_mask"].repeat(self.args.num_generations, 1)
        ####################################################

        with unwrap_model_for_generation(
            model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            output = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
            )

        completion_ids = output[:, prompt_ids.size(1) :]
        completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)

        return prompt_ids, prompt_mask, completion_ids, completion_mask
