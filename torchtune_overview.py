# %%
from torchtune.models.gemma import gemma_2b, lora_gemma_2b

# %%
# Build Gemma2 without any lora layers
base_model = gemma_2b()

# The debfault settings for lora_gemma_2b will match those for gemma_2b
# we just need to define which layers we want LoRA applied to
# within each self-attention, we can choose from ["q_proj", "k_proj", "v_proj" and "output_proj"]
# we can also set apply_lora_to_mlp=True or apply_lora_to_output=True to apply LoRA to other linear
# layers outside of the self-attention layers
lora_model = lora_gemma_2b(lora_attn_modules=["q_proj", "k_proj", "v_proj"])

# %% [markdown]
> Calling lora_llama_2_7b alone will not handle the definition of which parameters are trainable. See below for how to do this.
# %%
# Print the first layer's self-attention in the usual gemma2 model
print(base_model.layers[0].attn)
# %%
# Print the same for gemma with LoRA weights
print(lora_model.layers[0].attn)
# %%
# Notice that our LoRA model’s layer contains additional weights in the Q and V projections, as expected.
# Additionally, inspecting the type of lora_model and base_model, would show that they are both instances
# of the same TransformerDecoder. (Feel free to verify this for yourself.)
#
# Why does this matter? torchtune makes it easy to load checkpoints for LoRA directly from our gemma model
# without any wrappers or custom checkpoint conversion logic.

# Assuming that base_model already has the pretrained Llama2 weights,
# this will directly load them into your LoRA model without any conversion necessary.
lora_model.load_state_dict(base_model.state_dict(), strict=False)

# %% [markdown]
> Whenever loading weights with strict=False, you should verify that any missing or extra keys in the loaded state_dict are as expected. torchtune’s LoRA recipes do this by default via e.g. torchtune.modules.peft.validate_state_dict_for_lora().

# %%
from torchtune.modules.peft.peft_utils import get_adapter_params, set_trainable_params

# Fetch all params from the model that are associated with LoRA.
lora_params = get_adapter_params(lora_model)

# Set requires_grad=True on lora_params, and requires_grad=False on all others.
set_trainable_params(lora_model, lora_params)

# Print the total number of parameters
total_params = sum([p.numel() for p in lora_model.parameters()])
trainable_params = sum([p.numel() for p in lora_model.parameters() if p.requires_grad])
print(
  f"""
  {total_params} total params,
  {trainable_params}" trainable params,
  {(100.0 * trainable_params / total_params):.2f}% of all params are trainable.
  """
)
# %% [markdown]
Finally, we can put it all together and finetune a model using torchtune's LoRA recipe. Make sure that you have first downloaded the Llama2 weights and tokenizer by following these instructions. You can then run the following command to perform a LoRA finetune of Llama2-7B with two GPUs (each having VRAM of at least 16GB):

```bash
tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed --config gemma/2B_lora
```