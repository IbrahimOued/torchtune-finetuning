tune download google/gemma-2-2b \
  --output-dir /tmp/google/gemma-2-2b \
  --hf-token $HF_TOKEN

tune run lora_finetune_distributed --config gemma/2B_lora epochs=1