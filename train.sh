tune download google/gemma-2-2b \
  --output-dir /tmp/google/gemma-2-2b \
  --hf-token $HF_TOKEN

!tune run --nnodes 1 --nproc_per_node 2 ./recipes/lora_finetune_distributed.py --config ./configs/custom_lora.yaml \
    dataset=torchtune.datasets.text_completion_dataset dataset.source=virattt/financial-qa-10K dataset.split=train