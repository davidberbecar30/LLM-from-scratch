# LLM From Scratch

A GPT-style large language model implemented from scratch in PyTorch, built by following Sebastian Raschka's book *[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)*.

The project walks through the full lifecycle of an LLM: building the transformer architecture piece by piece, preparing text data for training, pretraining the model on raw text, loading OpenAI's pretrained GPT-2 weights into the custom architecture, and finetuning the model for two downstream tasks.

## What's in here

- **Architecture**: custom implementation of multi-head self-attention, transformer blocks, layer norm, and the full GPT model, built up from first principles (`GPT_model.py`, `LLM_architecture.ipynb`, `attention-mechanisms.ipynb`)
- **Data pipeline**: byte-pair-encoding tokenization and PyTorch dataset/dataloader setup for feeding text into the model (`datasets.ipynb`, `text-data.ipynb`)
- **Pretraining**: training the model from scratch on raw text with a next-token-prediction loss (`pretraining.ipynb`)
- **Pretrained weights**: downloading and loading OpenAI's official GPT-2 weights into the custom architecture, so it can generate coherent text without training from zero (`gpt_download.py`)
- **Finetuning**: adapting the pretrained model for two tasks — spam/ham classification, and instruction-following via supervised finetuning on an instruction dataset (`finetuning-classification.ipynb`, `finetuning-instruction.ipynb`)

## Notes

Model checkpoints and downloaded GPT-2 weights are excluded via `.gitignore` (regenerable, too large for git). Datasets used for training/finetuning are kept.
