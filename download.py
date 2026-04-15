EXPORT HF_ENDPOINT="https://hf-mirror.com"
from datasets import load_dataset

italian_dataset = load_dataset("llamaindex/vdr-multilingual-train", "it", split="train")

english_dataset = load_dataset("llamaindex/vdr-multilingual-train", "en", split="train")

french_dataset = load_dataset("llamaindex/vdr-multilingual-train", "fr", split="train")

german_dataset = load_dataset("llamaindex/vdr-multilingual-train", "de", split="train")

spanish_dataset = load_dataset("llamaindex/vdr-multilingual-train", "es", split="train")
