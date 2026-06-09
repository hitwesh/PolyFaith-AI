from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

from backend.config.settings import MODEL_NAME


class ModelLoader:

    model = None
    tokenizer = None

    @classmethod
    def load(cls):

        if cls.model is None:

            print("Loading tokenizer...")

            cls.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME
            )

            print("Loading model...")

            cls.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            print("Model loaded successfully.")

        return cls.model, cls.tokenizer