import os

import requests
import torch

from core.src.models.gpt import GPTLanguageModel


class RuntimeModel:
    def __init__(self):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "weight/model.pth"
        )
        if os.path.exists(path) is False:
            RuntimeModel.download_model(path)

        self.model, self.checkpoint = self.load_model(path)

    def load_model(self, checkpoint_path: str):
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path, weights_only=True)
            else:
                checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
            

            model = GPTLanguageModel(checkpoint["vocab_size"])
            model.load_state_dict(checkpoint["model_state_dict"])

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            return model, checkpoint
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    @staticmethod
    def download_model(path: str):
        url = (
            "https://github.com/tnphucccc/GPTAtHome/releases/download/v1.0.1/model.pth"
        )

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        print(f"Downloading model from {url} ...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Write to a temp file first so an interrupted download never leaves
        # a truncated model.pth behind
        tmp_path = path + ".part"
        with open(tmp_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        os.replace(tmp_path, path)
        print(f"Model downloaded successfully to {path}")

    def generate_text(self, input_text: str = "", max_tokens: int = 2000):
        device = next(self.model.parameters()).device

        if input_text:
            stoi = self.checkpoint["stoi"]
            unknown = sorted(set(input_text) - stoi.keys())
            if unknown:
                raise ValueError(
                    "The model only knows the characters in its Shakespeare training text. "
                    f"Unsupported: {' '.join(repr(c) for c in unknown)}"
                )
            input_tokens = [stoi[c] for c in input_text]
            context = torch.tensor([input_tokens], dtype=torch.long, device=device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)

        # Return only the newly generated text, not the prompt it continues
        prompt_len = context.shape[1]
        output_tokens = self.model.generate(context, max_new_tokens=max_tokens)[
            0, prompt_len:
        ].tolist()

        itos = self.checkpoint["itos"]
        return "".join([itos[i] for i in output_tokens])

    def request(self, prompt: str, max_tokens: int = 2000):
        return self.generate_text(prompt, max_tokens)
