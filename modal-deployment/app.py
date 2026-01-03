import modal

# ---- Modal App ----
app = modal.App("query-expansion-topic-tagging")

# ---- Image with CUDA ----
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git")   # ✅ Install git first
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "sentencepiece",
        "peft",
        "bitsandbytes",
        "git+https://github.com/unslothai/unsloth.git",  # ✅ GitHub version
    )
)

# ---- GPU ----
GPU = modal.gpu.T4()

# ---- Volume (model cache) ----
volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

MODEL_DIR = "/models"

@app.cls(
    image=image,
    gpu=GPU,
    timeout=60 * 10,
    volumes={MODEL_DIR: volume},
)
class QueryExpansionService:
    def __enter__(self):
        import torch
        from unsloth import FastLanguageModel
        from transformers import AutoTokenizer

        model_name = "Qwen/Qwen2.5-3B-Instruct"  # change if needed
        max_seq_length = 2048

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto",
            cache_dir=MODEL_DIR,
        )

        FastLanguageModel.for_inference(self.model)

    @modal.method()
    def infer(self, prompt: str, max_new_tokens: int = 256):
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return decoded
