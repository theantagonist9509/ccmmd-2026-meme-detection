# %% [markdown]
# # Qwen3 → BERT Dataset Generation
# This script:
# 1. Runs Qwen3 (text-only) on each transcription with a configurable prompt.
# 2. Feeds Qwen3's generated text response to BERT.
# 3. Stores the resulting BERT [CLS] embeddings via EmbeddedMisoDataset.

# %%
import os
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("ROOT_PATH"))

from pathlib import Path
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
)

sys.path.append(str(Path(__file__).parent.parent))
from miso_utils.datasets import create_train_dataset, create_val_dataset, EmbeddedMisoDataset

# %%
# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------
QWEN_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"       # any text-only Qwen3 checkpoint
BERT_MODEL_NAME = "bert-base-uncased"
DATASET_PATH    = os.getenv("VAL_PATH")
SAVE_PATH       = os.getenv("ROOT_PATH") + "/datasets/qwen3_bert/val.pt"
BATCH_SIZE      = 16                       # keep small; Qwen3 is large
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prompt template – {text} is replaced with the meme transcription at runtime.
DEFAULT_PROMPT = (
    "You are analysing a social-media meme. "
    "The following text was extracted from the meme image:\n\n"
    "{text}\n\n"
    "Briefly describe any misogynistic sentiment present in the text above, "
    "or state that none is present."
)

# %%
# ---------------------------------------------------------------------------
# Qwen3 → BERT wrapper module
# ---------------------------------------------------------------------------
class Qwen3BertEmbedder(nn.Module):
    """
    Takes a dict batch (as produced by MisoDataset with mode='text') and
    returns a BERT embedding of Qwen3's textual response.

    Pipeline
    --------
    transcription  ─► [prompt template]  ─► Qwen3 (generate)
                  ─► Qwen3 response text ─► BERT ─► [CLS] pooler output

    Parameters
    ----------
    qwen_model_name : str
        HuggingFace identifier for the Qwen3 text-generation model.
    bert_model_name : str
        HuggingFace identifier for the BERT encoder.
    prompt_template : str
        A string with a single ``{text}`` placeholder that will be filled
        with each meme transcription before being passed to Qwen3.
    max_new_tokens : int
        Maximum number of tokens Qwen3 is allowed to generate per sample.
    """

    def __init__(
        self,
        qwen_model_name: str = QWEN_MODEL_NAME,
        bert_model_name: str = BERT_MODEL_NAME,
        prompt_template: str = DEFAULT_PROMPT,
        max_new_tokens: int = 128,
    ):
        super().__init__()
        self.prompt_template = prompt_template
        self.max_new_tokens  = max_new_tokens

        # --- Qwen3 (causal LM, text-only) ---
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_name, padding_side="left"
        )
        # Qwen3 tokenizers may not define a pad token by default
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.bfloat16,
        )

        # --- BERT encoder ---
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)

    # ------------------------------------------------------------------
    def _build_prompts(self, transcriptions: list[str]) -> list[str]:
        return [self.prompt_template.format(text=t) for t in transcriptions]

    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : dict
            A batch dict from MisoDataset / DataLoader.
            Must contain the key ``"transcription"`` (list[str]).

        Returns
        -------
        torch.Tensor  shape (B, hidden_size)
            BERT [CLS] pooler embeddings of Qwen3's responses.
        """
        transcriptions: list[str] = batch["transcription"]
        device = next(self.bert.parameters()).device

        # ---- 1. Build prompts & run Qwen3 --------------------------------
        prompts = self._build_prompts(transcriptions)

        qwen_inputs = self.qwen_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Qwen3 must be on the same device; move if needed
        if next(self.qwen.parameters()).device != device:
            self.qwen = self.qwen.to(device)

        with torch.no_grad():
            generated_ids = self.qwen.generate(
                **qwen_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,          # greedy; set True for sampling
                pad_token_id=self.qwen_tokenizer.pad_token_id,
            )

        # Trim the prompt tokens – keep only the newly generated part
        prompt_len     = qwen_inputs.input_ids.shape[1]
        new_ids        = generated_ids[:, prompt_len:]
        qwen_responses: list[str] = self.qwen_tokenizer.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # ---- 2. Encode Qwen3 responses with BERT -------------------------
        bert_inputs = self.bert_tokenizer(
            qwen_responses,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        bert_outputs = self.bert(**bert_inputs)

        # Return the [CLS] pooler output  (B, hidden_size)
        return bert_outputs.pooler_output


# %%
# ---------------------------------------------------------------------------
# Instantiate the model and build the dataset
# ---------------------------------------------------------------------------
model = Qwen3BertEmbedder(
    qwen_model_name = QWEN_MODEL_NAME,
    bert_model_name = BERT_MODEL_NAME,
    prompt_template = DEFAULT_PROMPT,
    max_new_tokens  = 128,
)

# %%
miso_val = create_val_dataset(path=DATASET_PATH, mode="text")

# %%
embedded_dataset = EmbeddedMisoDataset(
    miso_dataset = miso_val,
    model        = model,
    batch_size   = BATCH_SIZE,
    device       = DEVICE,
)

# %%
save_dir = Path(SAVE_PATH).parent
save_dir.mkdir(parents=True, exist_ok=True)
torch.save(embedded_dataset, SAVE_PATH)
print(f"Saved {len(embedded_dataset)} embeddings to {SAVE_PATH}")

# %%
