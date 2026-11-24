"""Pretrained ColBERTv2 token encoder with authentic 128-d linear projection."""
from typing import List, Tuple
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer


class ColBERTEncoder:
    """Encodes queries and passages into L2-normalized 128-d contextual token embeddings.

    Reuses the pretrained ColBERTv2 checkpoint (colbert-ir/colbertv2.0) without any
    fine-tuning or added parameters. Prepends ColBERT's [Q] ([unused0]) and [D] ([unused1])
    markers during encoding and strips special tokens before downstream pruning.
    """

    QUERY_MARKER_ID = 1  # [unused0] -> ColBERT [Q] marker
    DOC_MARKER_ID = 2    # [unused1] -> ColBERT [D] marker

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: str = None,
        q_maxlen: int = 32,
        d_maxlen: int = 180,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q_maxlen = q_maxlen
        self.d_maxlen = d_maxlen
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)

        # Load ColBERTv2's 768 -> 128 linear projection directly from model.safetensors
        self.proj = None
        self.dim = self.model.config.hidden_size
        try:
            st_path = hf_hub_download(model_name, "model.safetensors")
            state = load_file(st_path, device="cpu")
            proj_key = next((k for k in state.keys() if k.endswith("linear.weight")), None)
            if proj_key is not None:
                weight = state[proj_key]
                self.proj = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
                self.proj.weight.data.copy_(weight)
                self.proj = self.proj.eval().to(self.device)
                self.dim = weight.shape[0]
                print(f"[ColBERTEncoder] Loaded {proj_key} -> dim={self.dim} on {self.device}")
            else:
                print(f"[ColBERTEncoder] No linear projection key found; using dim={self.dim}")
        except Exception as e:
            print(f"[ColBERTEncoder] Projection load fallback ({e}); using dim={self.dim}")

        self._special_ids = set(self.tokenizer.all_special_ids) | {
            self.QUERY_MARKER_ID,
            self.DOC_MARKER_ID,
        }

    @torch.no_grad()
    def _encode_batch(
        self, texts: List[str], maxlen: int, marker_id: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        out_embs: List[torch.Tensor] = []
        out_ids: List[torch.Tensor] = []

        use_amp = self.device == "cuda"
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=maxlen - 1,  # reserve 1 slot for [Q]/[D] marker
                return_tensors="pt",
            )
            input_ids = enc["input_ids"]
            attn_mask = enc["attention_mask"]

            # Insert ColBERT [Q]/[D] marker immediately after [CLS]
            bsz = input_ids.size(0)
            marker_col = torch.full((bsz, 1), marker_id, dtype=input_ids.dtype)
            ones_col = torch.ones((bsz, 1), dtype=attn_mask.dtype)
            input_ids = torch.cat([input_ids[:, :1], marker_col, input_ids[:, 1:]], dim=1).to(self.device)
            attn_mask = torch.cat([attn_mask[:, :1], ones_col, attn_mask[:, 1:]], dim=1).to(self.device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                hidden = self.model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
                if self.proj is not None:
                    hidden = self.proj(hidden)
            hidden = F.normalize(hidden.float(), p=2, dim=-1).half()

            for b in range(bsz):
                ids_b = input_ids[b]
                valid = (attn_mask[b] == 1)
                # Strip [CLS], [Q]/[D], [SEP], [PAD]
                keep = valid.clone()
                for sid in self._special_ids:
                    keep &= (ids_b != sid)
                if keep.sum().item() == 0:
                    keep[0] = True
                out_embs.append(hidden[b][keep])
                out_ids.append(ids_b[keep])

        return out_embs, out_ids

    def encode_queries(self, queries: List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode queries into variable-length L2-normalized token embeddings."""
        return self._encode_batch(queries, self.q_maxlen, self.QUERY_MARKER_ID)

    def encode_docs(self, docs: List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode passages into variable-length L2-normalized token embeddings."""
        return self._encode_batch(docs, self.d_maxlen, self.DOC_MARKER_ID)
