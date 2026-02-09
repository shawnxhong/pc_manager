# -*- coding: utf-8 -*-
 
from __future__ import annotations
 
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import hashlib
import time
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from catalog import ToolItem, build_embedding_text
 
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
 
 
def _catalog_fingerprint(items: List[ToolItem]) -> str:
    payload = [it.to_dict() for it in items]
    b = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return _sha256_bytes(b)
 
 
def _default_models_dir() -> Path:
    return (Path(__file__).resolve().parent.parent / "models").resolve()
 
 
def _repo_to_dirname(repo_id: str) -> str:
    # "BAAI/bge-m3" -> "BAAI__bge-m3" (avoid path separators)
    return repo_id.replace("/", "__").replace("\\", "__")
 
 
class Embedder:
    """
    Supports two backends:
    - Transformers embeddings (recommended, better quality): loads locally from ..\\models first, downloads only if absent
    - Fallback: TF-IDF (no large model download needed, but lower quality)
    """
 
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        models_dir: Optional[str | Path] = None,
        revision: str = "main",
    ):
        self.model_name = model_name
        self.device = device
        self.revision = revision
        self.models_dir = Path(models_dir) if models_dir is not None else _default_models_dir()
 
        self._backend = None
        self._local_model_dir: Optional[Path] = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_backend_type = None  # "hf" or "tfidf"
 
    def _ensure_local_hf_model_dir(self) -> Path:
        """
        Return the fixed local model directory: ..\\models\\<repo_id_sanitized>
        - If the directory looks complete, return immediately (no network access)
        - Otherwise attempt to download into that directory
        """
        local_dir = (self.models_dir / _repo_to_dirname(self.model_name)).resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
 
        # Minimal "looks complete" check (not strict, to avoid missing sharded weights etc.)
        likely_files = [
            local_dir / "config.json",
            local_dir / "tokenizer.json",
            local_dir / "tokenizer_config.json",
        ]
        looks_ready = any(p.exists() for p in likely_files)
 
        if looks_ready:
            return local_dir
 
        # Local model incomplete -> attempt download
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "Missing huggingface_hub, cannot download embedding model. Please install: pip install huggingface_hub"
            ) from e
 
        # Only download when necessary; download target is always local_dir
        snapshot_download(
            repo_id=self.model_name,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            revision=self.revision,
        )
        return local_dir
 
    def _load_backend(self):
        if self._backend is not None:
            return
 
        # 1) Transformers embedder (preferred)
        try: 
            self.models_dir.mkdir(parents=True, exist_ok=True)
            local_dir = (self.models_dir / _repo_to_dirname(self.model_name)).resolve()
 
            # Step A: force local-only loading (no network)
            try:
                tok = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
                mdl = AutoModel.from_pretrained(str(local_dir), local_files_only=True)
                self._local_model_dir = local_dir
            except Exception:
                # Step B: local unavailable -> download to fixed directory, then load locally
                local_dir = self._ensure_local_hf_model_dir()
                tok = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
                mdl = AutoModel.from_pretrained(str(local_dir), local_files_only=True)
                self._local_model_dir = local_dir
 
            mdl.eval()
            if self.device:
                mdl.to(self.device)
            self._hf_tokenizer = tok
            self._hf_model = mdl
            self._hf_backend_type = "hf"
 
            def mean_pool(last_hidden, attn_mask):
                mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                return summed / counts
 
            def embed_texts(texts: List[str]) -> np.ndarray:
                with torch.no_grad():
                    batch = self._hf_tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=256,
                        return_tensors="pt",
                    )
                    if self.device:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                    out = self._hf_model(**batch)
                    vec = mean_pool(out.last_hidden_state, batch["attention_mask"])
                    vec = torch.nn.functional.normalize(vec, p=2, dim=1)
                    return vec.detach().cpu().numpy().astype(np.float32)
 
            self._backend = ("hf", embed_texts)
            return
 
        except Exception:
            pass
 
        # 2) TF-IDF fallback
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
 
            vec = TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=200000,
            )
            self._hf_backend_type = "tfidf"
            self._backend = ("tfidf", vec)
            return
        except Exception as e:
            raise RuntimeError(
                "No embedding backend available. Please install either:\n"
                "  - transformers + torch (+ huggingface_hub)\n"
                "or\n"
                "  - scikit-learn\n"
                f"Original error: {e}"
            )

    def close(self) -> None:
        """
        Explicitly release memory used by embedding model/vectorizer.
        """
        try:
            import torch
        except Exception:
            torch = None
 
        # Disconnect closure references
        self._backend = None
 
        # Release HF model
        self._hf_tokenizer = None
        hf_model = self._hf_model
        self._hf_model = None
 
        # Attempt GPU memory reclaim
        if torch is not None:
            try:
                if hf_model is not None:
                    del hf_model
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
 
        gc.collect()

    def embed_corpus(self, texts: List[str]) -> Tuple[str, Any, np.ndarray]:
        self._load_backend()
        backend_type, fn_or_vec = self._backend
 
        if backend_type == "hf":
            emb = fn_or_vec(texts)  # type: ignore
            return "hf", None, emb
 
        # tfidf
        vec = fn_or_vec  # type: ignore
        X = vec.fit_transform(texts)
        X = X.astype(np.float32)
        norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        X = X.multiply(1.0 / norms[:, None])
        emb = X.toarray().astype(np.float32)
        return "tfidf", vec, emb
 
    def embed_query(self, text: str, backend_type: str, backend_state: Any) -> np.ndarray:
        self._load_backend()
        btype, fn_or_vec = self._backend
 
        if backend_type != btype:
            raise RuntimeError(f"Embedder backend mismatch: index={backend_type}, runtime={btype}")
 
        if backend_type == "hf":
            emb = fn_or_vec([text])  # type: ignore
            return emb[0]
 
        vec = backend_state
        X = vec.transform([text]).astype(np.float32)
        norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        X = X.multiply(1.0 / norms[:, None])
        return X.toarray().astype(np.float32)[0] 
 
@dataclass
class Candidate:
    tool_id: str
    score: float
 
 
class ToolRAGIndex:
    def __init__(
        self,
        items: List[ToolItem],
        index_dir: Path,
        embedder: Embedder,
    ):
        self.items = items
        self.index_dir = Path(index_dir)
        self.embedder = embedder
 
        self._id_to_item: Dict[str, ToolItem] = {it.id: it for it in items}
 
        self._backend_type: Optional[str] = None
        self._backend_state: Any = None
        self._embeddings: Optional[np.ndarray] = None
        self._tool_ids: Optional[List[str]] = None
 
    @property
    def ready(self) -> bool:
        return self._embeddings is not None and self._tool_ids is not None
 
    def _paths(self) -> Dict[str, Path]:
        d = self.index_dir
        return {
            "meta": d / "meta.json",
            "items": d / "items.json",
            "emb": d / "embeddings.npz",
            "tfidf": d / "tfidf.pkl",
        }
 
    def load_or_build(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        p = self._paths()
        fp_now = _catalog_fingerprint(self.items)
 
        if p["meta"].exists() and p["items"].exists() and p["emb"].exists():
            try:
                meta = json.loads(p["meta"].read_text("utf-8"))
                if meta.get("catalog_fp") == fp_now:
                    self._backend_type = meta["backend_type"]
                    # load tool ids
                    tool_ids = json.loads(p["items"].read_text("utf-8"))
                    self._tool_ids = [x["id"] for x in tool_ids]
                    # load embeddings
                    data = np.load(str(p["emb"]))
                    self._embeddings = data["emb"].astype(np.float32)
 
                    # backend state
                    if self._backend_type == "tfidf":
                        import pickle
                        self._backend_state = pickle.loads(p["tfidf"].read_bytes())
                    else:
                        self._backend_state = None
                    return
            except Exception:
                # Any exception triggers rebuild
                pass
 
        self.build()
 
    def build(self) -> None:
        p = self._paths()
        texts = [build_embedding_text(it) for it in self.items]
        backend_type, backend_state, emb = self.embedder.embed_corpus(texts)
 
        # Save items (dict only, avoiding dataclass version issues)
        items_payload = [it.to_dict() for it in self.items]
        p["items"].write_text(json.dumps(items_payload, ensure_ascii=False, indent=2), "utf-8")
 
        # Save embeddings
        np.savez_compressed(str(p["emb"]), emb=emb.astype(np.float32))
 
        # Save backend state
        if backend_type == "tfidf":
            import pickle
            p["tfidf"].write_bytes(pickle.dumps(backend_state))
        else:
            if p["tfidf"].exists():
                try:
                    p["tfidf"].unlink()
                except Exception:
                    pass
 
        meta = {
            "backend_type": backend_type,
            "catalog_fp": _catalog_fingerprint(self.items),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedder_model": self.embedder.model_name,
        }
        p["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
 
        self._backend_type = backend_type
        self._backend_state = backend_state
        self._embeddings = emb.astype(np.float32)
        self._tool_ids = [it.id for it in self.items]
 
    def search(self, query: str, top_k: int = 3) -> List[Candidate]:
        if not self.ready:
            self.load_or_build()
        assert self._embeddings is not None
        assert self._tool_ids is not None
        assert self._backend_type is not None
 
        qv = self.embedder.embed_query(query, self._backend_type, self._backend_state)
        qv = qv.astype(np.float32)
 
        # Cosine similarity: embeddings are already normalized (hf) / or normalized during tfidf
        scores = self._embeddings @ qv
        k = max(1, min(int(top_k), len(self._tool_ids)))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
 
        return [
            Candidate(tool_id=self._tool_ids[i], score=float(scores[i]))
            for i in idx
        ]
 
    def get_item(self, tool_id: str) -> Optional[ToolItem]:
        return self._id_to_item.get(tool_id)
    
    def close(self) -> None:
        """
        Release RAG memory resources.
        """
        self._embeddings = None
        self._tool_ids = None
        self._backend_state = None
        self._backend_type = None
        gc.collect( )
