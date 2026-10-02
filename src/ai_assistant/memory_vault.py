"""
Semantic Vector RAG Memory Vault for Open FRIDAY.
Stores user profile, preferences, project context, and personal notes in human-readable Markdown files
and performs fast, local embedding-based semantic vector retrieval using FastEmbed (BGE-Small).
"""

import os
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


class MemoryVault:
    """Local Markdown-based Long-Term Memory Vault with Semantic Vector RAG."""

    def __init__(self, vault_dir: str = "memory_vault", model_name: str = "BAAI/bge-small-en-v1.5"):
        self.vault_dir = os.path.abspath(vault_dir)
        os.makedirs(self.vault_dir, exist_ok=True)
        self._ensure_default_files()

        self.model_name = model_name
        self._embed_model: Optional[Any] = None
        self._model_lock = threading.Lock()

        # In-memory vector cache
        self._cached_chunks: List[Dict[str, Any]] = []
        self._cache_dirty = True
        self._cache_lock = threading.Lock()

    def _get_embedding_model(self):
        """Lazy loader for the local FastEmbed ONNX model."""
        if not FASTEMBED_AVAILABLE:
            return None
        with self._model_lock:
            if self._embed_model is None:
                try:
                    self._embed_model = TextEmbedding(model_name=self.model_name)
                    print(f"[MEMORY VAULT] Loaded local semantic embedding model: {self.model_name}")
                except Exception as e:
                    print(f"[MEMORY VAULT ERROR] Failed to load embedding model: {e}")
                    self._embed_model = None
            return self._embed_model

    def _ensure_default_files(self):
        """Ensures core category markdown files exist with standard headers."""
        defaults: Dict[str, str] = {
            "user_profile.md": "# User Profile & Preferences\n\n",
            "projects.md": "# Active Projects & Code Context\n\n",
            "notes.md": "# General Facts & Saved Notes\n\n",
        }
        for filename, header in defaults.items():
            path = os.path.join(self.vault_dir, filename)
            if not os.path.exists(path):
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(header)
                except Exception as e:
                    print(f"[MEMORY VAULT ERROR] Failed to initialize {filename}: {e}")

    def _load_and_index_chunks(self):
        """Reads all Markdown files in vault and builds semantic vector embeddings."""
        if not os.path.exists(self.vault_dir):
            return

        model = self._get_embedding_model()

        raw_chunks: List[Tuple[str, str]] = []  # (filename, text)

        for fname in sorted(os.listdir(self.vault_dir)):
            if fname.endswith(".md"):
                fpath = os.path.join(self.vault_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines:
                            clean = line.strip()
                            # Clean bullet points or header lines
                            if clean.startswith("- **[") or (clean and not clean.startswith("#") and len(clean) > 4):
                                raw_chunks.append((fname, clean))
                except Exception as e:
                    print(f"[MEMORY VAULT ERROR] Failed to read {fname}: {e}")

        if not raw_chunks:
            self._cached_chunks = []
            self._cache_dirty = False
            return

        if model is not None:
            try:
                texts = [c[1] for c in raw_chunks]
                embeddings = list(model.embed(texts))
                self._cached_chunks = []
                for (fname, text), emb in zip(raw_chunks, embeddings):
                    norm = np.linalg.norm(emb)
                    normed_emb = emb / norm if norm > 0 else emb
                    self._cached_chunks.append({
                        "filename": fname,
                        "text": text,
                        "embedding": normed_emb
                    })
            except Exception as e:
                print(f"[MEMORY VAULT ERROR] Error generating chunk embeddings: {e}")
                self._cached_chunks = [{"filename": f, "text": t, "embedding": None} for f, t in raw_chunks]
        else:
            self._cached_chunks = [{"filename": f, "text": t, "embedding": None} for f, t in raw_chunks]

        self._cache_dirty = False

    def save_memory(self, filename: str, content: str) -> str:
        """Appends a timestamped memory entry to the specified markdown category file and updates vector index."""
        if not filename.endswith(".md"):
            filename += ".md"
        path = os.path.join(self.vault_dir, filename)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        clean_content = content.strip()
        entry = f"\n- **[{timestamp}]**: {clean_content}\n"

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(entry)
            print(f"[MEMORY VAULT] Saved memory to {filename}: '{clean_content}'")

            # Update cache
            with self._cache_lock:
                model = self._get_embedding_model()
                if model is not None:
                    try:
                        emb = list(model.embed([clean_content]))[0]
                        norm = np.linalg.norm(emb)
                        normed_emb = emb / norm if norm > 0 else emb
                        self._cached_chunks.append({
                            "filename": filename,
                            "text": f"- **[{timestamp}]**: {clean_content}",
                            "embedding": normed_emb
                        })
                    except Exception:
                        self._cache_dirty = True
                else:
                    self._cache_dirty = True

            return f"Memory successfully saved to {filename}."
        except Exception as e:
            err_msg = f"Failed to save memory to {filename}: {e}"
            print(f"[MEMORY VAULT ERROR] {err_msg}")
            return err_msg

    def retrieve_relevant_memories(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.60
    ) -> str:
        """
        Retrieves top-k relevant memory snippets using local semantic vector similarity search.
        """
        if not query or not query.strip():
            return ""

        with self._cache_lock:
            if self._cache_dirty or not self._cached_chunks:
                self._load_and_index_chunks()

            if not self._cached_chunks:
                return ""

            model = self._get_embedding_model()

            # 1. Semantic Vector Search with FastEmbed
            if model is not None:
                try:
                    q_emb = list(model.embed([query.strip()]))[0]
                    q_norm = np.linalg.norm(q_emb)
                    normed_q = q_emb / q_norm if q_norm > 0 else q_emb

                    scored = []
                    for item in self._cached_chunks:
                        emb = item.get("embedding")
                        if emb is not None:
                            sim = float(np.dot(normed_q, emb))
                            if sim >= min_similarity:
                                scored.append((sim, item["filename"], item["text"]))

                    if scored:
                        scored.sort(key=lambda x: x[0], reverse=True)
                        top_results = scored[:top_k]
                        return "\n".join([f"[{fname}] {text}" for _, fname, text in top_results])
                except Exception as e:
                    print(f"[MEMORY VAULT ERROR] Vector query error: {e}")

            # 2. Heuristic fallback if model unavailable or no vector matches
            q_lower = query.lower()
            keyword_matches = []
            for item in self._cached_chunks:
                if any(w in item["text"].lower() for w in q_lower.split() if len(w) > 3):
                    keyword_matches.append(f"[{item['filename']}] {item['text']}")

            if keyword_matches:
                return "\n".join(keyword_matches[:top_k])

            return ""

    def read_all_memories(self, max_chars: int = 3000) -> str:
        """Aggregates all vault markdown files into a unified context snippet."""
        combined = []
        if not os.path.exists(self.vault_dir):
            return ""

        for fname in sorted(os.listdir(self.vault_dir)):
            if fname.endswith(".md"):
                fpath = os.path.join(self.vault_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if text:
                            combined.append(f"### {fname}\n{text}")
                except Exception as e:
                    print(f"[MEMORY VAULT ERROR] Failed to read {fname}: {e}")

        full_text = "\n\n".join(combined)
        return full_text[-max_chars:] if len(full_text) > max_chars else full_text
