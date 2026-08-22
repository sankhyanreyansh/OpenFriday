import os
import re
from datetime import datetime
from typing import Dict, List, Tuple


class MemoryVault:
    """Zero-dependency local Markdown-based Long-Term Memory Vault with BM25/keyword RAG for FRIDAY."""

    def __init__(self, vault_dir: str = "memory_vault"):
        self.vault_dir = os.path.abspath(vault_dir)
        os.makedirs(self.vault_dir, exist_ok=True)
        self._ensure_default_files()

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

    def save_memory(self, filename: str, content: str) -> str:
        """Appends a timestamped memory entry to the specified markdown category file."""
        if not filename.endswith(".md"):
            filename += ".md"
        path = os.path.join(self.vault_dir, filename)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n- **[{timestamp}]**: {content.strip()}\n"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(entry)
            print(f"[MEMORY VAULT] Saved memory to {filename}: '{content.strip()}'")
            return f"Memory successfully saved to {filename}."
        except Exception as e:
            err_msg = f"Failed to save memory to {filename}: {e}"
            print(f"[MEMORY VAULT ERROR] {err_msg}")
            return err_msg

    def retrieve_relevant_memories(self, query: str, top_k: int = 3) -> str:
        """
        Retrieves top-k relevant memory snippets using BM25-style term-frequency scoring,
        recency boosting, and personal entity category matching.
        """
        if not query or not query.strip() or not os.path.exists(self.vault_dir):
            return ""

        chunks: List[Tuple[str, str, int]] = []  # (filename, chunk_text, line_index)
        line_counter = 0

        for fname in sorted(os.listdir(self.vault_dir)):
            if fname.endswith(".md"):
                fpath = os.path.join(self.vault_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines:
                            line_counter += 1
                            clean = line.strip()
                            if clean.startswith("- **[") or (clean and not clean.startswith("#") and len(clean) > 3):
                                chunks.append((fname, clean, line_counter))
                except Exception as e:
                    print(f"[MEMORY VAULT ERROR] Failed to read {fname} for RAG: {e}")

        if not chunks:
            return ""

        # Normalize query tokens (filter short noise words)
        raw_tokens = re.findall(r"[a-zA-Z0-9_]+", query.lower())
        stopwords = {
            "what", "is", "my", "the", "a", "an", "of", "and", "or", "to", "in",
            "for", "on", "with", "at", "by", "from", "about", "as", "into", "like",
            "through", "after", "over", "between", "out", "against", "during", "without",
            "before", "under", "around", "among", "do", "you", "remember", "know", "tell", "me"
        }
        query_tokens = [t for t in raw_tokens if t not in stopwords and len(t) > 1]

        if not query_tokens:
            query_tokens = [t for t in raw_tokens if len(t) > 1]

        # Check for personal identity hints
        is_personal_query = any(k in query.lower() for k in [
            "name", "birthday", "prefer", "favorite", "who am i", "who i am",
            "my ", "sister", "brother", "friend", "email", "phone", "profile"
        ])

        scored: List[Tuple[float, str, str]] = []
        total_docs = len(chunks)

        for fname, chunk, line_idx in chunks:
            chunk_lower = chunk.lower()
            chunk_tokens = re.findall(r"[a-zA-Z0-9_]+", chunk_lower)
            if not chunk_tokens:
                continue

            matched_terms = 0
            score = 0.0
            for q_term in query_tokens:
                if q_term in chunk_lower:
                    matched_terms += 1
                    tf = chunk_lower.count(q_term)
                    score += (tf / (len(chunk_tokens) + 2.0)) * 10.0

            # If no terms matched and not a targeted personal profile query, skip
            if matched_terms == 0 and not (is_personal_query and fname == "user_profile.md"):
                continue

            # Category file relevance boosts
            if is_personal_query and fname == "user_profile.md":
                score += 5.0
            if "project" in query.lower() and fname == "projects.md":
                score += 5.0
            if "note" in query.lower() and fname == "notes.md":
                score += 3.0

            # Recency bias (newer entries receive slight boost)
            recency_boost = (line_idx / max(total_docs, 1)) * 0.5
            score += recency_boost

            if score > 0.5:
                scored.append((score, fname, chunk))

        if not scored:
            # Fallback for personal queries to latest profile entries if no exact token overlap
            if is_personal_query:
                user_chunks = [c for c in chunks if c[0] == "user_profile.md"]
                if user_chunks:
                    top_user = user_chunks[-top_k:]
                    return "\n".join([f"[{fname}] {chunk}" for fname, chunk, _ in top_user])
            return ""

        scored.sort(key=lambda x: x[0], reverse=True)
        top_snippets = [f"[{fname}] {chunk}" for _, fname, chunk in scored[:top_k]]
        return "\n".join(top_snippets)

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
