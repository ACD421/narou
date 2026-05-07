from __future__ import annotations

import os
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import SGM_DIR


STOP = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to", "of",
    "and", "or", "for", "with", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "it", "this", "that", "i", "you", "he", "she", "we",
    "they", "my", "your", "his", "her",
}


def _stem(word: str) -> str:
    word = word.lower()
    if len(word) < 4:
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("es") and len(word) > 3:
        word = word[:-2] if word[-3] in "sxzo" else word[:-1]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    if word.endswith("ied"):
        word = word[:-3] + "y"
    elif word.endswith("ed") and len(word) > 4:
        if word[-3] in "aeiou" or word[-4] in "aeiou":
            word = word[:-2]
        elif word[-3] == word[-4]:
            word = word[:-3]
    if word.endswith("ing") and len(word) > 5:
        if word[-4] in "aeiou" or word[-5] in "aeiou":
            word = word[:-3]
        elif word[-4] == word[-5]:
            word = word[:-4]
    for sfx, rep in [
        ("ational", "ate"), ("tional", "tion"), ("ness", ""), ("ment", ""),
        ("ful", ""), ("less", ""), ("ly", ""), ("ity", ""), ("ive", ""),
        ("ize", ""), ("al", ""), ("er", ""), ("or", ""),
    ]:
        if word.endswith(sfx) and len(word) > len(sfx) + 2:
            word = word[:-len(sfx)] + rep
            break
    return word


_WORD_RE = re.compile(r"\b[a-z]+\b")


def _cw(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOP and len(w) > 2]


def _stems(text: str) -> set[str]:
    return {_stem(w) for w in _cw(text)}


def _mktok(sizes):
    def tok(text: str) -> list[str]:
        ng = []
        for word in _cw(text):
            w = f"^{word}$"
            for n in sizes:
                for i in range(len(w) - n + 1):
                    ng.append(w[i:i + n])
        return ng
    return tok


_TOKENIZERS = {
    "ng23": _mktok([2, 3]),
    "ng34": _mktok([3, 4]),
    "ng3": _mktok([3]),
}


def stem_overlap(t1: str, t2: str) -> float:
    s1, s2 = _stems(t1), _stems(t2)
    return len(s1 & s2) / len(s1 | s2) if s1 and s2 else 0.0


def exact_overlap(t1: str, t2: str) -> float:
    w1 = set(_WORD_RE.findall(t1.lower())) - STOP
    w2 = set(_WORD_RE.findall(t2.lower())) - STOP
    return len(w1 & w2) / len(w1 | w2) if w1 and w2 else 0.0


def containment(t1: str, t2: str) -> float:
    s1, s2 = _stems(t1), _stems(t2)
    if not s1 or not s2:
        return 0.0
    a, b = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    return len(a & b) / len(a)


def dice(t1: str, t2: str) -> float:
    s1, s2 = _stems(t1), _stems(t2)
    return 2 * len(s1 & s2) / (len(s1) + len(s2)) if s1 and s2 else 0.0


def char3(t1: str, t2: str) -> float:
    def c3(t: str) -> set[str]:
        t = t.lower()
        return {t[i:i + 3] for i in range(len(t) - 2)}

    a, b = c3(t1), c3(t2)
    return len(a & b) / len(a | b) if a and b else 0.0


@dataclass(frozen=True)
class TextFeatures:
    """Precomputed lexical features for a text block.

    Compute once, reuse across many comparisons. Designed to be serializable
    so the jobs cache can persist them and skip re-tokenization on every query.
    """
    stems: frozenset = field(default_factory=frozenset)
    char3: frozenset = field(default_factory=frozenset)
    words: frozenset = field(default_factory=frozenset)

    @property
    def is_empty(self) -> bool:
        return not self.stems

    def to_dict(self) -> dict:
        return {
            "stems": list(self.stems),
            "char3": list(self.char3),
            "words": list(self.words),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TextFeatures":
        return cls(
            stems=frozenset(d.get("stems") or ()),
            char3=frozenset(d.get("char3") or ()),
            words=frozenset(d.get("words") or ()),
        )


def compute_features(text: str) -> TextFeatures:
    """One-shot tokenize + stem + char3 for a block of text."""
    if not text:
        return TextFeatures()
    text_lc = text.lower()
    words_all = _WORD_RE.findall(text_lc)
    words = frozenset(w for w in words_all if w not in STOP and len(w) > 2)
    if not words:
        return TextFeatures()
    stems = frozenset(_stem(w) for w in words)
    if len(text_lc) >= 3:
        char3 = frozenset(text_lc[i:i + 3] for i in range(len(text_lc) - 2))
    else:
        char3 = frozenset()
    return TextFeatures(stems=stems, char3=char3, words=words)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _dice_from_sets(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    total = len(a) + len(b)
    if total == 0:
        return 0.0
    return 2 * len(a & b) / total


def _containment_from_sets(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    smaller, _larger = (a, b) if len(a) <= len(b) else (b, a)
    if not smaller:
        return 0.0
    return len(a & b) / len(smaller)


def _lenratio_from_features(a: TextFeatures, b: TextFeatures) -> float:
    l1, l2 = len(a.words), len(b.words)
    if max(l1, l2) == 0:
        return 1.0
    return min(l1, l2) / max(l1, l2)


def _bridge_from_features(
    a: TextFeatures, b: TextFeatures, syn_dict: dict
) -> float:
    """Feature-based synonym bridge -- approximation that reuses cached stems.

    Falls back to a pure Jaccard value when the two sides share no synonyms,
    which matches _bridge()'s behavior in the dense-overlap regime.
    """
    if not a.stems or not b.stems:
        return 0.0
    inter = a.stems & b.stems
    union = a.stems | b.stems
    if not union:
        return 0.0
    only_a = a.stems - b.stems
    only_b = b.stems - a.stems
    if not only_a and not only_b:
        return 1.0
    if not only_a or not only_b:
        return len(inter) / len(union)
    # Cheap synonym probe: look up a handful of (stem, stem) pairs.
    bridges = 0
    probed = 0
    for s1 in list(only_a)[:20]:
        best = 0.0
        for s2 in list(only_b)[:20]:
            c = syn_dict.get((s1, s2), 0)
            if c > best:
                best = c
        probed += 1
        if best > 0:
            bridges += min(1.0, best / 2.0)
    return bridges / probed if probed else 0.5


def fast_similarity_features(a: TextFeatures, b: TextFeatures) -> float:
    """SGMEngine.fast_similarity() but on precomputed features.

    Equivalent to 0.45*stem_overlap + 0.35*char3 + 0.20*dice, but avoids
    the 6 redundant tokenizations per pair that the string form does.
    """
    if a.is_empty or b.is_empty:
        return 0.0
    s_inter = len(a.stems & b.stems)
    if s_inter == 0 and not (a.char3 & b.char3):
        return 0.0
    s_union = len(a.stems | b.stems)
    stem_j = s_inter / s_union if s_union else 0.0
    s_sum = len(a.stems) + len(b.stems)
    dice_v = 2 * s_inter / s_sum if s_sum else 0.0
    if a.char3 and b.char3:
        c_inter = len(a.char3 & b.char3)
        c_union = len(a.char3 | b.char3)
        char3_j = c_inter / c_union if c_union else 0.0
    else:
        char3_j = 0.0
    return 0.45 * stem_j + 0.35 * char3_j + 0.20 * dice_v


def lenratio(t1: str, t2: str) -> float:
    l1, l2 = len(t1.split()), len(t2.split())
    return min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0


def numatch(t1: str, t2: str) -> float:
    n1 = set(re.findall(r"\b\d+(?:\.\d+)?\b", t1))
    n2 = set(re.findall(r"\b\d+(?:\.\d+)?\b", t2))
    if not n1 and not n2:
        return 0.5
    if not n1 or not n2:
        return 0.3
    return len(n1 & n2) / len(n1 | n2)


def _bridge(t1: str, t2: str, syn_dict: dict) -> float:
    words1, words2 = _cw(t1), _cw(t2)
    if not words1 or not words2:
        return 0.0
    stems1 = {_stem(w) for w in words1}
    stems2 = {_stem(w) for w in words2}
    only1 = [w for w in words1 if _stem(w) not in stems2]
    only2 = [w for w in words2 if _stem(w) not in stems1]
    if not only1 and not only2:
        return 1.0
    if not only1 or not only2:
        ov = len(stems1 & stems2)
        tot = len(stems1 | stems2)
        return ov / tot if tot > 0 else 0.5
    bridges, total = 0, 0
    for w1 in only1:
        best = 0.0
        for w2 in only2:
            c = syn_dict.get((w1, w2), 0)
            if c > best:
                best = c
            c2 = syn_dict.get((_stem(w1), _stem(w2)), 0)
            if c2 > best:
                best = c2
        total += 1
        if best > 0:
            bridges += min(1.0, best / 2.0)
    return bridges / total if total > 0 else 0.5


_LEX_FNS = {
    "stem": stem_overlap,
    "exact": exact_overlap,
    "contain": containment,
    "dice": dice,
    "char3": char3,
    "lenratio": lenratio,
    "numatch": numatch,
}


class SGMEngine:
    def __init__(self, model_dir: str | Path = SGM_DIR):
        self.model_dir = Path(model_dir)
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"SGM manifest not found at {manifest_path}")
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.weights = np.array(self.manifest["weights"], dtype=np.float32)
        self.feat_names = self.manifest["feature_names"]
        self.groups = []

        total_bytes = 0
        for g in self.manifest["groups"]:
            tok_fn = _TOKENIZERS[g["tok_key"]]
            with open(self.model_dir / g["vocab_file"]) as f:
                tok_list = json.load(f)
            vocab = {tok: i for i, tok in enumerate(tok_list)}
            total_bytes += os.path.getsize(self.model_dir / g["vocab_file"])
            idf = np.fromfile(
                self.model_dir / g["idf_file"], dtype=np.float16
            ).astype(np.float32)
            total_bytes += os.path.getsize(self.model_dir / g["idf_file"])
            embs = []
            for m in g["models"]:
                fpath = self.model_dir / m["file"]
                with open(fpath, "rb") as f:
                    scale = struct.unpack("f", f.read(4))[0]
                    q = np.frombuffer(
                        f.read(), dtype=np.int8
                    ).reshape(m["vocab_size"], m["dim"])
                embs.append(q.astype(np.float32) * scale)
                total_bytes += os.path.getsize(fpath)
            self.groups.append({
                "name": g["name"],
                "tok_fn": tok_fn,
                "vocab": vocab,
                "idf": idf,
                "embs": embs,
            })

        syn_path = self.model_dir / self.manifest["synonyms_file"]
        with open(syn_path) as f:
            raw = json.load(f)
        self.synonyms = {}
        for key, conf in raw.items():
            w1, w2 = key.split("|")
            self.synonyms[(w1, w2)] = conf
        total_bytes += os.path.getsize(syn_path)
        self._total_bytes = total_bytes

    @property
    def size_kb(self) -> float:
        return self._total_bytes / 1024

    @property
    def size_mb(self) -> float:
        return self._total_bytes / 1024 / 1024

    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v)
        return v / (n + 1e-8) if n > 0 else v

    def _embed(self, emb, vocab, tok_fn, text):
        idx = [vocab[t] for t in tok_fn(text) if t in vocab]
        if not idx:
            return None
        return emb[idx].mean(0)

    def _embed_norm(self, emb, vocab, tok_fn, text):
        v = self._embed(emb, vocab, tok_fn, text)
        return self._normalize(v) if v is not None else None

    def _embed_idf(self, emb, vocab, tok_fn, idf, text):
        idx = [vocab[t] for t in tok_fn(text) if t in vocab]
        if not idx:
            return None
        weights = idf[idx]
        return (emb[idx] * weights[:, None]).sum(0) / (weights.sum() + 1e-8)

    def _embed_idf_norm(self, emb, vocab, tok_fn, idf, text):
        v = self._embed_idf(emb, vocab, tok_fn, idf, text)
        return self._normalize(v) if v is not None else None

    @staticmethod
    def _cosine_normed(a, b) -> float:
        return float(np.dot(a, b))

    @staticmethod
    def _cosine(a, b) -> float:
        na = np.linalg.norm(a) + 1e-8
        nb = np.linalg.norm(b) + 1e-8
        return float(np.dot(a / na, b / nb))

    @staticmethod
    def _l2_sim(a, b) -> float:
        return 1.0 / (1.0 + float(np.linalg.norm(a - b)))

    def precompute_doc(self, text: str) -> dict:
        """Precompute all per-group cos/idf embeddings for a document.

        Returns a dict usable as either side of predict_pre(). Vectors are
        pre-normalized so cosine similarity is a single dot product.
        """
        out: dict = {"_text": text, "_features": compute_features(text), "_normed": True}
        for g in self.groups:
            cos_embeds = [
                self._embed_norm(emb, g["vocab"], g["tok_fn"], text)
                for emb in g["embs"]
            ]
            raw_embeds = [
                self._embed(emb, g["vocab"], g["tok_fn"], text)
                for emb in g["embs"]
            ]
            idf_embeds = [
                self._embed_idf_norm(emb, g["vocab"], g["tok_fn"], g["idf"], text)
                for emb in g["embs"]
            ]
            out[g["name"]] = {"cos": cos_embeds, "idf": idf_embeds, "raw": raw_embeds}
        return out

    def predict(self, s1: str, s2: str) -> float:
        return self.predict_pre(s1, s2)

    def predict_pre(self, doc1, doc2) -> float:
        """predict(s1, s2) but either argument can be a precomputed doc dict."""
        is_dict1 = isinstance(doc1, dict)
        is_dict2 = isinstance(doc2, dict)
        s1 = doc1["_text"] if is_dict1 else doc1
        s2 = doc2["_text"] if is_dict2 else doc2
        both_normed = (
            is_dict1 and is_dict2
            and doc1.get("_normed") and doc2.get("_normed")
        )
        cos_fn = self._cosine_normed if both_normed else self._cosine

        feats = []
        for g in self.groups:
            name = g["name"]
            cos_vals = []
            l2_vals = []
            for i, emb in enumerate(g["embs"]):
                if is_dict1:
                    e1 = doc1[name]["cos"][i]
                else:
                    e1 = self._embed(emb, g["vocab"], g["tok_fn"], s1)
                if is_dict2:
                    e2 = doc2[name]["cos"][i]
                else:
                    e2 = self._embed(emb, g["vocab"], g["tok_fn"], s2)
                if e1 is not None and e2 is not None:
                    cos_vals.append(cos_fn(e1, e2))
                    # L2 needs raw (un-normalized) vectors
                    if both_normed:
                        r1 = doc1[name].get("raw", [None])[i] if is_dict1 else e1
                        r2 = doc2[name].get("raw", [None])[i] if is_dict2 else e2
                        if r1 is not None and r2 is not None:
                            l2_vals.append(self._l2_sim(r1, r2))
                    else:
                        l2_vals.append(self._l2_sim(e1, e2))
            feats.append(np.mean(cos_vals) if cos_vals else 0.0)

            idf_vals = []
            for i, emb in enumerate(g["embs"]):
                if is_dict1:
                    e1 = doc1[name]["idf"][i]
                else:
                    e1 = self._embed_idf(emb, g["vocab"], g["tok_fn"], g["idf"], s1)
                if is_dict2:
                    e2 = doc2[name]["idf"][i]
                else:
                    e2 = self._embed_idf(emb, g["vocab"], g["tok_fn"], g["idf"], s2)
                if e1 is not None and e2 is not None:
                    idf_vals.append(cos_fn(e1, e2))
            feats.append(np.mean(idf_vals) if idf_vals else 0.0)

            feats.append(np.mean(l2_vals) if l2_vals else 0.0)

        # When both sides have cached features, use the feature-based lex
        # functions to skip re-tokenization. When not, fall back to the
        # original string-based implementations (correctness-preserving).
        f1 = doc1.get("_features") if is_dict1 else None
        f2 = doc2.get("_features") if is_dict2 else None
        if f1 is not None and f2 is not None and not f1.is_empty and not f2.is_empty:
            # words set here excludes stopwords but preserves exact tokens
            feats.append(_jaccard(f1.stems, f2.stems))               # stem
            feats.append(_jaccard(f1.words, f2.words))               # exact
            feats.append(_containment_from_sets(f1.stems, f2.stems)) # contain
            feats.append(_dice_from_sets(f1.stems, f2.stems))        # dice
            feats.append(_jaccard(f1.char3, f2.char3))               # char3
            feats.append(_lenratio_from_features(f1, f2))            # lenratio
            feats.append(_LEX_FNS["numatch"](s1, s2))                # numatch keeps string path
            feats.append(_bridge_from_features(f1, f2, self.synonyms))
        else:
            for fn in ["stem", "exact", "contain", "dice", "char3", "lenratio", "numatch"]:
                feats.append(_LEX_FNS[fn](s1, s2))
            feats.append(_bridge(s1, s2, self.synonyms))

        x = np.array(feats, dtype=np.float32)
        return float(np.dot(x, self.weights)) * 5.0

    def similarity(self, s1, s2) -> float:
        score = self.predict_pre(s1, s2)
        return max(0.0, min(1.0, score / 5.0))

    def similarity_pre(self, doc1, doc2) -> float:
        score = self.predict_pre(doc1, doc2)
        return max(0.0, min(1.0, score / 5.0))

    @staticmethod
    def fast_similarity(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        return (
            0.45 * stem_overlap(s1, s2)
            + 0.35 * char3(s1, s2)
            + 0.20 * dice(s1, s2)
        )


_engine_singleton: SGMEngine | None = None


def get_engine() -> SGMEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = SGMEngine()
    return _engine_singleton
