#!/usr/bin/env python3
"""
Standalone inference engine. NumPy only.
Load with: engine = Engine("deployable/")
Predict with: score = engine.predict("sentence one", "sentence two")
"""

import os, json, struct, re
import numpy as np

STOP = {'a','an','the','is','are','was','were','in','on','at','to','of',
        'and','or','for','with','be','been','being','has','have','had',
        'do','does','did','it','this','that','i','you','he','she','we',
        'they','my','your','his','her'}

def _stem(word):
    word = word.lower()
    if len(word) < 4: return word
    if word.endswith('ies') and len(word) > 4: word = word[:-3] + 'y'
    elif word.endswith('es') and len(word) > 3: word = word[:-2] if word[-3] in 'sxzo' else word[:-1]
    elif word.endswith('s') and not word.endswith('ss') and len(word) > 3: word = word[:-1]
    if word.endswith('ied'): word = word[:-3] + 'y'
    elif word.endswith('ed') and len(word) > 4:
        if word[-3] in 'aeiou' or word[-4] in 'aeiou': word = word[:-2]
        elif word[-3] == word[-4]: word = word[:-3]
    if word.endswith('ing') and len(word) > 5:
        if word[-4] in 'aeiou' or word[-5] in 'aeiou': word = word[:-3]
        elif word[-4] == word[-5]: word = word[:-4]
    for sfx, rep in [('ational','ate'),('tional','tion'),('ness',''),('ment',''),
                     ('ful',''),('less',''),('ly',''),('ity',''),('ive',''),
                     ('ize',''),('al',''),('er',''),('or','')]:
        if word.endswith(sfx) and len(word) > len(sfx) + 2:
            word = word[:-len(sfx)] + rep; break
    return word

def _cw(text):
    return [w for w in re.findall(r'\b[a-z]+\b', text.lower()) if w not in STOP and len(w) > 2]

def _stems(text): return set(_stem(w) for w in _cw(text))

def _mktok(sizes):
    def tok(text):
        ng = []
        for word in _cw(text):
            w = f'^{word}$'
            for n in sizes:
                for i in range(len(w) - n + 1):
                    ng.append(w[i:i+n])
        return ng
    return tok

_TOKENIZERS = {'ng23': _mktok([2,3]), 'ng34': _mktok([3,4]), 'ng3': _mktok([3])}

def _stem_overlap(t1, t2):
    s1, s2 = _stems(t1), _stems(t2)
    return len(s1 & s2) / len(s1 | s2) if s1 and s2 else 0.0

def _exact_overlap(t1, t2):
    w1 = set(re.findall(r'\b[a-z]+\b', t1.lower())) - STOP
    w2 = set(re.findall(r'\b[a-z]+\b', t2.lower())) - STOP
    return len(w1 & w2) / len(w1 | w2) if w1 and w2 else 0.0

def _containment(t1, t2):
    s1, s2 = _stems(t1), _stems(t2)
    if not s1 or not s2: return 0.0
    a, b = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    return len(a & b) / len(a)

def _dice(t1, t2):
    s1, s2 = _stems(t1), _stems(t2)
    return 2 * len(s1 & s2) / (len(s1) + len(s2)) if s1 and s2 else 0.0

def _char3(t1, t2):
    def c3(t): t = t.lower(); return set(t[i:i+3] for i in range(len(t)-2))
    a, b = c3(t1), c3(t2)
    return len(a & b) / len(a | b) if a and b else 0.0

def _lenratio(t1, t2):
    l1, l2 = len(t1.split()), len(t2.split())
    return min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0

def _numatch(t1, t2):
    n1 = set(re.findall(r'\b\d+(?:\.\d+)?\b', t1))
    n2 = set(re.findall(r'\b\d+(?:\.\d+)?\b', t2))
    if not n1 and not n2: return 0.5
    if not n1 or not n2: return 0.3
    return len(n1 & n2) / len(n1 | n2)

def _bridge(t1, t2, syn_dict):
    words1, words2 = _cw(t1), _cw(t2)
    if not words1 or not words2: return 0.0
    stems1 = set(_stem(w) for w in words1)
    stems2 = set(_stem(w) for w in words2)
    only1 = [w for w in words1 if _stem(w) not in stems2]
    only2 = [w for w in words2 if _stem(w) not in stems1]
    if not only1 and not only2: return 1.0
    if not only1 or not only2:
        ov = len(stems1 & stems2); tot = len(stems1 | stems2)
        return ov / tot if tot > 0 else 0.5
    bridges, total = 0, 0
    for w1 in only1:
        best = 0
        for w2 in only2:
            c = syn_dict.get((w1, w2), 0)
            if c > best: best = c
            c2 = syn_dict.get((_stem(w1), _stem(w2)), 0)
            if c2 > best: best = c2
        total += 1
        if best > 0: bridges += min(1.0, best / 2.0)
    return bridges / total if total > 0 else 0.5

_LEX_FNS = {'stem': _stem_overlap, 'exact': _exact_overlap, 'contain': _containment,
            'dice': _dice, 'char3': _char3, 'lenratio': _lenratio, 'numatch': _numatch}


class Engine:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, 'manifest.json')) as f:
            self.manifest = json.load(f)

        self.weights = np.array(self.manifest['weights'], dtype=np.float32)
        self.feat_names = self.manifest['feature_names']
        self.groups = []

        total_bytes = 0
        for g in self.manifest['groups']:
            tok_fn = _TOKENIZERS[g['tok_key']]

            # Load vocab
            with open(os.path.join(model_dir, g['vocab_file'])) as f:
                tok_list = json.load(f)
            vocab = {tok: i for i, tok in enumerate(tok_list)}
            total_bytes += os.path.getsize(os.path.join(model_dir, g['vocab_file']))

            # Load IDF
            idf = np.fromfile(os.path.join(model_dir, g['idf_file']), dtype=np.float16).astype(np.float32)
            total_bytes += os.path.getsize(os.path.join(model_dir, g['idf_file']))

            # Load models
            embs = []
            for m in g['models']:
                fpath = os.path.join(model_dir, m['file'])
                with open(fpath, 'rb') as f:
                    scale = struct.unpack('f', f.read(4))[0]
                    q = np.frombuffer(f.read(), dtype=np.int8).reshape(m['vocab_size'], m['dim'])
                embs.append(q.astype(np.float32) * scale)
                total_bytes += os.path.getsize(fpath)

            self.groups.append({'name': g['name'], 'tok_fn': tok_fn, 'vocab': vocab,
                               'idf': idf, 'embs': embs})

        # Load synonyms
        syn_path = os.path.join(model_dir, self.manifest['synonyms_file'])
        with open(syn_path) as f:
            raw = json.load(f)
        self.synonyms = {}
        for key, conf in raw.items():
            w1, w2 = key.split('|')
            self.synonyms[(w1, w2)] = conf
        total_bytes += os.path.getsize(syn_path)

        self._total_bytes = total_bytes

    @property
    def size_kb(self):
        return self._total_bytes / 1024

    @property
    def size_mb(self):
        return self._total_bytes / 1024 / 1024

    def _embed(self, emb, vocab, tok_fn, text):
        idx = [vocab[t] for t in tok_fn(text) if t in vocab]
        if not idx: return None
        return emb[idx].mean(0)

    def _cosine(self, a, b):
        na, nb = np.linalg.norm(a) + 1e-8, np.linalg.norm(b) + 1e-8
        return float(np.dot(a / na, b / nb))

    def _l2_sim(self, a, b):
        return 1.0 / (1.0 + np.linalg.norm(a - b))

    def _embed_idf(self, emb, vocab, tok_fn, idf, text):
        idx = [vocab[t] for t in tok_fn(text) if t in vocab]
        if not idx: return None
        weights = idf[idx]
        return (emb[idx] * weights[:, None]).sum(0) / (weights.sum() + 1e-8)

    def predict(self, s1, s2):
        feats = []
        for g in self.groups:
            # Cosine
            cos_vals = []
            for emb in g['embs']:
                e1 = self._embed(emb, g['vocab'], g['tok_fn'], s1)
                e2 = self._embed(emb, g['vocab'], g['tok_fn'], s2)
                if e1 is not None and e2 is not None:
                    cos_vals.append(self._cosine(e1, e2))
            feats.append(np.mean(cos_vals) if cos_vals else 0.0)

            # IDF cosine
            idf_vals = []
            for emb in g['embs']:
                e1 = self._embed_idf(emb, g['vocab'], g['tok_fn'], g['idf'], s1)
                e2 = self._embed_idf(emb, g['vocab'], g['tok_fn'], g['idf'], s2)
                if e1 is not None and e2 is not None:
                    idf_vals.append(self._cosine(e1, e2))
            feats.append(np.mean(idf_vals) if idf_vals else 0.0)

            # L2 sim
            l2_vals = []
            for emb in g['embs']:
                e1 = self._embed(emb, g['vocab'], g['tok_fn'], s1)
                e2 = self._embed(emb, g['vocab'], g['tok_fn'], s2)
                if e1 is not None and e2 is not None:
                    l2_vals.append(self._l2_sim(e1, e2))
            feats.append(np.mean(l2_vals) if l2_vals else 0.0)

        # Lexical
        for fn in ['stem', 'exact', 'contain', 'dice', 'char3', 'lenratio', 'numatch']:
            feats.append(_LEX_FNS[fn](s1, s2))

        # Bridge
        feats.append(_bridge(s1, s2, self.synonyms))

        x = np.array(feats, dtype=np.float32)
        return float(np.dot(x, self.weights)) * 5.0  # scale back to 0-5


if __name__ == '__main__':
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else 'deployable'
    eng = Engine(d)
    print(f'Loaded: {eng.size_kb:.0f} KB ({eng.size_mb:.2f} MB)')
    print(f'Features: {len(eng.feat_names)}')
    pairs = [
        ("A man is playing a guitar.", "A man is playing music.", 3.8),
        ("A cat sits on a mat.", "A dog runs in a park.", 1.0),
        ("The stock market crashed today.", "Financial markets experienced a major decline.", 4.0),
        ("Two children are playing in the snow.", "Two kids play outside in winter.", 4.2),
        ("A woman is slicing potatoes.", "A man is cutting onions.", 2.5),
    ]
    for s1, s2, expected in pairs:
        score = eng.predict(s1, s2)
        print(f'  {score:.2f} (exp ~{expected:.1f}) | {s1[:40]}... vs {s2[:40]}...')
