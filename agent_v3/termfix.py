"""Fixes misheard jargon: Parakeet has no vocabulary prompt, so names from the docs
("Kubernetes", "Meilisearch", "OnlineCook") are matched by sound against the transcript."""

import difflib
import re

WORD = re.compile(r"[A-Za-z][A-Za-z']*")
EDGE = {"and", "the", "a", "an", "of", "to", "for", "in", "on", "at", "with", "or", "is", "are", "you", "your", "my"}


def sound_key(s: str) -> str:
    """Rough phonetic key: consonant skeleton, so 'Cuban eats' ~ 'Kubernetes'."""
    s = re.sub(r"[^a-z]", "", s.lower())
    for a, b in (("ph", "f"), ("ck", "k"), ("c", "k"), ("q", "k"), ("z", "s"), ("x", "ks"), ("ee", "i")):
        s = s.replace(a, b)
    if not s:
        return ""
    head, tail = s[0], re.sub(r"[aeiouyhw]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", head + tail)


class TermFixer:
    def __init__(self, terms: list[str], common: set[str]):
        # Only distinctive names: ordinary words ("Google", "Router") are transcribed fine already.
        self.terms = [
            t for t in terms
            if len(re.sub(r"[^A-Za-z]", "", t)) >= 6 and "-" not in t and not t.islower()
            and (t.lower() not in common or any(c.isupper() for c in t[1:]))  # keep TypeScript
        ]
        self.keys = {t: (re.sub(r"[^a-z]", "", t.lower()), sound_key(t)) for t in self.terms}
        self.common = common

    def _is_common(self, w: str) -> bool:
        w = w.lower()
        return w in self.common or (len(w) >= 5 and w.endswith("s") and w[:-1] in self.common)  # "rooms"

    def _match(self, phrase: str, n: int) -> str | None:
        flat, key = re.sub(r"[^a-z]", "", phrase.lower()), sound_key(phrase)
        best, best_score = None, 0.0
        for t, (tflat, tkey) in self.keys.items():
            if flat == tflat:
                return t if phrase != t else None  # "online cook" -> "OnlineCook"
            if not (0.7 <= len(flat) / len(tflat) <= 1.35):
                continue
            chars = difflib.SequenceMatcher(None, flat, tflat).ratio()
            sound = difflib.SequenceMatcher(None, key, tkey).ratio()
            # A real word ("planets") needs a much closer match than nonsense ("Melis Art").
            all_common = all(self._is_common(w) for w in WORD.findall(phrase))
            need = (0.9, 0.85) if all_common and n == 1 else (0.75, 0.9) if all_common else (0.6, 0.8)
            if chars >= need[0] and sound >= need[1] and chars + sound > best_score:
                best, best_score = t, chars + sound
        return best

    def fix(self, text: str) -> str:
        words = list(WORD.finditer(text))
        out, i, last = [], 0, 0
        while i < len(words):
            for n in (1, 2, 3):
                if i + n > len(words):
                    continue
                span = text[words[i].start():words[i + n - 1].end()]
                if n > 1 and (re.search(r"[.,!?;:]", span) or {words[i].group().lower(), words[i + n - 1].group().lower()} & EDGE):
                    continue
                term = self._match(span, n)
                if term:
                    out += [text[last:words[i].start()], term]
                    last, i = words[i + n - 1].end(), i + n
                    break
            else:
                i += 1
        return "".join(out) + text[last:]
