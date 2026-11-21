"""Deterministic high-risk word and phrase check. Meant to run on every incoming message before any model.

The list (data/risk_terms.json) is a recall aid, never a replacement for a model or a person: it catches
plain wording in the languages it covers and misses indirect or coded wording.

Standard library only.
"""
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

LEVELS = ('none', 'sensitive', 'urgent', 'critical')


def fold(text):
    """Lowercase and strip accents so 'Violée' and 'violee' match the same rule."""
    decomposed = unicodedata.normalize('NFKD', text.casefold())
    return ''.join(c for c in decomposed if not unicodedata.combining(c)).replace('’', "'")


def higher(a, b):
    return a if LEVELS.index(a) >= LEVELS.index(b) else b


@dataclass
class RuleResult:
    level: str = 'none'
    category: str | None = None
    matched: list = field(default_factory=list)


class RiskTerms:
    def __init__(self, terms):
        self.rules = [(re.compile(r'(?<!\w)' + re.escape(fold(p)) + r'(?!\w)'), term['level'], term['category'], term['lang'])
                      for term in terms for p in term['patterns']]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8'))['terms'])

    def check(self, text):
        folded = fold(text)
        result, longest = RuleResult(), 0
        for pattern, level, category, lang in self.rules:
            found = pattern.search(folded)
            if found:
                # Record which rule fired (language, level, category), never the matched words.
                result.matched.append(f'{lang}:{level}:{category or "sensitive"}')
                # At equal level the longest phrase names the risk: "threatened to kill me" is a threat, not "kill me".
                span = len(found.group(0))
                if LEVELS.index(level) > LEVELS.index(result.level) or (level == result.level and span > longest):
                    result.level, result.category, longest = level, category, span
        return result
