"""Kirundi SMS text tools: normalisation, glossary matching, Kinyarwanda-drift markers, reply language.

Language models read Kirundi only moderately, and SMS Kirundi (dropped vowels, glued words, French
mixing) even less. This deterministic layer gives a model and a human moderator an anchor: known
terms with their French meaning, candidate topics and risk signals. Nothing here decides alone.

Standard library only.
"""
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

LEVELS = ('none', 'sensitive', 'urgent', 'critical')


def normalise(text):
    """Lower-case, strip accents, commas-as-apostrophes (n,impore), and letters repeated for emphasis."""
    text = unicodedata.normalize('NFKD', text.casefold())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"(?<=\w)[,’`](?=\w)", "'", text)
    text = re.sub(r'(\w)\1{2,}', r'\1', text)           # mahorooo -> mahoro
    text = re.sub(r"[^\w' ]+", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokens(text):
    return [t for t in re.split(r"[ ']", text) if t]


@dataclass
class Match:
    hits: list = field(default_factory=list)
    subthemes: dict = field(default_factory=dict)
    risk: str = 'none'
    normalised: str = ''

    def top_subthemes(self, n=3):
        return [s for s, _ in sorted(self.subthemes.items(), key=lambda kv: -kv[1])[:n]]

    def gloss(self):
        """Compact hint for a model or a moderator: 'kirundi (written « as typed ») = french'."""
        return [f"{h['kirundi']}{' (written « ' + h['matched'] + ' »)' if h['matched'] != h['kirundi'].casefold() else ''} = {h['french']}"
                + ('' if h['status'] == 'approved' else ' [not validated]') for h in self.hits]


class Glossary:
    """A Kirundi–French glossary with informal spellings, matched against SMS text.

    Each term is a dict: {"kirundi": "...", "variants": ["sms spelling", ...], "french": "...",
    "subthemes": ["topic", ...], "risk_signal": "none|sensitive|urgent|critical", "status": "draft|approved"}.
    See data/glossary_example.json.
    """

    def __init__(self, terms):
        self.phrases, self.words = [], {}
        for i, term in enumerate(terms):
            row = {'id': i, 'variants': [], 'subthemes': [], 'risk_signal': 'none', 'status': 'draft'} | term
            if row['status'] == 'rejected':
                continue
            for form in {row['kirundi'], *row['variants']}:
                norm = normalise(form)
                if not norm:
                    continue
                if ' ' in norm or "'" in norm:
                    self.phrases.append((norm, row))
                else:
                    self.words.setdefault(norm, []).append(row)
        self.phrases.sort(key=lambda p: -len(p[0]))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(data['items'] if isinstance(data, dict) else data)

    def match(self, text):
        norm = normalise(text)
        result = Match(normalised=norm)
        seen = set()

        def add(row, matched):
            if row['id'] in seen:
                return
            seen.add(row['id'])
            result.hits.append({'kirundi': row['kirundi'], 'french': row['french'], 'matched': matched,
                                'subthemes': row['subthemes'], 'risk': row['risk_signal'], 'status': row['status']})
            weight = 2 if row['status'] == 'approved' else 1
            for s in row['subthemes']:
                result.subthemes[s] = result.subthemes.get(s, 0) + weight
            if LEVELS.index(row['risk_signal']) > LEVELS.index(result.risk):
                result.risk = row['risk_signal']

        padded = f' {norm} '
        for phrase, row in self.phrases:
            if f' {phrase}' in padded:
                add(row, phrase)
        for token in tokens(norm):
            if token in self.words:
                for row in self.words[token]:
                    add(row, token)
                continue
            for form, rows in self.words.items():
                # Long stems inside agglutinated words (ndiyahura ⊃ iyahura); SMS forms missing the final vowel.
                if (len(form) >= 6 and form in token) or (len(token) >= 4 and form.startswith(token) and len(form) - len(token) == 1):
                    for row in rows:
                        add(row, token)
        return result


# Kinyarwanda is close to Kirundi and models often drift into it. These spellings are typical of
# Kinyarwanda and (almost) absent from standard Kirundi orthography.
KINYARWANDA_MARKERS = [
    (re.compile(r'\bcyangwa\b'), 'cyangwa (Kirundi: canke)'),
    (re.compile(r'\b\w*cy[aeiou]\w*'), 'cy… (Kirundi: c…, e.g. icyo → ico)'),
    (re.compile(r'\b(?:by|bi)by\w*|\bby[aeiou]\w*'), 'by… (Kirundi: vy…, e.g. byiza → vyiza)'),
    (re.compile(r'\b\w*jy[aeiou]\w*'), 'jy… (Kirundi: j…/z…)'),
    (re.compile(r'\b\w*shy[aeiou]\w*'), 'shy… (Kirundi: sh…)'),
]


def kinyarwanda_markers(text):
    """Kinyarwanda-looking forms in a supposedly Kirundi text (for a warning, never an automatic fix)."""
    folded = normalise(text)
    found = []
    for pattern, label in KINYARWANDA_MARKERS:
        words = sorted({m.group(0) for m in pattern.finditer(folded)})
        if words:
            found.append(f"{', '.join(words[:4])}: {label}")
    return found


# Frequent Kirundi words in SMS (greetings, pronouns, question words, verbs of asking and wanting).
KIRUNDI_WORDS = frozenset('''
amahoro bite mwaramutse mwiriwe waramutse wiriwe murakoze urakoze ego oya ndi uri turi muri bari ndafise mfise ufise
dufise mufise ndashaka nshaka ndasaba nsaba ndabaza mbaza nabonye ndakeneye nkeneye sinzi simbizi nta ntaco ntabwo
kandi canke ariko kuko kubera kugira ngo nimba none rero cane gusa vuba ubu ico ivyo iyo izo uwo abo aho uko
iki iyihe gute ryari hehe nde ninde kubera iki umuntu abantu umwana abana umukobwa abakobwa umuhungu umugore
umugabo inda indwara amagara ivuriro muganga ubufasha imfashanyo amakuru ubutumwa ndababaye ndwaye ndarwaye
nanje wewe we twebwe mwebwe bo yanje yawe yacu yanyu wanje wawe kwa mu ku ni'''.split())
# Frequent French words; one-letter elisions (j', l', c', d', n', qu') are split off by normalise().
FRENCH_WORDS = frozenset('''
je j me m moi mon ma mes tu te toi ton ta tes il elle on nous vous ils elles le la les l un une des de du d et ou
est es suis sont ai as a avez avons ont pas ne n plus que qu qui quoi quel quelle pour par avec sans dans sur
chez mais donc car si oui non bonjour bonsoir salut slt bjr cv merci svp stp jai comment pourquoi quand combien ou aide aider besoin
veux voudrais peux puis faire fait ce cette ces c ca cela tres bien mal enceinte grossesse regles depuis apres
avant encore aussi deja mari femme enfant enfants'''.split())
# A Kirundi word shape: noun-class or person prefix, open syllables, no French-only letter groups.
_KIRUNDI_SHAPE = re.compile(r'^(?:umu|aba|ama|imi|ibi|ivy|iki|aka|utu|ubu|uku|aha|aga|ugu|ndi|nda|ndu|ntu|tu|mu|ku|ba)'
                            r'(?:[bcdfghjkmnprstvwyz]{1,3}[aeiou])+$')
_FRENCH_ONLY = re.compile(r'qu|x|eau|ou[rs]$|tion|ll|ph|th|[aeiou]{3}')


def reply_language(text, glossary_hits=()):
    """'rn', 'fr' or None for a person's message, before any model has read it.

    Kirundi when the message shows any clear Kirundi, French when it is French only. Mixed messages get Kirundi.
    """
    words = tokens(normalise(text))
    kirundi = 2 * sum(w in KIRUNDI_WORDS for w in words) + len(glossary_hits)
    kirundi += sum(1 for w in words if w not in FRENCH_WORDS and w not in KIRUNDI_WORDS and len(w) >= 5
                   and _KIRUNDI_SHAPE.match(w) and not _FRENCH_ONLY.search(w))
    french = sum(w in FRENCH_WORDS for w in words)
    if kirundi >= 2 or (kirundi and not french):
        return 'rn'
    if french:
        return 'fr'
    return None
