"""Run with: uv run --with pytest pytest tests"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]

from kirundi_kit import Glossary, RiskTerms, kinyarwanda_markers, normalise, reply_language  # noqa: E402


def test_normalise_sms_spelling():
    assert normalise('Mahorooo!!  N,impore') == "mahoro n'impore"
    assert normalise('Violée') == 'violee'


def test_kinyarwanda_markers():
    assert kinyarwanda_markers('Icyo ni byiza cyangwa?')
    assert not kinyarwanda_markers('Ico ni vyiza canke?')


def test_reply_language():
    assert reply_language('Amahoro, ndashaka amakuru') == 'rn'
    assert reply_language('Bonjour, je voudrais de l aide') == 'fr'
    assert reply_language('Bonjour, ndashaka ubufasha') == 'rn'
    assert reply_language('ok') is None


def test_glossary_matches_informal_and_glued_forms():
    glossary = Glossary.load(ROOT / 'data' / 'glossary_example.json')
    match = glossary.match('Ndashaka kumenya aho baronkera dukingirizo kwivuriro')
    assert {h['kirundi'] for h in match.hits} == {'udukingirizo', 'ivuriro'}
    glued = glossary.match('ndiyahura')
    assert glued.risk == 'critical' and glued.hits[0]['kirundi'] == 'kwiyahura'


def test_risk_terms():
    terms = RiskTerms.load(ROOT / 'data' / 'risk_terms_example.json')
    assert terms.check("Hier il m'a VIOLÉE").level == 'critical'
    assert terms.check('Mon mari me frappe').level == 'urgent'
    assert terms.check('Bonjour').level == 'none'
