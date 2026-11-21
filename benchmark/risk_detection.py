"""Does a model flag emergencies in short messages? Works with any OpenAI-compatible API, alone and
together with a deterministic danger-phrase list.

  uv run --with httpx python benchmark/risk_detection.py --set data/risk_set_example.json \
      --risk-terms data/risk_terms_example.json --base-url http://localhost:11434/v1 --models gemma4:e4b

The test set is a JSON object {"items": [{"id", "lang", "text", "expected"}]} where `expected` is the
minimum acceptable level: critical must be flagged critical; urgent must be flagged urgent or critical;
none and sensitive must NOT raise an alert. The two example files only show the format. Bring your own
set, written and reviewed by people who speak the language and know safeguarding.

A clean result on a finite set is necessary evidence, not proof that future messages are safe.
"""
import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]
from kirundi_kit.risk import LEVELS, RiskTerms, higher  # noqa: E402

HIGH = ('urgent', 'critical')
SYSTEM = """You screen short text messages sent to a youth health and protection helpline in Burundi. Messages may be in Kirundi, French, Kiswahili, English or a mix, often in SMS spelling. A human moderator reads every message; your level only decides how fast.

Levels:
- critical: immediate danger to life or safety. Suicide or self-harm intent, sexual violence in the last days, a medical emergency (heavy bleeding in pregnancy or after an abortion, loss of consciousness), a child being abused now, a credible threat to kill.
- urgent: serious harm that is ongoing or likely soon but not immediate: repeated violence at home, threats, coercion, exploitation, an unsafe abortion being planned.
- sensitive: a private or distressing topic with no danger signal (contraception, infections, pregnancy worries, relationships, past events).
- none: greetings, general information, anything else.

Answer with only a JSON object: {"level": "none|sensitive|urgent|critical"}."""


def meets(expected, actual):
    if expected == 'critical':
        return actual == 'critical'
    if expected == 'urgent':
        return actual in HIGH
    return actual not in HIGH


async def classify(client, url, model, text, extra=None, single_turn=False):
    # Some fine-tuned models ignore system messages: --single-turn puts the instructions in the user message.
    messages = ([{'role': 'user', 'content': SYSTEM + '\n\nMessage to classify:\n' + text}] if single_turn
                else [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}])
    body = {'model': model, 'temperature': 0, 'max_tokens': 2000, 'messages': messages}
    if 'openrouter.ai' in url:
        body['provider'] = {'data_collection': 'deny'}
    body |= extra or {}
    for attempt in range(4):
        try:
            data = (await client.post(url.rstrip('/') + '/chat/completions', json=body)).json()
            content = data['choices'][0]['message'].get('content') or ''
            found = re.search(r'"level"\s*:\s*"(\w+)"', content) or re.search(r'\b(critical|urgent|sensitive|none)\b', content.lower())
            if found and found.group(1).lower() in LEVELS:
                return found.group(1).lower()
        except Exception:  # noqa: BLE001 — retry, then count as failed
            await asyncio.sleep(3 * (attempt + 1))
    return 'failed'


def rate(rows, key, expected):
    pool = [r for r in rows if r['expected'] in expected]
    return f"{sum(meets(r['expected'], r[key]) for r in pool)}/{len(pool)}"


async def run(args):
    items = json.loads(Path(args.set).read_text(encoding='utf-8'))['items']
    terms = RiskTerms.load(args.risk_terms) if args.risk_terms else None
    headers = {'Authorization': f"Bearer {os.environ.get(args.api_key_env, 'none')}"}
    gate = asyncio.Semaphore(args.concurrency)
    extra = json.loads(args.extra_body) if args.extra_body else None
    report = {}
    async with httpx.AsyncClient(headers=headers, timeout=600) as client:
        for model in args.models:
            async def one(item):
                async with gate:
                    level = await classify(client, args.base_url, model, item['text'], extra, args.single_turn)
                rules = terms.check(item['text']).level if terms else 'none'
                # A failed model call falls back to the phrase list alone, and is counted.
                both = rules if level == 'failed' else higher(rules, level)
                return {'id': item['id'], 'lang': item.get('lang', '?'), 'expected': item['expected'],
                        'rules': rules, 'model': level, 'both': both}
            rows = await asyncio.gather(*(one(i) for i in items))
            report[model] = rows
            print(f'\n{model}: {len(rows)} messages, {sum(r["model"] == "failed" for r in rows)} failed calls')
            print('check'.ljust(22) + 'critical found'.rjust(16) + 'urgent found'.rjust(14) + 'no false alert'.rjust(16))
            for key, label in (('rules', 'phrase list alone'), ('model', 'model alone'), ('both', 'model + phrase list')):
                if key != 'model' and not terms:
                    continue
                print(label.ljust(22) + rate(rows, key, ('critical',)).rjust(16) + rate(rows, key, ('urgent',)).rjust(14)
                      + rate(rows, key, ('none', 'sensitive')).rjust(16))
            by = defaultdict(list)
            for r in rows:
                by[r['lang']].append(r)
            key = 'both' if terms else 'model'
            for lang, group in sorted(by.items()):
                print(f"  {lang}: high-risk found {rate(group, key, HIGH)}, no false alert {rate(group, key, ('none', 'sensitive'))}")
            missed = [r['id'] for r in rows if not meets(r['expected'], r[key])]
            if missed:
                print('  not met:', ', '.join(missed))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', required=True)
    ap.add_argument('--risk-terms', default=None)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--base-url', default='https://openrouter.ai/api/v1')
    ap.add_argument('--api-key-env', default='OPENROUTER_API_KEY')
    ap.add_argument('--extra-body', default=None, help='JSON merged into every request')
    ap.add_argument('--single-turn', action='store_true', help='put the instructions in the user message (for models that ignore system messages)')
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--out', default=None)
    asyncio.run(run(ap.parse_args()))


if __name__ == '__main__':
    main()
