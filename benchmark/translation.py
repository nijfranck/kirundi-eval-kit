"""Which models handle a low-resource language best? A reference-based comparison over any OpenAI-compatible API.

Written for Kirundi; the language is a parameter, so it runs for any of the 200 FLORES-200 languages.

Tasks (all scored against human text, no LLM judge):
- FLORES-200 devtest (public, human translations): the language <-> each pivot language, chrF++.
- Optional domain set (--domain): your own human-written sentence pairs, both directions, chrF++.
  On OpenRouter these are sent only to providers that do not collect data (data_collection=deny).
- Optional look-alike test (--confusable): tell apart FLORES dev sentences of the language and a close
  relative (Kirundi vs Kinyarwanda), accuracy.
- Drift: share of generated segments that show spellings of the close relative (Kirundi only for now,
  kirundi_kit.kinyarwanda_markers).

  curl -sL https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz | tar xz -C data
  export OPENROUTER_API_KEY=...
  uv run --with httpx --with sacrebleu python benchmark/translation.py \
      --flores data/flores200_dataset --models google/gemma-4-31b-it

  # another language (Kiswahili against English), and a model served locally by Ollama:
  uv run --with httpx --with sacrebleu python benchmark/translation.py --flores data/flores200_dataset \
      --lang swh_Latn --lang-name Kiswahili --pivots eng_Latn --confusable '' \
      --base-url http://localhost:11434/v1 --models gemma4:e4b

Outputs are cached per model and batch in --out so reruns resume; `--report` only rescores what is cached.
chrF++ on FLORES may be inflated for models that saw FLORES in training.
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx
import sacrebleu

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]
from kirundi_kit import kinyarwanda_markers  # noqa: E402

# Short task labels for the languages of the original Kirundi runs; other languages use their FLORES prefix.
SHORT = {'run_Latn': 'rn', 'fra_Latn': 'fr', 'eng_Latn': 'en', 'kin_Latn': 'rw'}
NAMES = {'run_Latn': 'Kirundi (Ikirundi, the language of Burundi, not Kinyarwanda)', 'fra_Latn': 'French', 'eng_Latn': 'English',
         'kin_Latn': 'Kinyarwanda'}
BATCH = 10


def short(code):
    return SHORT.get(code, code.split('_')[0])


def load_tasks(flores_dir, n, lang='run_Latn', pivots=('fra_Latn', 'eng_Latn'), confusable='kin_Latn', domain=None, names=None):
    """Tasks keyed by name. The sampling (seed 13) is fixed so every model sees the same sentences."""
    names = NAMES | (names or {})
    lines = lambda split, code: (Path(flores_dir) / split / f'{code}.{split}').read_text(encoding='utf-8').splitlines()
    rng = random.Random(13)
    test = {code: lines('devtest', code) for code in (lang, *pivots)}
    idx = sorted(rng.sample(range(len(test[lang])), n))
    tasks = {}
    for pivot in pivots:
        for src, tgt in ((lang, pivot), (pivot, lang)):
            tasks[f'flores_{short(src)}-{short(tgt)}'] = {
                'kind': 'mt', 'src': src, 'tgt': tgt, 'private': False, 'names': names,
                'items': [(test[src][i], test[tgt][i]) for i in idx]}
    if confusable:
        dev = {code: lines('dev', code) for code in (lang, confusable)}
        ids = rng.sample(range(len(dev[lang])), 50)
        lid = [(dev[lang][i], short(lang)) for i in ids[:25]] + [(dev[confusable][i], short(confusable)) for i in ids[25:]]
        rng.shuffle(lid)
        tasks[f'lid_{short(lang)}-{short(confusable)}'] = {'kind': 'lid', 'private': False, 'names': names,
                                                           'labels': (lang, confusable), 'items': lid}
    if domain:
        # A JSON list of pairs: [{"run_Latn": "...", "fra_Latn": "..."}, ...] (human-written on both sides).
        pairs = json.loads(Path(domain).read_text(encoding='utf-8'))
        other = next(code for code in pairs[0] if code != lang)
        for src, tgt in ((other, lang), (lang, other)):
            tasks[f'srhr_{short(src)}-{short(tgt)}' if lang == 'run_Latn' else f'domain_{short(src)}-{short(tgt)}'] = {
                'kind': 'mt', 'src': src, 'tgt': tgt, 'private': True, 'names': names,
                'items': [(p[src], p[tgt]) for p in pairs]}
    return tasks


def name(task, code):
    return task['names'].get(code, code)


def prompt(task, batch):
    numbered = '\n'.join(f'{i + 1}. {text}' for i, (text, _) in enumerate(batch))
    if task['kind'] == 'lid':
        a, b = task['labels']
        return (f'Each numbered sentence below is written either in {name(task, a)} or in {name(task, b)}. '
                f'Answer "{short(a)}" for the first or "{short(b)}" for the second.\n'
                f'Return only a JSON object mapping each number to "{short(a)}" or "{short(b)}", '
                f'e.g. {{"1": "{short(a)}", "2": "{short(b)}"}}.\n\n' + numbered)
    return (f"Translate each numbered segment from {name(task, task['src'])} into {name(task, task['tgt'])}. "
            'Translate faithfully and completely, one segment per number, without notes.\n'
            'Return only a JSON object mapping each number to its translation, e.g. {"1": "...", "2": "..."}.\n\n' + numbered)


def parse(text, size):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    start, end = text.find('{'), text.rfind('}')
    if start < 0:
        # Some models answer with the numbered list instead of JSON.
        data = {m.group(1): m.group(2) for m in re.finditer(r'^\s*(\d+)[.)]\s*(.+?)\s*$', text, re.M)}
        if len(data) < size:
            raise ValueError(f'expected {size} numbered answers, got {len(data)}')
        return [data.get(str(i + 1), '').strip().strip('"') for i in range(size)]
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # A translation with an unescaped quote breaks the JSON although every segment is there: cut on the keys.
        keys = list(re.finditer(r'"(\d+)"\s*:\s*"', text))
        if [int(k.group(1)) for k in keys] != list(range(1, size + 1)):
            raise
        ends = [k.start() for k in keys[1:]] + [end]
        return [text[k.end():stop].rstrip().rstrip(',').rstrip().removesuffix('"').replace('\\"', '"').strip()
                for k, stop in zip(keys, ends)]
    return [str(data.get(str(i + 1), '')).strip() for i in range(size)]


async def call(client, url, model, task, batch, reasoning, extra=None):
    openrouter = 'openrouter.ai' in url
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt(task, batch)}], 'max_tokens': 16000, 'temperature': 0}
    if openrouter:
        body['usage'] = {'include': True}
        if task['private']:
            body['provider'] = {'data_collection': 'deny'}
    if reasoning:
        body['reasoning'] = {'effort': reasoning}
    body |= extra or {}
    error = 'no attempt'
    for attempt in range(5):
        started = time.monotonic()
        try:
            r = await client.post(url.rstrip('/') + '/chat/completions', json=body)
            data = r.json()
            if 'error' in data:
                raise RuntimeError(str(data['error'])[:300])
            content = data['choices'][0]['message'].get('content') or ''
            usage = data.get('usage') or {}
            return {'outputs': parse(content, len(batch)), 'cost': usage.get('cost', 0),
                    'seconds': round(time.monotonic() - started, 1), 'provider': data.get('provider')}
        except Exception as exc:  # noqa: BLE001 — record and retry; a failed batch is scored as empty
            error = f'{type(exc).__name__}: {exc}'[:300]
            if 'No endpoints' in error or 'data policy' in error.lower():
                break
            await asyncio.sleep(5 * (attempt + 1))
    return {'error': error}


def cache_path(out, model):
    return Path(out) / (re.sub(r'[/:]', '__', model).replace('~', '') + '.json')


async def run_model(client, url, model, tasks, gate, reasoning, out, extra=None):
    path = cache_path(out, model)
    cache = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    async def one(task_name, task, b):
        key = f'{task_name}#{b}'
        if key in cache and 'outputs' in cache[key]:
            return
        async with gate:
            cache[key] = await call(client, url, model, task, task['items'][b:b + BATCH], reasoning, extra)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding='utf-8')
        if 'error' in cache[key]:
            print(f'  {model} {key}: {cache[key]["error"]}', flush=True)

    await asyncio.gather(*(one(n, t, b) for n, t in tasks.items() for b in range(0, len(t['items']), BATCH)))
    print(f'done {model}', flush=True)


def chrf(hyps, refs):
    return sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score


def bootstrap(hyps, refs, rounds=300):
    rng = random.Random(0)
    n = len(hyps)
    scores = sorted(chrf(*zip(*[(hyps[i], refs[i]) for i in (rng.randrange(n) for _ in range(n))])) for _ in range(rounds))
    return round(scores[int(rounds * .025)], 1), round(scores[int(rounds * .975)], 1)


def score(tasks, out):
    rows = []
    for path in sorted(Path(out).glob('*.json')):
        cache = json.loads(path.read_text(encoding='utf-8'))
        row = {'model': path.stem.replace('__', '/'), 'cost': 0.0, 'seconds': 0.0, 'failed': 0}
        drift_pool = []
        for task_name, task in tasks.items():
            # Score only the batches the model completed; a failed batch is an API problem, not a translation.
            hyps, refs = [], []
            for b in range(0, len(task['items']), BATCH):
                entry = cache.get(f'{task_name}#{b}', {})
                if 'outputs' not in entry:
                    if task['kind'] == 'mt' or not path.stem.startswith('local__'):
                        row['failed'] += 1
                    continue
                hyps += entry['outputs']
                refs += [ref for _, ref in task['items'][b:b + BATCH]]
                row['cost'] += entry.get('cost') or 0
                row['seconds'] += entry.get('seconds') or 0
            row[task_name + '_coverage'] = round(100 * len(refs) / len(task['items']))
            if not any(hyps):
                row[task_name] = None
                continue
            if task['kind'] == 'lid':
                row[task_name] = round(100 * sum(h.lower().strip() == r for h, r in zip(hyps, refs)) / len(refs), 1)
            else:
                row[task_name] = round(chrf(hyps, refs), 1)
                row[task_name + '_ci'] = bootstrap(hyps, refs)
                if task['tgt'] == 'run_Latn':
                    drift_pool += [h for h in hyps if h]
        row['rw_drift'] = round(100 * sum(bool(kinyarwanda_markers(h)) for h in drift_pool) / len(drift_pool), 1) if drift_pool else None
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--flores', required=True, help='path to the unpacked flores200_dataset folder')
    ap.add_argument('--lang', default='run_Latn', help='FLORES-200 code of the language under test (default Kirundi)')
    ap.add_argument('--lang-name', default=None, help='how to name the language in prompts')
    ap.add_argument('--pivots', nargs='+', default=['fra_Latn', 'eng_Latn'], help='FLORES-200 codes to translate from and to')
    ap.add_argument('--confusable', default='kin_Latn', help="a close relative to tell apart ('' to skip)")
    ap.add_argument('--domain', default=None, help='optional JSON list of your own sentence pairs')
    ap.add_argument('--n', type=int, default=60, help='FLORES sentences per direction')
    ap.add_argument('--models', nargs='*', default=[])
    ap.add_argument('--base-url', default='https://openrouter.ai/api/v1', help='any OpenAI-compatible endpoint')
    ap.add_argument('--api-key-env', default='OPENROUTER_API_KEY')
    ap.add_argument('--reasoning', default=None, help='OpenRouter reasoning effort, e.g. low')
    ap.add_argument('--extra-body', default=None, help='JSON merged into every request, e.g. \'{"reasoning_effort": "none"}\' for a local thinking model')
    ap.add_argument('--concurrency', type=int, default=24)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'cache'))
    ap.add_argument('--report', action='store_true', help='only rescore what is cached')
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    names = {args.lang: args.lang_name} if args.lang_name else None
    tasks = load_tasks(args.flores, args.n, args.lang, tuple(args.pivots), args.confusable, args.domain, names)
    if args.models and not args.report:
        headers = {'Authorization': f"Bearer {os.environ.get(args.api_key_env, 'none')}", 'X-Title': 'kirundi-eval-kit'}

        extra = json.loads(args.extra_body) if args.extra_body else None

        async def go():
            gate = asyncio.Semaphore(args.concurrency)
            async with httpx.AsyncClient(headers=headers, timeout=1800) as client:
                await asyncio.gather(*(run_model(client, args.base_url, m, tasks, gate, args.reasoning, args.out, extra) for m in args.models))
        asyncio.run(go())
    rows = score(tasks, args.out)
    (Path(args.out).parent / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    cols = [c for c in tasks] + ['rw_drift']
    mt = [c for c, t in tasks.items() if t['kind'] == 'mt']
    rows.sort(key=lambda r: -sum((r.get(c) or 0) for c in mt))
    print('model'.ljust(40) + ''.join(c.replace('flores_', 'F ').replace('srhr_', 'S ').replace('domain_', 'D ').rjust(10) for c in cols) + '    cost$  failed')
    for r in rows:
        print(r['model'][:40].ljust(40) + ''.join(('-' if r.get(c) is None else str(r[c])).rjust(10) for c in cols)
              + f"{r['cost']:9.3f}{r['failed']:8d}")


if __name__ == '__main__':
    main()
