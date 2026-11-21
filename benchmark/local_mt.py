"""Small open translation models run on your own machine (nothing leaves it): NLLB-200 and MADLAD-400.

  uv run --with torch --with transformers --with sentencepiece --with protobuf --with httpx --with sacrebleu \
      python benchmark/local_mt.py --flores data/flores200_dataset --model facebook/nllb-200-distilled-1.3B
  ... --model google/madlad400-3b-mt

Takes the same language options as benchmark/translation.py and writes the same cache format, so
`benchmark/translation.py --report` ranks these models with the API models.

Licences differ: NLLB-200 is CC BY-NC 4.0 (non-commercial); MADLAD-400 is Apache 2.0.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path[:0] = [str(Path(__file__).resolve().parent)]
from translation import BATCH, ROOT, load_tasks  # noqa: E402

# MADLAD-400 uses its own language tags (<2rn>); add yours here when it differs from the FLORES prefix.
MADLAD = {'run_Latn': 'rn', 'fra_Latn': 'fr', 'eng_Latn': 'en', 'kin_Latn': 'rw', 'swh_Latn': 'sw'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--flores', required=True)
    ap.add_argument('--lang', default='run_Latn')
    ap.add_argument('--pivots', nargs='+', default=['fra_Latn', 'eng_Latn'])
    ap.add_argument('--domain', default=None)
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--model', default='facebook/nllb-200-distilled-1.3B')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'cache'))
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    madlad = 'madlad' in args.model.lower()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.float16 if args.device != 'cpu' else None).to(args.device)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / ('local__' + args.model.split('/')[-1] + '.json')
    cache = {}
    for name, task in load_tasks(args.flores, args.n, args.lang, tuple(args.pivots), '', args.domain).items():
        if task['kind'] != 'mt':
            continue
        extra = {}
        if not madlad:
            tok.src_lang = task['src']
            extra['forced_bos_token_id'] = tok.convert_tokens_to_ids(task['tgt'])
        for b in range(0, len(task['items']), BATCH):
            texts = [src for src, _ in task['items'][b:b + BATCH]]
            if madlad:
                texts = [f"<2{MADLAD.get(task['tgt'], task['tgt'].split('_')[0])}> {t}" for t in texts]
            started = time.monotonic()
            enc = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(args.device)
            out = model.generate(**enc, max_new_tokens=512, num_beams=4, **extra)
            cache[f'{name}#{b}'] = {'outputs': tok.batch_decode(out, skip_special_tokens=True), 'cost': 0,
                                   'seconds': round(time.monotonic() - started, 1), 'provider': 'local'}
            path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding='utf-8')
        print('done', name, flush=True)


if __name__ == '__main__':
    main()
