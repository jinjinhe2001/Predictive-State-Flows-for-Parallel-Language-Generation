"""bd3lms/mdlm `sample_eval` CSVs -> JSONL ('text' field) for `scripts.eval_gen --texts`.

The official writer appends rows WITHOUT a header (the file is created by the 'a' open before
the exists-check), in the fixed order: gen_ppl, gen_nfes, gen_entropy, gen_lengths, samples
(a JSON list holding one string, which starts with the BOS text "<|endoftext|>"), seed.

Two files are written:
  <out>            document-faithful: text cut at the first generated <|endoftext|> (what ELF's
                   `mask_after_eos` does, and what our own fixed-canvas docs are) -- BD3-LM emits
                   EOS mid-canvas and starts a new document after it (~737 gpt2 tokens kept on avg).
  <out>_fulleos    their protocol: the whole 1024-token canvas with the literal "<|endoftext|>"
                   markers kept, so the judge sees the EOS token as an in-sequence separator exactly
                   as their gen-ppl / entropy code does (multi-document rows -> higher unigram entropy).
  python csv_to_jsonl.py sample_logs/x/gpu*.csv --out sample_logs/x/all.jsonl
"""
import argparse
import ast
import csv
import json
import os
import sys

csv.field_size_limit(sys.maxsize)
BOS = "<|endoftext|>"


def parse_sample(cell: str) -> str:
    try:
        v = json.loads(cell)
    except Exception:
        try:
            v = ast.literal_eval(cell)
        except Exception:
            v = cell
    s = v[0] if isinstance(v, (list, tuple)) and v else str(v)
    while s.startswith(BOS):
        s = s[len(BOS):]
    return s


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("csvs", nargs="+")
p.add_argument("--out", required=True)
a = p.parse_args()
stem, ext = os.path.splitext(a.out)
n = n_full = 0
with open(a.out, "w", encoding="utf-8") as out, open(f"{stem}_fulleos{ext}", "w", encoding="utf-8") as out_full:
    for path in a.csvs:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 5 or row[0] == "gen_ppl":
                    continue
                full = parse_sample(row[4])
                meta = {"their_gen_ppl": row[0], "their_nfes": row[1], "their_entropy": row[2],
                        "their_len": row[3]}
                if full.strip():
                    out_full.write(json.dumps({"text": full.strip(), **meta}, ensure_ascii=False) + "\n")
                    n_full += 1
                doc = full.split(BOS)[0].strip()
                if doc:
                    out.write(json.dumps({"text": doc, **meta}, ensure_ascii=False) + "\n")
                    n += 1
print(f"wrote {n} first-document samples -> {a.out}; {n_full} full-canvas samples -> {stem}_fulleos{ext}")
