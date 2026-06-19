import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

LABEL_DEFAULTS = {
    "safe": 0,
    "prompt_injection": 1,
    "jailbreak": 2,
    "gcg": 3,
    "crescendo": 4,
    "pair": 5,
    "roleplay": 6,
    "encoding_attack": 7,
    "pii": 8,
}

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def infer_label(folder_name: str) -> int:
    name = folder_name.lower().replace("-", "_").replace(" ", "_")
    if name.isdigit():
        return int(name)
    return LABEL_DEFAULTS.get(name, 0)

def normalize_row(row: dict[str, Any], default_label: int, source: str, file_name: str):
    text = row.get("text") or row.get("prompt") or row.get("content") or ""
    if not text.strip():
        return None
    label = int(row.get("label", default_label))
    session_id = row.get("session_id", "")
    turn_index = int(row.get("turn_index", 0))
    role = row.get("role", "user")
    return {
        "text": text.strip(),
        "label": label,
        "source": row.get("source", source),
        "file": file_name,
        "session_id": session_id,
        "turn_index": turn_index,
        "role": role,
    }

def load_rows_from_file(path: Path, default_label: int, source: str):
    rows = []
    if path.suffix.lower() == ".jsonl":
        iterable = read_jsonl(path)
    elif path.suffix.lower() == ".csv":
        iterable = read_csv(path)
    else:
        return rows

    for row in iterable:
        nr = normalize_row(row, default_label, source, path.name)
        if nr:
            rows.append(nr)
    return rows

def build_conversations(turn_rows):
    sessions = defaultdict(list)
    for r in turn_rows:
        sid = r.get("session_id")
        if sid:
            sessions[sid].append(r)

    conv_rows = []
    for sid, turns in sessions.items():
        turns = sorted(turns, key=lambda x: x.get("turn_index", 0))
        conversation_text = "\n".join(
            f'{t.get("role","user")}: {t["text"]}' for t in turns
        )
        label = max(int(t["label"]) for t in turns)
        conv_rows.append({
            "session_id": sid,
            "label": label,
            "conversation_text": conversation_text,
            "num_turns": len(turns),
            "source": turns[0].get("source", "unknown"),
        })
    return conv_rows

def split(rows, train_ratio=0.90, val_ratio=0.05):
    rows = sorted(rows, key=lambda x: (str(x.get("source", "")), str(x.get("text", x.get("conversation_text", "")))[:64]))
    n = len(rows)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Folder containing label subfolders or dataset folders")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-ratio", type=float, default=0.90)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    turn_rows = []

    for folder in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        default_label = infer_label(folder.name)
        source = folder.name

        for f in folder.rglob("*"):
            if f.is_file():
                turn_rows.extend(load_rows_from_file(f, default_label, source))

    turn_rows = [r for r in turn_rows if r.get("text")]
    conversation_rows = build_conversations(turn_rows)

    turn_train, turn_val, turn_test = split(turn_rows, args.train_ratio, args.val_ratio)
    conv_train, conv_val, conv_test = split(conversation_rows, args.train_ratio, args.val_ratio)

    out = Path(args.out_dir)
    write_jsonl(out / "turn_train.jsonl", turn_train)
    write_jsonl(out / "turn_val.jsonl", turn_val)
    write_jsonl(out / "turn_test.jsonl", turn_test)

    write_jsonl(out / "conversation_train.jsonl", conv_train)
    write_jsonl(out / "conversation_val.jsonl", conv_val)
    write_jsonl(out / "conversation_test.jsonl", conv_test)

    label_map = {name: idx for name, idx in LABEL_DEFAULTS.items()}
    with (out / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"turn_train={len(turn_train)} turn_val={len(turn_val)} turn_test={len(turn_test)}")
    print(f"conv_train={len(conv_train)} conv_val={len(conv_val)} conv_test={len(conv_test)}")

if __name__ == "__main__":
    main()