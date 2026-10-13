#!/usr/bin/env python3
"""Copy a local ChunkRL checkpoint into the shared container model workspace."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path, help='chunk_rl checkout or robomimic_multitask_v5 checkpoint directory')
parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1]/'workspace/model/chunkrl/robomimic_multitask_v5')
args=parser.parse_args()
source=args.source.expanduser().resolve()
if (source/'ckpt/robomimic_multitask_v5').is_dir(): source=source/'ckpt/robomimic_multitask_v5'
required=[source/'best_eval.pt',source/'config.json']
for file in required:
    if not file.is_file(): parser.error(f'Missing {file}')
args.destination.mkdir(parents=True,exist_ok=True)
for file in required:
    destination=args.destination/file.name
    if destination.resolve()!=file.resolve(): shutil.copy2(file,destination)
print(json.dumps({'destination':str(args.destination.resolve()),'best_eval_sha256':hashlib.sha256((args.destination/'best_eval.pt').read_bytes()).hexdigest()},indent=2))
