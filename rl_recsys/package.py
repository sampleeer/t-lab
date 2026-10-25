"""Build a self-contained code/report/checkpoint bundle, excluding raw data."""
import json
from pathlib import Path
import zipfile

from .data import ROOT, digest, dump


def main():
    files=[]
    for path in ROOT.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(ROOT)
        if any(p in ['data','__pycache__','.venv'] for p in relative.parts):continue
        if path.suffix=='.log' or path.name=='.DS_Store':continue
        # Include evaluation inputs and all fitted weights/diagnostics; training
        # arrays are recreated by prepare from the pinned source dataset.
        if relative.as_posix() in ['artifacts/train.npz','artifacts/validation.npz']:continue
        files.append(path)
    target=ROOT.parent/'deliverables/rl_recsys_complete.zip'
    target.parent.mkdir(exist_ok=True)
    dump(ROOT/'reports/bundle_manifest.json',{'files':{
        str(path.relative_to(ROOT)):digest(path) for path in sorted(files)
        if path.name!='bundle_manifest.json'}})
    files=[p for p in files if p.name!='bundle_manifest.json']+[ROOT/'reports/bundle_manifest.json']
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files):z.write(path,'rl_recsys/'+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
        if bad:raise ValueError(f'ZIP CRC failed: {bad}')
    print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,
                      'sha256':digest(target),'files':len(files)},indent=2))


if __name__=='__main__':main()
