"""Build/verify the self-contained final bundle, excluding raw Parquet/runtime caches."""
import argparse
import json
import zipfile
from pathlib import Path
from .data import ROOT,dump,digest

MANIFEST=ROOT/'reports/final_bundle_manifest.json'


def included(path):
    relative=path.relative_to(ROOT)
    if not path.is_file() or path==MANIFEST: return False
    if any(p in {'.venv','__pycache__','.pytest_cache','.git'} for p in relative.parts): return False
    if path.name=='.DS_Store' or path.suffix in {'.writing','.pyc','.parquet','.download'}: return False
    return True


def verify():
    meta=json.loads(MANIFEST.read_text()); errors=[]
    for name,sha in meta['files'].items():
        p=ROOT/name
        if not p.is_file() or digest(p)!=sha: errors.append(name)
    if errors: raise SystemExit('Checksum mismatch: '+', '.join(errors))
    print(f"Verified {len(meta['files'])} final files")


def build():
    assert json.loads((ROOT/'reports/refined/verification.json').read_text())['passed']
    paths=sorted(p for p in ROOT.rglob('*') if included(p))
    dump(MANIFEST,{'format_version':1,'excluded':'Raw Parquet, runtime environments/caches; all prepared inputs included',
        'files':{str(p.relative_to(ROOT)):digest(p) for p in paths}})
    verify()
    target=ROOT.parent/'deliverables/yambda_rl_final.zip'; target.parent.mkdir(exist_ok=True)
    temporary=target.with_suffix('.writing')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in paths+[MANIFEST]: archive.write(p,'rl_recsys/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        names=archive.namelist(); assert len(names)==len(set(names))
        assert json.loads(archive.read('rl_recsys/reports/final_bundle_manifest.json'))['files']==json.loads(MANIFEST.read_text())['files']
    temporary.replace(target)
    checksum=digest(target)
    target.with_suffix('.sha256').write_text(f'{checksum}  {target.name}\n')
    print(f'{target}\n{target.stat().st_size} bytes\nSHA256 {checksum}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--verify',action='store_true')
    verify() if parser.parse_args().verify else build()
