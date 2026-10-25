"""Optional verified parallel range downloader for slow/truncating CDN streams."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
import urllib.request

import pyarrow.parquet as pq

from .data import ROOT, digest, dump


def main():
    source = ROOT/'reports/source_manifest.json'
    if not source.exists():
        raise FileNotFoundError('A pinned source_manifest.json is required')
    meta = json.loads(source.read_text())
    directory = ROOT/'data'
    directory.mkdir(exist_ok=True)
    target = directory/'listens.parquet'
    if target.exists() and digest(target) == meta['sha256']:
        dump(directory/'manifest.json', meta)
        print('Verified existing dataset')
        return
    chunks = directory/'chunks'
    chunks.mkdir(exist_ok=True)
    size = meta['bytes']; block_size = 2**20
    total = (size+block_size-1)//block_size

    def get(i):
        start = i*block_size; end = min(size,(i+1)*block_size)-1
        path = chunks/f'{i:04d}.part'
        if path.exists() and path.stat().st_size == end-start+1:
            return
        for retry in range(4):
            try:
                req = urllib.request.Request(meta['url']+f'?part={i}',headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=90) as response:
                    if response.status != 206:
                        raise ValueError('Server did not honor Range request')
                    content = response.read()
                if len(content) != end-start+1:
                    raise ValueError(f'Incomplete chunk {i}')
                path.write_bytes(content)
                return
            except Exception:
                if retry==3: raise
                time.sleep(retry+1)

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(get,i) for i in range(total)]
        for n, future in enumerate(as_completed(futures),1):
            future.result()
            if n%32==0: print(f'{n}/{total}',flush=True)
    temporary = target.with_suffix('.assembled')
    with temporary.open('wb') as out:
        for i in range(total):
            out.write((chunks/f'{i:04d}.part').read_bytes())
    if digest(temporary) != meta['sha256']:
        raise ValueError('Assembled dataset SHA256 mismatch')
    pq.read_metadata(temporary)
    temporary.replace(target)
    dump(directory/'manifest.json',meta)
    for chunk in chunks.iterdir(): chunk.unlink()
    chunks.rmdir()
    print('Dataset verified',flush=True)


if __name__=='__main__': main()
