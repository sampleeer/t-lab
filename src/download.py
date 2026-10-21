"""Pinned official archive; deliberately avoid the truncated standalone ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

URL = ('https://raw.githubusercontent.com/irsafilo/KION_DATASET/'
       'f69775be31fa5779907cf0a92ddedb70037fb5ae/data_original.zip')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/raw')
    args = parser.parse_args()
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / 'data_original.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.download')
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
        for name in ['interactions.csv', 'users.csv', 'items.csv']:
            # Copy named members only, never extract untrusted paths.
            member = 'data_original/' + name
            (directory / name).write_bytes(z.read(member))
    manifest = {'url': URL, 'sha256': {
        p.name: sha256(p) for p in [archive] +
        [directory / n for n in ['interactions.csv', 'users.csv', 'items.csv']]}}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
