import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen


WEIGHTS_URL = 'https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download'
TARGET_NAME = 'rd64-uni.pth'


def download_and_extract(force=False):
    repo_root = Path(__file__).resolve().parents[1]
    weights_dir = repo_root / 'weights'
    weights_dir.mkdir(exist_ok=True)
    target_path = weights_dir / TARGET_NAME

    if target_path.exists() and not force:
        print(f'weights already present: {target_path}')
        return target_path

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / 'weights.zip'
        with urlopen(WEIGHTS_URL) as response, archive_path.open('wb') as output_file:
            shutil.copyfileobj(response, output_file)

        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if TARGET_NAME not in {Path(name).name for name in names}:
                raise FileNotFoundError(f'{TARGET_NAME} not found in downloaded archive')

            for name in names:
                if Path(name).name == TARGET_NAME:
                    with archive.open(name) as source, target_path.open('wb') as target_file:
                        shutil.copyfileobj(source, target_file)
                    break

    print(f'downloaded {TARGET_NAME} to {target_path}')
    return target_path


def main():
    parser = argparse.ArgumentParser(description='Download the official CLIPSeg rd64-uni weights.')
    parser.add_argument('--force', action='store_true', help='overwrite an existing local weight file')
    args = parser.parse_args()
    download_and_extract(force=args.force)


if __name__ == '__main__':
    main()
