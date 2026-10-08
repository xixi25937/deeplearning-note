import hashlib
import os
import requests

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce'
)
DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
)

def download(name, cache_dir=os.path.join('.', 'data')):
    """Download a file inserted into DATA_HUB, return the local filename."""
    if name not in DATA_HUB:
        raise KeyError(f'{name} does not exist in DATA_HUB')

    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])

    def valid_file(path):
        sha1 = hashlib.sha1()
        with open(path, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        return sha1.hexdigest() == sha1_hash

    if os.path.exists(fname):
        if valid_file(fname):
            return fname  # Hit cache

        os.remove(fname)

    print(f'Downloading {fname} from {url}...')
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with open(fname, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    if not valid_file(fname):
        os.remove(fname)
        raise RuntimeError(f'Hash mismatch for downloaded file: {fname}')

    return fname


if __name__ == '__main__':
    for dataset_name in ('kaggle_house_train', 'kaggle_house_test'):
        print(download(dataset_name))
