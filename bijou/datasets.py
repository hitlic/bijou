import torch
import pickle
import gzip
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
from math import ceil

def download(url, path, fname):
    print(f"Downloading from {url}")
    resp = requests.get(url=url, stream=True)
    total = ceil(int(resp.headers['Content-Length'])/1024)
    with open(path/fname, "wb") as f:
        for data in tqdm(iterable=resp.iter_content(1024), total=total, unit='k'):
            f.write(data)
    zf = zipfile.ZipFile(path/fname)
    zf.extractall(path=path)
    (path/fname).unlink()


def mnist(path='./datasets'):
    path = Path(path)
    if not (path).exists():
        path.mkdir(parents=True)
    if not (path/'mnist').exists():
        url = 'https://github.com/hitlic/bijou/raw/master/datasets/mnist.zip'
        download(url, path, 'mnist.zip')

    path = path/'mnist'/'mnist.pkl.gz'
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding='latin-1')
    return map(torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test))


def cora(path='./datasets'):
    '''
    https://people.cs.umass.edu/~mccallum/data/
    '''
    path = Path(path)
    if not (path).exists():
        path.mkdir(parents=True)
    if not (path/'Cora').exists():
        url = 'https://github.com/hitlic/bijou/raw/master/datasets/Cora.zip'
        download(url, path, 'Cora.zip')
    return path/'Cora'


def yoochoose_10k(path='./datasets'):
    '''
    https://2015.recsyschallenge.com/challenge.html
    '''
    path = Path(path)
    if not (path).exists():
        path.mkdir(parents=True)
    if not (path/'yoochoose_10k').exists():
        url = 'https://github.com/hitlic/bijou/raw/master/datasets/yoochoose_10k.zip'
        download(url, path, 'yoochoose_10k.zip')
    return path/'yoochoose_10k'
