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


def pyg_cora(path='./datasets'):
    '''
    https://people.cs.umass.edu/~mccallum/data/
    '''
    path = Path(path)
    if not (path).exists():
        path.mkdir(parents=True)
    if not (path/'PyG-Cora').exists():
        url = 'https://github.com/hitlic/bijou/raw/master/datasets/PyG-Cora.zip'
        download(url, path, 'PyG-Cora.zip')
    return path/'PyG-Cora'


def pyg_yoochoose_10k(path='./datasets'):
    '''
    https://2015.recsyschallenge.com/challenge.html
    '''
    path = Path(path)
    if not (path).exists():
        path.mkdir(parents=True)
    if not (path/'PyG-yoochoose_10k').exists():
        url = 'https://github.com/hitlic/bijou/raw/master/datasets/PyG-yoochoose_10k.zip'
        download(url, path, 'PyG-yoochoose_10k.zip')
    return path/'PyG-yoochoose_10k'


def dgl_cora(path='./datasets'):
    '''
    https://people.cs.umass.edu/~mccallum/data/
    '''
    path = Path(path)
    if not (path).exists():
        path.mkdir(parents=True)
    if not (path/'DGL-Cora').exists():
        url = 'https://github.com/hitlic/bijou/raw/master/datasets/DGL-Cora.zip'
        download(url, path, 'DGL-Cora.zip')
    return path/'DGL-Cora'
