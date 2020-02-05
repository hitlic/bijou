from bijou import __vsersion__
import setuptools

desc = 'A fastai-like framework for training, tuning and probing pytorch models, which is compatible with pytorch_geometric.'

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="bijou",
    version=__vsersion__,
    author="hitlic",
    author_email="liuchen.lic@gmail.com",
    license='MIT',
    description=desc,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/hitlic/bijou",
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='',
    packages=setuptools.find_packages(exclude=['examples', 'datasets',]),
    py_modules=[],
    install_requires=['torch > 1.1', 'tqdm > 4.40', 'matplotlib > 3.1', 'networkx >= 2.3', 'requests > 2.20.0'],
    python_requires='>=3.5'
)
