from bijou import __vsersion__
import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="bijou",
    version=__vsersion__,
    author="hitlic",
    author_email="liuchen.lic@gmail.com",
    license='MIT',
    description="",
    long_description=long_description,
    url="",
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='',
    packages=setuptools.find_packages(),
    py_modules=[], # any single-file Python modules that aren’t part of a package
    install_requires=['torch > 1.1', 'tqdm > 4.40', 'matplotlib > 3.1', 'networkx >= 2.3'],
    python_requires='>=3.5'
)
