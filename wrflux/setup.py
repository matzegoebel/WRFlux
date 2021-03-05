"""A setuptools based setup module adapted from PyPa's sample project.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

We reproduce their license terms here:

Copyright (c) 2016 The Python Packaging Authority (PyPA)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this file (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from pathlib import Path
here = Path(__file__)

# Get the long description from the README file
with open(here.parents[1] / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='wrflux',
    version='1.0.0',
    description='With this package time-averaged tendencies of the WRF model can be retrieved '
    'in the Cartesian coordinate system and decomposed into mean and resolved turbulent components.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/matzegoebel/wrflux',
    author='Matthias Göbel',
    author_email='matthias-goebel@uibk.ac.at',
    keywords='WRF',
    packages=find_packages(),
    python_requires=">=3",
    install_requires=['xarray', 'matplotlib', 'netcdf4', 'bottleneck'],
    extras_require={
        'test': ['pytest', 'run_wrf'],
    },
)
