from setuptools import setup, find_packages

setup(
    name='cuna',
    version='0.1.0',
    description='CUNA: Cytosine Uracil Neural Algorithm for detecting deamination from nanopore sequencing data',
    author='Iris del Bosque',
    author_email='irisdbf101@gmail.com',
    url='https://github.com/iris1901/CUNA',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'cuna = cuna.CUNA:main', 
        ],
    },
    install_requires=[
        'numpy',
        'pysam',
        'torch',
        'h5py',
        'tqdm',
        'numba',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'pod5',
        'scipy'
    ],
    python_requires='>=3.8',
    include_package_data=True,
)