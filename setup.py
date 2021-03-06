import setuptools

setuptools.setup(
    name="dlp", 
    version="1.5.0",
    description='Vehicle parking dataset',
    author='Xu Shen, Michelle Pan',
    author_email='xu_shen@berkeley.edu',
    packages=['dlp'],
    install_requires=[
        "pillow>=9.0.0",
        "matplotlib>=3.4.1",
        "numpy>=1.21.0",
        "pandas>=1.2.4",
        "tqdm>=4.60.0",
        "pyyaml>=5.4.1",
        "jupyterlab>=3.0.17"
    ]
)