import setuptools

setuptools.setup(
    name="dlp", 
    version="1.0.0",
    description='Vehicle parking dataset',
    author='Xu Shen, Michelle Pan',
    author_email='xu_shen@berkeley.edu',
    packages=['dlp'],
    install_requires=[
    "matplotlib==3.4.1",
    "numpy==1.20.2",
    "pandas==1.2.4",
    "tqdm==4.60.0",
    "pyyaml==5.4.1",
    "ipykernel==6.0.1",
    "jupyterlab==3.0.16"
    ]
)