from distutils.core import setup
import setuptools

with open("./README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='fast_torch',
    packages=setuptools.find_packages(),
    version='1.1.1',
    license='MIT',
    description='Library that implements boiler plate code in PyTorch for training, testing and plotting your model',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Sakir Ozturk',
    author_email='ozturk213@hotmail.fr',
    url='https://github.com/YnsOzt/FastTorch',
    keywords=['PyTorch', 'boiler_plate', 'Train', 'Test', 'Plot'],
    install_requires=[
        'sklearn',
        'torch',
        'torchvision',
        'tqdm',
        'numpy',
        'matplotlib'
    ],
)
