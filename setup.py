from distutils.core import setup

with open("./README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='fast_torch',
    packages=['fast_torch'],
    version='0.2.2',
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
        'pytorch',
        'numpy',
        'matplotlib',
        'mpl_toolkits',
        'itertools'
    ]
)
