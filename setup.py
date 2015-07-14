from setuptools import setup

setup(
    name='im2ascii',
    version='1.0',
    author='Asher Bender',
    author_email='a.bender.dev@gmail.com',
    description=('Convert images to ASCII art.'),
    scripts=['im2ascii'],
    install_requires=[
        'numpy',           # Tested on 1.9.2
        'scikit-image',    # Tested on 0.11.3
    ]
)
