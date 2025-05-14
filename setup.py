from setuptools import setup, find_packages


__version__ = '0.0.1'

setup(
    name='CoSense3D',
    version=__version__,
    author='Yunshuang Yuan',
    author_email='yunshuang.yuan@ikg.uni-hannover.de',
    url='-',
    license='MIT',
    packages=find_packages(include=['cosense3d']),
    zip_safe=False,
)
