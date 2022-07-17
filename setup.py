import pathlib
import setuptools

NAME = 'resampler'

def get_version(package=None):
  if package is None:
    package, = setuptools.find_packages()
  path = pathlib.Path(__file__).resolve().parent / package / '__init__.py'
  for line in path.read_text().splitlines():
    if line.startswith("__version__ = '"):
      _, version, _ = line.split("'")
      return version
  raise RuntimeError(f'Unable to find version string in {path}.')


def get_long_description():
  return pathlib.Path('README.md').read_text()


def get_requirements():
  path = pathlib.Path(NAME) / 'requirements.txt'
  return [line.strip() for line in path.read_text().splitlines()]


setuptools.setup(
  name=NAME,
  version=get_version(),
  author='Hugues Hoppe',
  author_email='hhoppe@gmail.com',
  description='Fast differentiable resizing and warping of arbitrary grids',
  long_description=get_long_description(),
  long_description_content_type='text/markdown',
  url=f'https://github.com/hhoppe/{NAME}.git',
  packages=setuptools.find_packages(),
  package_data={
    package: ['py.typed', 'requirements.txt'] for package in setuptools.find_packages()
  },
  classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Image Processing',
  ],
  python_requires='>=3.7',
  install_requires=get_requirements(),
)
