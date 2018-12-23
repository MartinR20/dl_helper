from setuptools import setup

setup(name='dl_helper',
      version='0.1',
      description='helper for colab project',
      url='https://github.com/MartinR20/dl_helper',
      author='MartinR20',
      author_email='example.email@email.com',
      packages=['dl_helper'],
      install_requires=[
          'torch',
          'sklearn',
          'pandas',
          'numpy',
          'psutil'
      ],
      zip_safe=False)
