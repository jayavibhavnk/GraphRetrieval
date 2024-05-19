from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent

setup(
    name='GraphRetrieval',
    version='0.1.1',
    description='Graph retrieval',
    long_description=(this_directory / "README.md").read_text(),
    author='JVNK',
    author_email='jaya11vibhav@gmail.com',
    url='https://github.com/jayavibhavnk/GraphRetrieval',
    packages=find_packages(),
    install_requires=[
      'langchain_openai',
      'langchain',
      'sentence_transformers',
      'langchain-community', 
      'langchain-openai',
      'langchain-experimental',
      'neo4j',
      'wikipedia',
      'tiktoken',
      'pypdf2'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha'
    ],
)
