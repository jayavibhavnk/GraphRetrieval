from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='GraphRetrieval',
    version='0.1.1',
    description='Graph retrieval',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
