from setuptools import *

with open('README.md', 'r', encoding='utf-8') as fh:
	long_description= fh.read()


setup(
    name = "extremelearning",
    version = "0.1.3",
    author = "Kyle Hall",
    author_email = "kjhall@iri.columbia.edu",
    description = ("A selection of ELM-based Machine Learning Approaches"),
    license = "MIT",
    keywords = "ELM Extreme Learning Machine Machine-Learning AI Fast AI",
    url = "https://github.com/kjhall01/extremelearning/",
    packages=['extremelearning', 'extremelearning.probabilistic', 'extremelearning.deterministic'],
	package_dir={'extremelearning': 'src', 'extremelearning.probabilistic':'src/probabilistic', 'extremelearning.deterministic':'src/deterministic'},
	python_requires=">=3.4",
    long_description=long_description,
	long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
    ],
)
