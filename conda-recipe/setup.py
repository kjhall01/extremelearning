from setuptools import *

with open('README.md', 'r', encoding='utf-8') as fh:
	long_description= fh.read()


setup(
    name = "pyridl",
    version = "0.2.1",
    author = "Kyle Hall",
    author_email = "hallkjc01@gmail.com",
    description = ("Stopgap IRIDL API"),
    license = "MIT",
    keywords = "IRI DATA LIBRARY",
    url = "https://bitbucket.org/hallkjc01/pyridl/src/master/",
    packages=['pyridl', 'pyridl.datalibrary_source'],
	package_dir={'pyridl': 'src', 'pyridl.datalibrary_source':'src/datalibrary_source'},
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
