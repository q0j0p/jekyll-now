---
layout: post
title:  Go-to references for data science tools and concepts
---
These are the sources I use to implement best practices in data science.

### Data storage format options
What's an efficient way to store derived data, features, parameters when creating models?

#### Best practices and format comparisions
- <https://stackoverflow.com/questions/14262433/large-data-work-flows-using-pandas?noredirect=1&lq=1>  
- <https://www.slideshare.net/wesm/python-data-wrangling-preparing-for-the-future>  

#### Feather
- <http://blog.cloudera.com/blog/2016/03/feather-a-fast-on-disk-format-for-data-frames-for-r-and-python-powered-by-apache-arrow/>  

#### Pandas and mongodb
- <http://blog.cloudera.com/blog/2016/03/feather-a-fast-on-disk-format-for-data-frames-for-r-and-python-powered-by-apache-arrow/>  

#### HDF5
- [The python HDF5 ecosystem](https://www.hdfgroup.org/2015/09/python-hdf5-a-vision/)
- HDF5 take 2 - h5py & PyTables | SciPy 2017 Tutorial | Tom Kooij (https://www.youtube.com/watch?v=ofLFhQ9yxCw)


#### Organizing files in HDFS (<https://www.linkedin.com/learning/hadoop-for-data-science-tips-tricks-techniques/organize-files-in-hdfs>)

#### AWS

### Workflow pipepline
- luigi (<http://luigi.readthedocs.io/en/stable/workflows.html>)

### Coding best practices
- LowClass Python: Style guide for data scientists (<http://columbia-applied-data-science.github.io/pages/lowclass-python-style-guide.html>)
- **[Intermediate and advanced software carpentry](https://intermediate-and-advanced-software-carpentry.readthedocs.io/en/latest/structuring-python.html)**  

### Debugging, testing, refactoring of code
- pytest (<https://docs.pytest.org/en/latest/>)
- `pdb` (<https://dgkim5360.github.io/blog/python/2017/10/a-cheatsheet-for-python-pdb-debugger/>)
  - [cheatsheet]('resources/pdb-cheatsheet.pdf')

### Virtual environments, requirements, containers
- conda env  
- pip freeze (<https://pip.pypa.io/en/stable/reference/pip_freeze/>)  

### Code Documentation  
- Sphinx  <https://thomas-cokelaer.info/tutorials/sphinx/index.html>  
  - Google examples <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>  
- [PEP257](https://www.python.org/dev/peps/pep-0257/):Python docstring example <http://blog.dolphm.com/pep257-good-python-docstrings-by-example/>
- doctest (<https://docs.python.org/3/library/doctest.html>)
