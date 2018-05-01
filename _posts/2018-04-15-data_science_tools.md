---
layout: post
title:  Go-to references for key components of data science
---
These are the sources I used to implement best practices in data science.

### Data storage format options
What's an efficient way to store derived data, features, parameters when creating models?

#### Best practices and format comparisions
- <https://stackoverflow.com/questions/14262433/large-data-work-flows-using-pandas?noredirect=1&lq=1>  
- <https://www.slideshare.net/wesm/python-data-wrangling-preparing-for-the-future>  

#### Feather
- <http://blog.cloudera.com/blog/2016/03/feather-a-fast-on-disk-format-for-data-frames-for-r-and-python-powered-by-apache-arrow/>  

#### Pandas and mongodb
- <http://blog.cloudera.com/blog/2016/03/feather-a-fast-on-disk-format-for-data-frames-for-r-and-python-powered-by-apache-arrow/>  

Use of AWS S3 as mainstay storage

### Workflow pipepline
- luigi (<http://luigi.readthedocs.io/en/stable/workflows.html>)

## Coding best practices
- LowClass Python: Style guide for data scientists (<http://columbia-applied-data-science.github.io/pages/lowclass-python-style-guide.html>)

### debugging, testing, refactoring of code
- pytest (<https://docs.pytest.org/en/latest/>)
- `pdb` [cheatsheet]('resources/pdb-cheatsheet.pdf')

### Documentation  
- Sphinx  <https://thomas-cokelaer.info/tutorials/sphinx/index.html>  
  - Google examples <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>  
- PEP257:Python docstring example <http://blog.dolphm.com/pep257-good-python-docstrings-by-example/>
-
