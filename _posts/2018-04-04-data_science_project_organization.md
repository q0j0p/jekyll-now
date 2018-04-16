---
layout: post 
title: Data science project organization 
---
What's the best way to organize data science projects?  I have implemented the following.  

# References
### Repository structure
```
.
├── data           : processed data, versioned
├── notes          : communicable notes for the project
│   └── blog       : linked to _posts
├── raw            : files in this directory are shared as necessary
│   ├── data       : original data, treated as immutable
│   ├── notes      : taken throughout the course of the project, ordered by date
│   └── resources  : all resources pertaining to the project
├── results        : output of analysis or models
└── src            : all code for the project
```

 <https://medium.com/outlier-bio-blog/a-quick-guide-to-organizing-data-science-projects-updated-for-2016-4cbb1e6dac71>
- <https://drivendata.github.io/cookiecutter-data-science/#directory-structure>  
- <http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000424>
- <https://drivendata.github.io/cookiecutter-data-science/#directory-structure>  
  - analysis as DAG

### Organizing data
- Raw data must be immutable
- Processed data should be versioned  
- Refs:
 - <https://www.data.cam.ac.uk/data-management-guide/organising-your-data>  

### Data storage format options
What's an efficient way to store derived data, features, parameters?

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

### unit testing
- pytest (<https://docs.pytest.org/en/latest/>)
### Documentation  
- Sphinx  <https://thomas-cokelaer.info/tutorials/sphinx/index.html>  
  - Google examples <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>  
- PEP257:Python docstring example <http://blog.dolphm.com/pep257-good-python-docstrings-by-example/>
-
