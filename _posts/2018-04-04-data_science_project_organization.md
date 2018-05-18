---
layout: post
title: Data science project organization
---
What's the best way to organize data science projects?  I use the following structure.  


### Repository structure
```
.
├── data           : processed data, versioned
├── docs           : communicable notes for the project
│   └── blog       : linked to _posts
├── raw            : files in this directory are shared as necessary
│   ├── data       : original data, treated as immutable
│   ├── docs       : "lab notes" throughout the course of the project, ordered by date
│   └── resources  : all resources pertaining to the project
├── results        : output of analysis or models
└── src            : all code for the project
    └── scripts    : scripts used in experiments, usually a staging ground for integration
```

## References

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
