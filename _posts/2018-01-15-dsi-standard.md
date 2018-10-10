---
layout: post
title:  Galvanize Data Science Immersive Program
---
## Data Science Immersive Program (January 2017)
The following is the standard of Galvanize Inc's 3 month full-time immersive program.  It closely resembles the body of knowledge obtained, though more time and work was needed for proficiency in application.  

### What is a Standard?  
Standards are the core-competencies of data scientists - the knowledge, skills, and habits every Galvanize graduate should possess. These were carefully crafted in a joint effort by your lead instructors, and represent those knowledge, skills, and habits we believe students need to get your foot in the door and be successful in industry.  

### Standards by Topic  

1. Python
  1. Explain the difference between mutable and immutable types and their relationship to dictionaries.
  2. Compare the strengths and weaknesses of lists vs. dictionaries.
  3. Choose the appropriate collection (dict, Counter, defaultdict) to simplify a problem.
  4. Compare the strengths and weaknesses of lists vs. generators.
  5. Write pythonic code.

2. Version Control / Git
  1. Explain the basic function and purpose of version control.
  2. Use a basic Git workflow to track project changes over time, share code, and write useful commit messages.

3. OOP
  1. Given the code for a python class, instantiate a python object and call the methods and list the attributes.
  2. Write the python code for a simple class.
  3. Match key “magic” methods to their functionality.
  4. Design a program or algorithm in object oriented fashion.
  5. Compare and contrast functional and object oriented programming.

4. SQL
  1. Connect to a SQL database via command line (i.e. Postgres).
  2. Connect to a database from within a python program.
  3. State function of basic SQL commands.
  4. Write simple queries on a single table including SELECT, FROM, WHERE, CASE clauses and aggregates.
  5. Write complex queries including JOINS and subqueries.
  6. Explain how indexing works in Postgres.
  7. Create and dump tables.
  8. Format a query to follow a standard style.
  9. Move data from SQL database to text file.

5. Pandas
  1. Explain/use the relationship between DataFrame and Series
  2. Know how to set, reset indexes
  3. Use iloc, loc, ix, and iat appropriately
  4. Use index alignment and know when it applies
  5. Use Split-Apply-Combine Methods
  6. Be able to read and write data to pandas
  7. Recognize problems that can probably be solved with Pandas (as opposed to writing vanilla Python functions).
  8. Use basic DateTimeIndex functionality

6. Plotting
  1. Describe the architecture of a matplotlib figure
  2. Plot in and outside of notebooks with matplotlib and seaborn
  3. Combine multiple datasets/categories in same plot
  4. Use subplots effectively
  5. Plot with Pandas
  6. Use and explain scatter_matrix output
  7. Use and explain a correlation heatmap
  8. Visualize pairwise relationships with seaborn
  9. Compare within-class distributions
  10. Use matplotlib techniques with seaborn

7. Visualization
  1. Explain the difference between exploratory and explanatory visualizations.
  2. Explain what a visualization is
  3. Don’t lie with data
  4. Visualize multidimensional relationships with data using position, size, color,  alpha, facets.
  5. Create an explanatory visualization that makes a relationship in data explicit.

8. Workflow
  1. Perform basic file operations from the command line, while consulting man/help/Google if necessary.
  2. Get help using man (ex man grep)
  3. Perform “survival” edits using vi, emacs, nano, or pico
  4. Configure environment & aliases in .bashrc/.bash_profile/.profile
  5. Install data science stack
  6. Manage a process with job control
  7. Examine system performance and kill processes
  8. Work on a remote machine with ssh/scp
  9. State what an RE (regular expression) is and write a simple one
  10. State the features and use cases of grep/sed/awk/cut/paste to process/clean a text file

8. Probability
  1. Define what a random variable is.
  2. Explain difference between permutations and combinations.  
  3. Recite and perform major probability laws from memory:
    * Bayes Rule
    * LOTP
    * Chain Rule
  4. Recite and perform major random variable formulas from memory:
    * E(X)
    * Var(X)
    * Cov(X,Y)
  5. Describe what a joint distribution is and be able to perform a simple calculation using joint distribution.
  6. Define each major probability distributions and give 1 clear example of each
  7. Explain independence of 2 r.v.’s and implications with respect to probability formulas, covariance formulas, etc.
  8. Compute expectation of aX+bY and explain that it is a linear operator, where X and Y are random variables
  9. Compute variance of aX + bY
  10. Discuss why correlation is not causation
  11. Describe correlation and its perils, with reference to Anscombe’s quartet

9. Sampling
  1. Compute MLE estimate for simple example (such as coin-flipping)
  2. Pseudocode Bootstrapping for a given sample of size N.
  3. Construct confidence interval for case where parametric construction does not work
  4. Discuss examples of times when you need bootstrapping.
  5. Define the Central Limit Theorem
  6. Compute standard error
  7. Compare and contrast the use cases of parametric and nonparametric estimation

10. Hypothesis Testing
  1. Given a dataset, set up a null and alternative hypothesis, and calculate and interpret the p-value for the difference of means or proportions.
  2. Given a dataset, set up a null and alternative hypothesis, and calculate and interpret the p-value for Chi-square test of independence
  3. Describe a situation in which a one-tailed test would be appropriate (vs. a two-tailed test).
  4. Given a particular situation, correctly choose among the following options:
    * z-test
    * t-test
    * 2 sample t-test (one-sided and two-sided)
    * 2 sample z-test (one-sided and two-sided)
  5. Define p-value, Type I error, Type II error, significance level and discuss their significance in an example problem.
  6. Account for the multiple comparisons problem via Bonferroni correction.
  7. Compute the difference of two independent random normal variables.
  8. Discuss when to use an A/B test to evaluate the efficacy of a treatment

11. Power
  1. Define Power and relate it to the Type II error.
  2. Compute power given a dataset and a problem.
  3. Explain how the following factors contribute to power:
    * sample size
    * effect size (difference between sample statistics and statistic formulated under the null)
    * significance level
  4. Identify what can be done to increase power.
  5. Estimate sample size required of a test (power analysis) for one sample mean or proportion case
  6. Solve by hand for the posterior distribution for a uniform prior based on coin flips.
  7. Solve Discrete Bayes problem with some data
  8. What is the difference between Bayesian and Frequentist inference, with respect to fixed parameters and prior beliefs?
  9. Define power - Be able to draw the picture with two normal curves with different means and highlight the section that represents Power.
  10. Explain trade off between significance and power

12. Multi Armed Bandit
  1. Explain the difference between a frequentist A/B test and a Bayesian A/B test.
  2. Define and explain prior, likelihood, and posterior.
  3. Explain what a conjugate prior is and how it applies to A/B testing.
  4. Analyze an A/B test with the Bayesian approach.
  5. Explain how multi-armed bandit addresses the tradeoff between exploitation and exploration, and the relationship to regret.
  6. Write pseudocode for the Multi-Armed Bandit algorithm.

13. Linear Algebra in Python
  1. Perform basic Linear Algebra operations by hand: Multiply matrices, subtract matrices, Transpose matrices, verify inverses.
  2. Perform linear algebra operations (multiply matrices, transpose matrices, and invert matrices) in numpy.

14. Exploratory Data Analysis (EDA)
  1. Define EDA in your own words.
  2. Identify the key questions of EDA.
  3. Perform EDA on a dataset.

15. Linear Regression
  1. State and troubleshoot the assumptions of linear regression model. Describe, interpret, and visualize the model form of linear regression: Y = B0+B1X1+B2X2+....
  2. Relate Beta vector solution of Ordinary Least Squares to the cost function (residual sum of squares)
  3. Perform ordinary least squares (OLS) with statsmodels and interpret the output: Beta coefficients, p-values, R^2, adjusted-R^2, AIC, BIC
  4. Explain how to incorporate interactions and categorical variables into linear regression
  5. Explain how one can detect outliers

16. Cross Validation & Regularized Linear Regression
  1. Perform (one-fold) cross-validation on dataset (train test splitting)
  2. Algorithmically, explain k-fold cross-validation
  3. Give the reasoning for using k-fold cross-validation
  4. Given one full model and one regularized model, name 2 appropriate ways to compare the two models. Name 1 inappropriate way.
  5. Generally, when we increase flexibility or complexity of model, what happens to bias? variance? training error? test error?
  6. Compare and contrast Lasso and Ridge regression.
  7. What happens to Bias and Variance as we change the following factors: sample size, number of parameters, etc.
  8. What is the cost function for Ridge? for Lasso?
  9. Build test error curve for Ridge regression, while varying the alpha parameter, to determine optimal level or regularization
  10. Build and interpret Learning curves for two learning algorithms, one that is overfit (high variance, low bias) and one that is underfit (low variance, high bias)

17. Logistic Regression
  1. Place logistic regression in the taxonomy of ML algorithms
  2. Fit and interpret a logistic regression model in scikit-learn
  3. Interpret the coefficients of logistic regression, using odds ratio
  4. Explain ROC curves
  5. Explain the key differences and similarities between logistic and linear regression.

18. Gradient Descent
  1. Identify and justify use cases for and failure modes of gradient descent.
  2. Write pseudocode of the gradient descent and stochastic gradient descent algorithms.
  3. Compare and contrast batch and stochastic gradient descent - the algorithms, costs, and benefits.

19. Decision Trees
  1. Thoroughly explain the construction of a decision tree (classification or regression), including selecting an impurity measure (gini, entropy, variance)
  2. Recognize overfitting and explain pre/post pruning and why it helps.
  3. Pick the ‘best’ tree via cross-validation, for a given data set.
  4. Discuss pros and cons

20. k-th nearest neighbor (kNN)
  1. Write pseudocode for the kNN algorithm from scratch
  2. State differences between kNN regression and classification
  3. Discuss Pros and Cons of kNN

21. Random Forest
  1. Thoroughly explain the construction of a random forest (classification or regression) algorithm
  2. Explain the relationship and difference between random forest and bagging.
  3. Explain why random forests are more accurate than a single decision tree.
  4. Explain how to get feature importances from a random forest using an algorithm
  5. How is OOB error calculated and what is it an estimate of?

22. Boosted Trees
  1. Define boosting in your own words.
  2. Be able to interpret boosting output
  3. List advantages and disadvantages of boosting.
  4. Compare and contrast boosting with other ensemble methods
  5. Explain each of the tuning parameters and specifically how they affect the model
  6. Learn, tune, and score a model using scikit-learn’s boosting class
  7. Implement AdaBoost

23. Support Vector Machines (SVM)
  1. Compute a hyperplane as a decision boundary in SVC
  2. Explain what a support vector is in plain english
  3. Recognize that preprocessing, specifically making sure all predictors are on the same scale, is a necessary step
  4. Explain SVC using the hyperparameter, C
  5. Tune a SVM with an RBF using both hyperparameters C and gamma
  6. Tune a SVM with a polynomial kernel using both hyperparameters C and degree
  7. Describe why generally speaking, an SVM with RBF kernel is more likely to perform well on “tall” data as opposed to “wide” data.
  8. For SVMs with RBF, state what happens to bias and variance as we increase the hyperparameter “C”. State what happens to bias and variance as we increase the hyperparameter “gamma”.
  9. State how the “one-vs-one” and “one-vs-rest” approaches for multi-class problems are implemented.
  10. Describe the kernel trick, being able to calculate as if high dimensional space.

24. Profit Curves
  1. Describe the issues with imbalanced classes.
  2. Explain the profit curve method for thresholding.
  3. Explain sampling methods and give examples of sampling methods.
  4. Explain how they deal with imbalanced classes.
  5. Explain cost sensitive learning and how it deals with imbalanced classes.

25. Webscraping
  1. Compare and contrast SQL and noSQL.
  2. Complete basic operations with mongo.
  3. Explain the basic concepts of HTML.
  4. Write python code to pull out an element from a web page.
  5. Fetch data from an existing API

26. Naive Bayes
  1. Derive the naive bayes algorithm and discuss its assumptions.
  2. Contrast generative and discriminative models.
  3. Discuss the pros and cons of Naive Bayes.

27. NLP
  1. Identify and explain ways of featurizing text.
  2. List and explain distance metrics used in document classification.
  3. Featurize a text corpus in Python using nltk and scikit-learn.

28. Clustering
  1. List the characteristics of a dataset necessary to perform K-means
  2. Detail the k-means algorithm in steps, commenting on convergence or lack thereof.
  3. Use the elbow method to determine K and evaluate the choice
  4. Interpret Silhouette plot
  5. Interpret clusters by examining cluster centers, and exploring the data within each cluster (dataframe inspection, plotting, decision trees for cluster membership)
  6. Build and interpret a dendrogram using hierarchical clustering.
  7. Compare and contrast k-means and hierarchical clustering.

29. Churn Case Study
  1. List and explain the steps in CRISP-DM (Cross-Industry Standard Process for Data Mining)
  2. Perform EDA standards on case study including visualizations
  3. Discuss ramifications of deleting missing values when
    * MAR (missing at random)
    * MCAR (missing completely at random)
    * MNAR (missing not at random)
  4. Explain imputing missing using at least 2 different methods, list pros and cons of each method
  5. Explain when dropping rows is okay, when dropping features is okay?
  6. Be able to perform the feature engineering process
  7. Be able to identify target leak, and explain why this happens
  8. State appropriate business goal and evaluation metric

30. Dimensionality Reduction
  1. List reasons for reducing the dimensions.
  2. Describe how the principal components are constructed in PCA.
  3. Interpret the principal components of PCA.
  4. Determine how many principal components to keep.
  5. Describe the relationship between PCA and SVD.
  6. Compute and interpret PCA using sklearn.
  7. Memorize the eigenvalue equation

31. NMF
  1. Write down and explain the NMF equation.
  2. Compare and contrast NMF, SVD, and PCA, and k-means
  3. Implement Alternating-Least-Squares algorithm for NMF
  4. Find and interpret latent topics in a corpus of documents with NMF
  5. Explain how to interpret H matrix? W matrix?
  6. Explain regularization in the context of NMF.

32. Recommender Systems
  1. Survey approaches to recommenders, their pros & cons, and when each is likely to be best.
  2. Describe the cold start problem and know how it affects different recommendation strategies
  3. Explain either the collaborative filtering algorithm or the matrix factorization recommender algorithm.
  4. Discuss recommender evaluation.
  5. Discuss performance concerns for recommenders.

33. Graphs
  1. Define a graph and discuss the implementation.
  2. List common applications of graph models.
  3. Discuss the searching algorithms and applications of them.
  4. Explain the various ways of measuring the importance of a node.
  5. Explain methods and applications of clustering on a graph.
  6. Use appropriate package to build graph data structure in Python and execute common algorithms (shortest path, connected components, …)
  7. Explain the various ways of measuring the importance of a node.
  8. Explain methods and applications of clustering on a graph.

34. Cloud Computing
  1. Scope & Configure a data science environment on AWS.
  2. Protect AWS resources against unauthorized access.
  3. Manage AWS resources using awscli, ssh, scp, or boto3.
4. Monitor and control costs incurred on AWS

35. Parallel Computing
  1. Define and contrast processes vs. threads
  2. Define and contrast parallelism and concurrency.
  3. Recognize problems that require parallelism or concurrency
  4. Implement parallel and concurrent solutions
  5. Instrument approaches to see the benefit of threading/parallelism.

36. Map Reduce
  1. Explain Types of Problems which benefit from MapReduce
  2. Describe map-reduce, and how it relates to Hadoop
  3. Explain how to select the number of mappers and reducers
  4. Describe the role of keys in MapReduce
5. Perform MapReduce in python using MRJob.

37. Time Series
  1. Recognize when time series analysis could be applied
  2. Define key times series concepts
  3. Determine structure of a time-series using graphical tools
  4. Compute a forecast using Box-Jenkins Methodology
  5. Evaluate models/forecasts using cross validation and statistical tests
  6. Engineer features to handle seasonal, calendar, and periodic components
  7. Explain taxonomy of exponential smoothing using ETS framework

38. Spark
  1. Configure a machine to use spark effectively
  2. Describe differences and similarities between MapReduce and Spark
  3. Get data into spark for processing.
  4. Describe lazy evaluation in the context of Spark.
  5. Cache RDDs effectively to improve performance.
  6. Use Spark to do compute basic statistics
  7. Know the difference between Spark data types: RDD, DataFrame, DAG
  8. Use MLLib

39. SQL in Spark
  1. Identify what distinguishes a Spark DataFrame from an RDD
  2. Explain how to create a Spark DataFrame
  3. Query a DF with SQL
  4. Transform a DF with dataframe methods
  5. Describe the challenges and requirements of saving schema’d datasets.
  6. Use user-defined functions

40. Data Products
  1. Explain REST architecture/API
  2. Write a basic Flask API
  3. Describe web architecture at a high level
  4. Know the role of javascript in a web application
  5. Know how to use developer tools to inspect an application
  6. Write a basic Flask web application
  7. Be able to describe the difference between online and offline computation

41. Fraud Case Study
  1. Build an MVP (minimum viable product) quickly
  2. Build a dashboard
  3. Build system to take in online data from a stream
  4. Build production-quality product

42. Whiteboarding
  1. Explain the meaning of Big-Oh.
  2. Analyze the runtime of code.
  3. Solve whiteboarding interview questions.
  4. Apply different techniques to addressing a whiteboarding interview problem

43. Business Analytics
  1. Explain funnel metrics and applications
  2. Identify red flags in a set of funnel metrics
  3. Identify and discuss appropriate use cases for cohort analysis
  4. Identify and explain the limits of data analysis
  5. Given an open ended question, identify the business goal, metrics, and relevant data science solution.
  6. Identify excessive or improper use of data analysis
  7. Explain how data science is used in industry
  8. Understand range of business problems where AB testing applies
