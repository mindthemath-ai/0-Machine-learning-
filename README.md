Classical machine learning algorithms, explained from first principles and implemented from scratch in NumPy, then validated against scikit-learn.

Each notebook follows the same structure: intuition → math → from-scratch implementation → demo on a real dataset → decision-boundary visualization → complexity analysis, pros/cons and sources.

## Notebooks

| # | Notebook | Topics |
|---|----------|--------|
| 0 | [ML Foundations](0_ml_foundations.ipynb) | Learning problem, loss functions, Bayes error, empirical risk minimization, cross-entropy & KL divergence, train/val/test split, k-fold CV, metrics, feature scaling |
| 1 | [Decision Trees](1_decision_tree.ipynb) | Gini impurity, entropy & information gain, variance reduction, CART, stopping criteria, overfitting vs. `max_depth`, feature importance |
| 2 | [Ensemble Methods](2_random_forest.ipynb) | Bias–variance tradeoff, bagging, Random Forest & OOB error, AdaBoost, Gradient Boosting, XGBoost, voting |
| 3 | [Support Vector Machines](3_support_vector_machines.ipynb) | Margins, hard/soft margin, Lagrangian dual & KKT, kernel trick, Mercer's theorem, kernel SVM, tuning C and γ, SVR, multi-class (OvO / OvR) |
| 4 | [K-Nearest Neighbors](4_knn.ipynb) | Distance metrics, weighted KNN, choosing K, feature scaling, curse of dimensionality |
| 5 | [Naive Bayes & Bayesian Networks](5_naive_bayes.ipynb) | Gaussian, Multinomial and Bernoulli NB, log-probability trick, Laplace smoothing, text classification, Bayesian belief networks, d-separation, exact inference |
| 6 | [Discriminant Analysis](6_discriminant_analysis.ipynb) | LDA / QDA, Fisher's criterion, scatter matrices, LDA as dimensionality reduction, regularized LDA, comparison with Naive Bayes and logistic regression |

## Datasets
All loaded through scikit-learn, no manual download needed: Iris, Breast Cancer Wisconsin, California Housing, 20 Newsgroups.

## Requirements
pip install numpy scipy matplotlib scikit-learn xgboost jupyter
