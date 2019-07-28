combo: A Python Toolbox for Machine Learning Model Combination
==============================================================


**Deployment & Documentation & Stats**

.. image:: https://img.shields.io/pypi/v/combo.svg?color=brightgreen
   :target: https://pypi.org/project/combo/
   :alt: PyPI version


.. image:: https://readthedocs.org/projects/pycombo/badge/?version=latest
   :target: https://pycombo.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://img.shields.io/github/stars/yzhao062/combo.svg
   :target: https://github.com/yzhao062/combo/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/yzhao062/combo.svg?color=blue
   :target: https://github.com/yzhao062/combo/network
   :alt: GitHub forks


.. image:: https://pepy.tech/badge/combo
   :target: https://pepy.tech/project/combo
   :alt: Downloads


.. image:: https://pepy.tech/badge/combo/month
   :target: https://pepy.tech/project/combo
   :alt: Downloads


-----


**Build Status & Coverage & Maintainability & License**


.. image:: https://travis-ci.org/yzhao062/combo.svg?branch=master
   :target: https://travis-ci.org/yzhao062/combo
   :alt: Build Status


.. image:: https://ci.appveyor.com/api/projects/status/te7uieha87305ike/branch/master?svg=true
   :target: https://ci.appveyor.com/project/yzhao062/combo/branch/master
   :alt: Build status


.. image:: https://coveralls.io/repos/github/yzhao062/combo/badge.svg
   :target: https://coveralls.io/github/yzhao062/combo
   :alt: Coverage Status


.. image:: https://api.codeclimate.com/v1/badges/465ebba81e990abb357b/maintainability
   :target: https://codeclimate.com/github/yzhao062/combo/maintainability
   :alt: Maintainability


.. image:: https://img.shields.io/github/license/yzhao062/combo.svg
   :target: https://github.com/yzhao062/combo/blob/master/LICENSE
   :alt: License


-----


**combo** is a Python toolbox for combining or aggregating ML models and
scores for various tasks, including **classification**, **clustering**,
**anomaly detection**, and **raw score**. It has been widely used in data
science competitions and real-world tasks, such as Kaggle.

Model and score combination can be regarded as a subtask of
`ensemble learning <https://en.wikipedia.org/wiki/Ensemble_learning>`_,
but is often beyond the scope of ensemble learning. For instance,
averaging the results of multiple runs of a ML model is deemed as
a reliable way of eliminating the randomness for better stability. See
figure below for some popular combination approaches.

.. image:: https://raw.githubusercontent.com/yzhao062/combo/master/docs/figs/framework_demo.png
   :target: https://raw.githubusercontent.com/yzhao062/combo/master/docs/figs/framework_demo.png
   :alt: Combination Framework Demo


combo is featured for:

* **Unified APIs, detailed documentation, and interactive examples** across various algorithms.
* **Advanced models**, including dynamic classifier/ensemble selection and LSCP.
* **Broad applications** for classification, clustering, anomaly detection, and raw score.
* **Comprehensive coverage** for supervised, unsupervised, and semi-supervised scenarios.
* **Optimized performance with JIT and parallelization** when possible, using `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_.


**API Demo**\ :


   .. code-block:: python


       from combo.models.stacking import Stacking
       # base classifiers
       classifiers = [DecisionTreeClassifier(), LogisticRegression(),
                      KNeighborsClassifier(), RandomForestClassifier(),
                      GradientBoostingClassifier()]

       clf = Stacking(base_clfs=classifiers) # initialize a Stacking model
       clf.fit(X_train)

       # predict on unseen data
       y_test_labels = clf.predict(X_test)  # label prediction
       y_test_proba = clf.predict_proba(X_test)  # probability prediction


**Table of Contents**\ :


* `Installation <#installation>`_
* `API Cheatsheet & Reference <#api-cheatsheet--reference>`_
* `Proposed Algorithms <#proposed-algorithms>`_
* `Quick Start for Classifier Combination <#quick-start-for-classifier-combination>`_
* `Quick Start for Clustering Combination <#quick-start-for-clustering-combination>`_
* `An Example of Stacking <#an-example-of-stacking>`_
* `Development Status <#development-status>`_


----



Installation
^^^^^^^^^^^^

It is recommended to use **pip** for installation. Please make sure
**the latest version** is installed, as combo is updated frequently:

.. code-block:: bash

   pip install combo            # normal install
   pip install --upgrade combo  # or update if needed
   pip install --pre combo      # or include pre-release version for new features

Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/yzhao062/combo.git
   cd combo
   pip install .


**Required Dependencies**\ :


* Python 3.5, 3.6, or 3.7
* joblib
* matplotlib
* numpy>=1.13
* numba>=0.35
* scipy>=0.19.1
* scikit_learn>=0.19.1


----


API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

Full API Reference: (https://pycombo.readthedocs.io/en/latest/api.html). API cheatsheet for most of the models:


* **fit(X)**\ : Fit an estimator.
* **predict(X)**\ : Predict on a particular sample once the estimator is fitted.
* **predict_proba(X)**\ : Predict the probability of a sample belonging to each class. Only applicable for classification tasks.


----


Proposed Algorithms
^^^^^^^^^^^^^^^^^^^


**combo** groups combination frameworks by tasks.

* For most of the tasks, the following **combination methods for raw scores** are feasible [#Zhou2012Ensemble]_:

  1. Averaging & Weighted Averaging & Median
  2. Maximization
  3. Majority Vote & Weighted Majority Vote
  4. Median

Some of the methods are tasks specific:

* **Classifier combination**: combine multiple supervised classifiers together
  for training and prediction

  1. SimpleClassifierAggregator: combining classifiers by (i) (weighted) average (ii) maximization (iii) median and (iv) (weighted) majority vote
  2. Dynamic Classifier Selection & Dynamic Ensemble Selection [#Ko2008From]_ (work-in-progress)
  3. Stacking (meta ensembling): build an additional classifier to learn base estimator weights [#Gorman2016Kaggle]_


* **Cluster combination**: combine and align unsupervised clustering results

  1. Clusterer Ensemble [#Zhou2006Clusterer]_


* **Anomaly detection**: combine unsupervised (and supervised) outlier detectors

  1. SimpleDetectorCombination: combining outlier score results by (i) (weighted) average (ii) maximization (iii) median and (iv) (weighted) majority vote
  2. Average of Maximum (AOM) [#Aggarwal2015Theoretical]_
  3. Maximum of Average (MOA) [#Aggarwal2015Theoretical]_
  4. Thresholding
  5. Locally Selective Combination (LSCP) [#Zhao2019LSCP]_
  6. XGBOD: a semi-supervised combination framework for outlier detection [#Zhao2018XGBOD]_


-----


Quick Start for Classifier Combination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`"examples/classifier_comb_example.py" <https://github.com/yzhao062/combo/blob/master/examples/classifier_comb_example.py>`_
demonstrates the basic API of predicting with multiple classifiers. **It is noted that the API across all other algorithms are consistent/similar**.

#. Initialize a group of classifiers as base estimators

   .. code-block:: python


       # initialize a group of classifiers
       classifiers = [DecisionTreeClassifier(), LogisticRegression(),
                      KNeighborsClassifier(), RandomForestClassifier(),
                      GradientBoostingClassifier()]


#. Initialize, fit, predict, and evaluate with a simple aggregator (average)

   .. code-block:: python


       from combo.models.classifier_comb import SimpleClassifierAggregator

       clf = SimpleClassifierAggregator(classifiers, method='average')
       clf.fit(X_train, y_train)
       y_test_predicted = clf.predict(X_test)
       evaluate_print('Combination by avg   |', y_test, y_test_predicted)



#. See a sample output of classifier_comb_example.py

   .. code-block:: python


       Decision Tree        | Accuracy:0.9386, ROC:0.9383, F1:0.9521
       Logistic Regression  | Accuracy:0.9649, ROC:0.9615, F1:0.973
       K Neighbors          | Accuracy:0.9561, ROC:0.9519, F1:0.9662
       Gradient Boosting    | Accuracy:0.9605, ROC:0.9524, F1:0.9699
       Random Forest        | Accuracy:0.9605, ROC:0.961, F1:0.9693

       Combination by avg   | Accuracy:0.9693, ROC:0.9677, F1:0.9763
       Combination by w_avg | Accuracy:0.9781, ROC:0.9716, F1:0.9833
       Combination by max   | Accuracy:0.9518, ROC:0.9312, F1:0.9642
       Combination by w_vote| Accuracy:0.9649, ROC:0.9644, F1:0.9728
       Combination by median| Accuracy:0.9693, ROC:0.9677, F1:0.9763


-----


Quick Start for Clustering Combination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`"examples/cluster_comb_example.py" <https://github.com/yzhao062/combo/blob/master/examples/cluster_comb_example.py>`_
demonstrates the basic API of combining multiple base clustering estimators.

#. Initialize a group of clustering methods as base estimators

   .. code-block:: python


       from combo.models.cluster_comb import ClustererEnsemble

       # Initialize a set of estimators
       estimators = [KMeans(n_clusters=n_clusters),
                     MiniBatchKMeans(n_clusters=n_clusters),
                     AgglomerativeClustering(n_clusters=n_clusters)]


#. Initialize an Clusterer Ensemble class and fit the model

   .. code-block:: python


       # combine by Clusterer Ensemble
       clf = ClustererEnsemble(estimators, n_clusters=n_clusters)
       clf.fit(X)


#. Get the aligned results

   .. code-block:: python


       # generate the labels on X
       aligned_labels = clf.aligned_labels_
       predicted_labels = clf.labels_


-----


An Example of Stacking
^^^^^^^^^^^^^^^^^^^^^^

`"examples/stacking_example.py" <https://github.com/yzhao062/combo/blob/master/examples/stacking_example.py>`_
demonstrates the basic API of stacking (meta ensembling).


#. Initialize a group of classifiers as base estimators

   .. code-block:: python


       # initialize a group of classifiers
       classifiers = [DecisionTreeClassifier(), LogisticRegression(),
                      KNeighborsClassifier(), RandomForestClassifier(),
                      GradientBoostingClassifier()]


#. Initialize, fit, predict, and evaluate with Stacking

   .. code-block:: python


       from combo.models.stacking import Stacking

       clf = Stacking(base_clfs=classifiers, n_folds=4, shuffle_data=False,
                   keep_original=True, use_proba=False, random_state=random_state)

       clf.fit(X_train, y_train)
       y_test_predict = clf.predict(X_test)
       evaluate_print('Stacking | ', y_test, y_test_predict)


#. See a sample output of stacking_example.py

   .. code-block:: python


       Decision Tree        | Accuracy:0.9386, ROC:0.9383, F1:0.9521
       Logistic Regression  | Accuracy:0.9649, ROC:0.9615, F1:0.973
       K Neighbors          | Accuracy:0.9561, ROC:0.9519, F1:0.9662
       Gradient Boosting    | Accuracy:0.9605, ROC:0.9524, F1:0.9699
       Random Forest        | Accuracy:0.9605, ROC:0.961, F1:0.9693

       Stacking             | Accuracy:0.9868, ROC:0.9841, F1:0.9899


-----


Development Status
^^^^^^^^^^^^^^^^^^

combo is currently **under development** as of July 24, 2019. A concrete plan has
been laid out and will be implemented in the next few months.

Similar to other libraries built by us, e.g., Python Outlier Detection Toolbox
(`pyod <https://github.com/yzhao062/pyod>`_),
combo is also targeted to be published in *Journal of Machine Learning Research (JMLR)*,
`open-source software track <http://www.jmlr.org/mloss/>`_. A demo paper to
*AAAI* or *IJCAI* may be submitted soon for progress update.

**Watch & Star** to get the latest update! Also feel free to send me an email (zhaoy@cmu.edu)
for suggestions and ideas.


----


Reference
^^^^^^^^^

.. [#Aggarwal2015Theoretical] Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and algorithms for outlier ensembles. *ACM SIGKDD Explorations Newsletter*, 17(1), pp.24-47.

.. [#Gorman2016Kaggle] Gorman, B. (2016). A Kaggler's Guide to Model Stacking in Practice. [online] The Official Blog of Kaggle.com. Available at: http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice [Accessed 26 Jul. 2019].

.. [#Ko2008From] Ko, A.H., Sabourin, R. and Britto Jr, A.S., 2008. From dynamic classifier selection to dynamic ensemble selection. *Pattern recognition*, 41(5), pp.1718-1731.

.. [#Zhao2019LSCP] Zhao, Y., Nasrullah, Z., Hryniewicki, M.K. and Li, Z., 2019, May. LSCP: Locally selective combination in parallel outlier ensembles. In *Proceedings of the 2019 SIAM International Conference on Data Mining (SDM)*, pp. 585-593. Society for Industrial and Applied Mathematics.

.. [#Zhao2018XGBOD] Zhao, Y. and Hryniewicki, M.K. XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning. *IEEE International Joint Conference on Neural Networks*, 2018.

.. [#Zhou2006Clusterer] Zhou, Z.H. and Tang, W., 2006. Clusterer ensemble. *Knowledge-Based Systems*, 19(1), pp.77-83.

.. [#Zhou2012Ensemble] Zhou, Z.H., 2012. Ensemble methods: foundations and algorithms. Chapman and Hall/CRC.