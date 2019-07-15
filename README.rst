combo: A python Toolbox for ML Combination Methods
==================================================

**Deployment & Documentation & Stats**

.. image:: https://img.shields.io/pypi/v/combo.svg?color=brightgreen
   :target: https://pypi.org/project/combo/
   :alt: PyPI version

.. image:: https://img.shields.io/github/stars/yzhao062/combo.svg
   :target: https://github.com/yzhao062/combo/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/yzhao062/combo.svg?color=blue
   :target: https://github.com/yzhao062/combo/network
   :alt: GitHub forks


.. image:: https://img.shields.io/github/license/yzhao062/pyod.svg
   :target: https://github.com/yzhao062/pyod/blob/master/LICENSE
   :alt: License


-----


**combo** is a comprehensive Python model combination toolbox for
fusing/aggregating/selecting multiple base ML estimators,
under **supervised**, **unsupervised**, and **semi-supervised** scenarios. It
consists methods for various tasks, including **classification**,
**clustering**, **anomaly detection**, and **raw score combination**.

Model combination is an important task in
`ensemble learning <https://en.wikipedia.org/wiki/Ensemble_learning>`_,
but is often beyond the scope of ensemble learning. For instance, simple
averaging the results of the same classifiers with multiple runs is deemed as
a good way to eliminate the randomness in the classifier for a better stability.
Model combination has been widely used in data science competitions and
real-world tasks, such as Kaggle. See figure below for some popular combination
approaches.

.. image:: https://raw.githubusercontent.com/yzhao062/combo/master/docs/figs/framework_demo.png
   :target: https://raw.githubusercontent.com/yzhao062/combo/master/docs/figs/framework_demo.png
   :alt: Combination Framework Demo


combo is featured for:

* **Unified APIs, detailed documentation, and interactive examples** across various algorithms.
* **Advanced models**, including dynamic classifier/ensemble selection.
* **Comprehensive coverage** for supervised, unsupervised, and semi-supervised scenarios.
* **Rich applications** for classification, clustering, anomaly detection, and raw score combination.
* **Optimized performance with JIT and parallelization** when possible, using `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_.


**Table of Contents**\ :


* `Installation <#installation>`_
* `Proposed Algorithms <#proposed-algorithms>`_
* `Quick Start for classifier Combination <#quick-start-for-classifier-combination>`_
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
* numpy>=1.13
* numba>=0.35
* scipy>=0.19.1
* scikit_learn>=0.19.1


-----


Proposed Algorithms
^^^^^^^^^^^^^^^^^^^

**combo** will include various model combination frameworks by tasks:

* **Classifier combination**: combine multiple supervised classifiers together for training and prediction
* **Raw score & probability combination**: combine scores without invoking classifiers
* **Cluster combination**: combine unsupervised clustering results
  * Clusterer Ensemble [#Zhou2006Clusterer]_
* **Anomaly detection**: combine unsupervised outlier detectors


For each of the tasks, various methods may be introduced:

* **Simple methods**: averaging, maximization, weighted averaging, thresholding
* **Bucket methods**: average of maximization, maximization of average
* **Learning methods**: stacking (build an additional classifier to learn base estimator weights)
* **Selection methods**: dynamic classifier/ensemble selection [#Ko2008From]_
* Other models


-----


Quick Start for Classifier Combination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`"examples/classifier_comb_example.py" <https://github.com/yzhao062/combo/blob/master/examples/classifier_comb_example.py>`_
demonstrates the basic API of predicting with multiple classifiers. **It is noted that the API across all other algorithms are consistent/similar**.

#. Initialize a group of classifiers as base estimators

   .. code-block:: python


       from combo.models.classifier_comb import BaseClassiferAggregator

       # initialize a group of classifiers
       classifiers = [DecisionTreeClassifier(random_state=random_state),
                      LogisticRegression(random_state=random_state),
                      KNeighborsClassifier(),
                      RandomForestClassifier(random_state=random_state),
                      GradientBoostingClassifier(random_state=random_state)]


#. Initialize an aggregator class and pass in initialized classifiers for training

   .. code-block:: python


       # combine by averaging
       clf = BaseClassiferAggregator(classifiers)
       clf.fit(X_train, y_train)


#. Predict by averaging base classifier results and then evaluate

   .. code-block:: python


       # combine by averaging

       y_test_predicted = clf.predict(X_test, method='average')
       evaluate_print('Combination by avg  |', y_test, y_test_predicted)


#. Predict by maximizing base classifier results and then evaluate

   .. code-block:: python


       # combine by maximization

       y_test_predicted = clf.predict(X_test, method='maximization')
       evaluate_print('Combination by max  |', y_test, y_test_predicted)


#. See a sample output of classifier_comb_example.py

   .. code-block:: python


       Decision Tree       | Accuracy:0.9386, ROC:0.9383, F1:0.9521
       Logistic Regression | Accuracy:0.9649, ROC:0.9615, F1:0.973
       K Neighbors         | Accuracy:0.9561, ROC:0.9519, F1:0.9662
       Gradient Boosting   | Accuracy:0.9605, ROC:0.9524, F1:0.9699
       Random Forest       | Accuracy:0.9605, ROC:0.961, F1:0.9693

       Combination by avg  | Accuracy:0.9693, ROC:0.9677, F1:0.9763
       Combination by max  | Accuracy:0.9518, ROC:0.9312, F1:0.9642


-----


Development Status
^^^^^^^^^^^^^^^^^^

combo is currently **under development** as of July 15, 2019. A concrete plan has
been laid out and will be implemented in the next few months.

**Watch & Star** to get the latest update! Also feel free to send me an email (zhaoy@cmu.edu)
for suggestions and ideas.


----


Reference
^^^^^^^^^


.. [#Ko2008From] Ko, A.H., Sabourin, R. and Britto Jr, A.S., 2008. From dynamic classifier selection to dynamic ensemble selection. *Pattern recognition*, 41(5), pp.1718-1731.
.. [#Zhou2006Clusterer] Zhou, Z.H. and Tang, W., 2006. Clusterer ensemble. *Knowledge-Based Systems*, 19(1), pp.77-83.