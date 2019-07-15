combo: A python toolbox for combining machine learning models
=============================================================

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
* `Quick Start for Score Combination <#quick-start-for-score-combination>`_
* `Proposed Algorithms <#proposed-algorithms>`_


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


Proposed Functionalities
^^^^^^^^^^^^^^^^^^^^^^^^

**combo** will include various model combination frameworks by tasks:

* **Classifier combination**: combine multiple supervised classifiers together for training and prediction
* **Raw score & probability combination**: combine scores without invoking classifiers
* **Cluster combination**: combine unsupervised clustering results
* **Anomaly detection**: combine unsupervised outlier detectors


For each of the tasks, various methods may be introduced:

* **Simple methods**: averaging, maximization, weighted averaging, thresholding
* **Bucket methods**: average of maximization, maximization of average
* **Learning methods**: stacking (build an additional classifier to learn base estimator weights)
* **Selection methods**: dynamic classifier/ensemble selection
* Other models


Quick Start for Score Combination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Development Status
^^^^^^^^^^^^^^^^^^

combo is currently **under development** as of July 14, 2019. A concrete plan has
been laid out and will be implemented in the next few months.

**Watch & Star** to get the latest update! Also feel free to send me an email (zhaoy@cmu.edu)
for suggestions and ideas.