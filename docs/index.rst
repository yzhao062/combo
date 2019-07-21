.. combo documentation master file, created by
   sphinx-quickstart on Tue Jul 16 15:42:55 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to combo's documentation!
=================================


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


----


Proposed Algorithms
^^^^^^^^^^^^^^^^^^^

**combo** will include various model combination frameworks by tasks:

* **Classifier combination**: combine multiple supervised classifiers together for training and prediction

  1. Averaging & Weighted Averaging :cite:`a-zhou2012ensemble`
  2. Maximization
  3. Majority Vote & Weighted Majority Vote :cite:`a-zhou2012ensemble`
  4. Dynamic Classifier Selection & Dynamic Ensemble Selection :cite:`a-ko2008dynamic` (work-in-progress)
  5. Stacking: build an additional classifier to learn base estimator weights (work-in-progress)

* **Raw score & probability combination**: combine scores without invoking classifiers

  1. Averaging & Weighted Averaging
  2. Maximization
  3. Average of Maximum (AOM)
  4. Maximum of Average (MOA)

* **Cluster combination**: combine unsupervised clustering results

  1. Clusterer Ensemble :cite:`a-zhou2006clusterer`

* **Anomaly detection**: combine unsupervised outlier detectors

  1. Averaging & Weighted Averaging
  2. Maximization
  3. Average of Maximum (AOM)
  4. Maximum of Average (MOA)
  5. Thresholding
  6. Locally Selective Combination (LSCP) :cite:`a-zhao2019lscp`


----


Development Status
^^^^^^^^^^^^^^^^^^

combo is currently **under development** as of July 15, 2019. A concrete plan has
been laid out and will be implemented in the next few months.

Similar to other libraries built by us, e.g., Python Outlier Detection Toolbox
(`pyod <https://github.com/yzhao062/pyod>`_),
combo is also targeted to be published in *Journal of Machine Learning Research (JMLR)*,
`open-source software track <http://www.jmlr.org/mloss/>`_.

**Watch & Star** to get the latest update! Also feel free to send me an email (zhaoy@cmu.edu)
for suggestions and ideas.


----


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   example


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   about


----


.. rubric:: References

.. bibliography:: zreferences.bib
   :cited:
   :labelprefix: A
   :keyprefix: a-



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
