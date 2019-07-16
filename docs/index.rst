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


----


Proposed Algorithms
^^^^^^^^^^^^^^^^^^^

**combo** will include various model combination frameworks by tasks:

* **Classifier combination**: combine multiple supervised classifiers together for training and prediction
* **Raw score & probability combination**: combine scores without invoking classifiers
* **Cluster combination**: combine unsupervised clustering results
  * Clusterer Ensemble :cite:`a-zhou2006clusterer`
* **Anomaly detection**: combine unsupervised outlier detectors


For each of the tasks, various methods may be introduced:

* **Simple methods**: averaging, maximization, weighted averaging, thresholding
* **Bucket methods**: average of maximization, maximization of average
* **Learning methods**: stacking (build an additional classifier to learn base estimator weights)
* **Selection methods**: dynamic classifier/ensemble selection :cite:`a-ko2008dynamic`
* Other models


----


Development Status
^^^^^^^^^^^^^^^^^^

combo is currently **under development** as of July 15, 2019. A concrete plan has
been laid out and will be implemented in the next few months.

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

   combo


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
