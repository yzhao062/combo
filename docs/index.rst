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


.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/yzhao062/combo/master
   :alt: Binder

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


----


**Build Status & Coverage & Maintainability & License**


.. image:: https://github.com/yzhao062/combo/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/yzhao062/combo/actions/workflows/testing.yml
   :alt: testing


.. image:: https://circleci.com/gh/yzhao062/combo.svg?style=svg
   :target: https://circleci.com/gh/yzhao062/combo
   :alt: Circle CI


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


----


**combo** is a comprehensive Python toolbox for **combining machine learning (ML) models and scores**.
**Model combination** can be considered as a subtask of `ensemble learning <https://en.wikipedia.org/wiki/Ensemble_learning>`_,
and has been widely used in real-world tasks and data science competitions like Kaggle :cite:`a-bell2007lessons`.
**combo** has been used/introduced in various research works since its inception :cite:`a-zhao2019pyod,a-raschka2020machine`.

**combo** library supports the combination of models and score from
key ML libraries such as `scikit-learn <https://scikit-learn.org/stable/index.html>`_,
`xgboost <https://xgboost.ai/>`_, and `LightGBM <https://github.com/microsoft/LightGBM>`_,
for crucial tasks including classification, clustering, anomaly detection.
See figure below for some representative combination approaches.

.. image:: https://raw.githubusercontent.com/yzhao062/combo/master/docs/figs/framework_demo.png
   :target: https://raw.githubusercontent.com/yzhao062/combo/master/docs/figs/framework_demo.png
   :alt: Combination Framework Demo


**combo** is featured for:

* **Unified APIs, detailed documentation, and interactive examples** across various algorithms.
* **Advanced and latest models**, such as Stacking/DCS/DES/EAC/LSCP.
* **Comprehensive coverage** for classification, clustering, anomaly detection, and raw score.
* **Optimized performance with JIT and parallelization** when possible, using `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_.


**API Demo**\ :

.. code-block:: python


   from combo.models.classifier_stacking import Stacking
   # initialize a group of base classifiers
   classifiers = [DecisionTreeClassifier(), LogisticRegression(),
                  KNeighborsClassifier(), RandomForestClassifier(),
                  GradientBoostingClassifier()]

   clf = Stacking(base_estimators=classifiers) # initialize a Stacking model
   clf.fit(X_train, y_train) # fit the model

   # predict on unseen data
   y_test_labels = clf.predict(X_test)  # label prediction
   y_test_proba = clf.predict_proba(X_test)  # probability prediction


**Citing combo**\ :

`combo paper <http://www.andrew.cmu.edu/user/yuezhao2/papers/20-aaai-combo.pdf>`_ is published in
`AAAI 2020 <https://aaai.org/Conferences/AAAI-20/>`_ (demo track).
If you use combo in a scientific publication, we would appreciate citations to the following paper::

    @inproceedings{zhao2020combo,
      title={Combining Machine Learning Models and Scores using combo library},
      author={Zhao, Yue and Wang, Xuejian and Cheng, Cheng and Ding, Xueying},
      booktitle={Thirty-Fourth AAAI Conference on Artificial Intelligence},
      month = {Feb},
      year={2020},
      address = {New York, USA}
    }

or::

    Zhao, Y., Wang, X., Cheng, C. and Ding, X., 2020. Combining Machine Learning Models and Scores using combo library. Thirty-Fourth AAAI Conference on Artificial Intelligence.


**Key Links and Resources**\ :

* `awesome-ensemble-learning <https://github.com/yzhao062/awesome-ensemble-learning>`_ (ensemble learning related books, papers, and more)
* `View the latest codes on Github <https://github.com/yzhao062/combo>`_
* `View the documentation & API <https://pycombo.readthedocs.io/>`_
* `View all examples <https://github.com/yzhao062/combo/tree/master/examples>`_
* `View the demo video for AAAI 2020 <https://youtu.be/PaSJ49Ij7w4>`_
* `Execute Interactive Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/combo/master>`_


----


API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

Full API Reference: (https://pycombo.readthedocs.io/en/latest/api.html). The
following APIs are applicable for most models for easy use.

* :func:`combo.models.base.BaseAggregator.fit`: Fit estimator. y is optional for unsupervised methods.
* :func:`combo.models.base.BaseAggregator.predict`: Predict on a particular sample once the estimator is fitted.
* :func:`combo.models.base.BaseAggregator.predict_proba`: Predict the probability of a sample belonging to each class once the estimator is fitted.
* :func:`combo.models.base.BaseAggregator.fit_predict`: Fit estimator and predict on X. y is optional for unsupervised methods.


For raw score combination (after the score matrix is generated),
use individual methods from
`"score_comb.py" <https://github.com/yzhao062/combo/blob/master/combo/models/score_comb.py>`_ directly.
Raw score combination API: (https://pycombo.readthedocs.io/en/latest/api.html#score-combination).


----


Implemented Algorithms
^^^^^^^^^^^^^^^^^^^^^^


**combo** groups combination frameworks by tasks. General purpose methods are
fundamental ones which can be applied to various tasks.

==================================================================  ===================  ======================================================================================================  =====  ===========================================
Class/Function                                                      Task                 Algorithm                                                                                               Year   Ref
==================================================================  ===================  ======================================================================================================  =====  ===========================================
:mod:`combo.models.score_comb.average`                              General Purpose      Average & Weighted Average: average across all scores/prediction results, maybe with weights            N/A    :cite:`a-zhou2012ensemble`
:mod:`combo.models.score_comb.maximization`                         General Purpose      Maximization: simple combination by taking the maximum scores                                           N/A    :cite:`a-zhou2012ensemble`
:mod:`combo.models.score_comb.median`                               General Purpose      Median: take the median value across all scores/prediction results                                      N/A    :cite:`a-zhou2012ensemble`
:mod:`combo.models.score_comb.majority_vote`                        General Purpose      Majority Vote & Weighted Majority Vote                                                                  N/A    :cite:`a-zhou2012ensemble`
:class:`combo.models.classifier_comb.SimpleClassifierAggregator`    Classification       SimpleClassifierAggregator: combining classifiers by general purpose methods above                      N/A    N/A
:class:`combo.models.classifier_dcs.DCS_LA`                         Classification       DCS: Dynamic Classifier Selection (Combination of multiple classifiers using local accuracy estimates)  1997   :cite:`a-woods1997combination`
:class:`combo.models.classifier_des.DES_LA`                         Classification       DES: Dynamic Ensemble Selection (From dynamic classifier selection to dynamic ensemble selection)       2008   :cite:`a-ko2008dynamic`
:class:`combo.models.classifier_stacking.Stacking`                  Classification       Stacking (meta ensembling): use a meta learner to learn the base classifier results                     N/A    :cite:`a-gorman2016kaggle`
:class:`combo.models.cluster_comb.ClustererEnsemble`                Clustering           Clusterer Ensemble: combine the results of multiple clustering results by relabeling                    2006   :cite:`a-zhou2006clusterer`
:class:`combo.models.cluster_eac.EAC`                               Clustering           Combining multiple clusterings using evidence accumulation (EAC)                                        2002   :cite:`a-fred2005combining`
:class:`combo.models.detector_comb.SimpleDetectorAggregator`        Anomaly Detection    SimpleDetectorCombination: combining outlier detectors by general purpose methods above                 N/A    :cite:`a-aggarwal2017outlier`
:mod:`combo.models.score_comb.aom`                                  Anomaly Detection    Average of Maximum (AOM): divide base detectors into subgroups to take the maximum, and then average    2015   :cite:`a-aggarwal2015theoretical`
:mod:`combo.models.score_comb.moa`                                  Anomaly Detection    Maximum of Average (MOA): divide base detectors into subgroups to take the average, and then maximize   2015   :cite:`a-aggarwal2015theoretical`
:class:`combo.models.detector_xgbod.XGBOD`                          Anomaly Detection    XGBOD: a semi-supervised combination framework for outlier detection                                    2018   :cite:`a-zhao2018xgbod`
:class:`combo.models.detector_lscp.LSCP`                            Anomaly Detection    Locally Selective Combination (LSCP)                                                                    2019   :cite:`a-zhao2019lscp`
==================================================================  ===================  ======================================================================================================  =====  ===========================================


**The comparison among selected implemented models** is made available below
(\ `Figure <https://raw.githubusercontent.com/yzhao062/combo/master/examples/compare_selected_classifiers.png>`_\ ,
`compare_selected_classifiers.py <https://github.com/yzhao062/combo/blob/master/examples/compare_selected_classifiers.py>`_\, `Interactive Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/combo/master>`_\ ).
For Jupyter Notebooks, please navigate to **"/notebooks/compare_selected_classifiers.ipynb"**.


.. image:: https://raw.githubusercontent.com/yzhao062/combo/master/examples/compare_selected_classifiers.png
   :target: https://raw.githubusercontent.com/yzhao062/combo/master/examples/compare_selected_classifiers.png
   :alt: Comparison of Selected Models


----


Development Status
^^^^^^^^^^^^^^^^^^

**combo** is currently **under development** as of Feb, 2020. A concrete plan has
been laid out and will be implemented in the next few months.

Similar to other libraries built by us, e.g., Python Outlier Detection Toolbox
(`pyod <https://github.com/yzhao062/pyod>`_),
**combo** is also targeted to be published in *Journal of Machine Learning Research (JMLR)*,
`open-source software track <http://www.jmlr.org/mloss/>`_. A demo paper has been presented in
*AAAI 2020* for progress update.

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

   api_cc
   api


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   about
   faq
   whats_new


----


.. rubric:: References

.. bibliography::
   :cited:
   :labelprefix: A
   :keyprefix: a-



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
