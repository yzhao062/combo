API Reference
=============


----


Classifier Combination
^^^^^^^^^^^^^^^^^^^^^^


* :class:`combo.models.classifier_comb.SimpleClassifierAggregator`: a collection of classifier
  combination methods, e.g., average, median, and majority vote.
* :class:`combo.models.classifier_stacking.Stacking`: Stacking (meta ensembling). Check this `introductory
  article by Kaggle <http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/>`_.
* :class:`combo.models.classifier_dcs.DCS_LA`: Dynamic classifier selection (DCS) by local accuracy.
* :class:`combo.models.classifier_des.DES_LA`: Dynamic ensemble selection (DES) by local accuracy.


----


Cluster Combination
^^^^^^^^^^^^^^^^^^^


* :class:`combo.models.cluster_comb.ClustererEnsemble`: Clusterer Ensemble combines multiple base clustering estimators by alignment.
* :func:`combo.models.cluster_comb.clusterer_ensemble_scores`: Clusterer Ensemble on clustering results directly.
* :class:`combo.models.cluster_eac.EAC`: Combining multiple clusterings using evidence accumulation (EAC).


----


Outlier Detector Combination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* :class:`combo.models.detector_comb.SimpleDetectorAggregator`: a collection of
  outlier detector combination methods, e.g., average, median, and maximization.
  Refer `PyOD <https://github.com/yzhao062/pyod>`_ for more information.
* :class:`combo.models.detector_lscp.LSCP`: Locally Selective Combination of Parallel Outlier Ensembles (LSCP).

----


Score Combination
^^^^^^^^^^^^^^^^^


:mod:`combo.models.score_comb`: a collection of (raw) score
combination methods.

* :func:`combo.models.score_comb.average`
* :func:`combo.models.score_comb.maximization`
* :func:`combo.models.score_comb.median`
* :func:`combo.models.score_comb.majority_vote`
* :func:`combo.models.score_comb.aom`
* :func:`combo.models.score_comb.moa`


----


All Models
^^^^^^^^^^

.. toctree::
   :maxdepth: 4

   combo.models
   combo.utils


