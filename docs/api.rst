API Reference
=============


----


Classifier Combination
^^^^^^^^^^^^^^^^^^^^^^


:mod:`combo.models.classifier_comb`: a collection of classifier
combination methods.


.. toctree::
    :maxdepth: 4

    modules/classifier_comb/BaseClassifierAggregator
    modules/classifier_comb/SimpleClassifierAggregator

:mod:`combo.models.stacking`: Stacking (meta ensembling). Check this `introductory
article by Kaggle <http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/>`_.


----


Cluster Combination
^^^^^^^^^^^^^^^^^^^


:mod:`combo.models.cluster_comb`: a collection of cluster
combination methods.


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


