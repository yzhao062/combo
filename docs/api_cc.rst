API CheatSheet
==============

Full API Reference: (https://pycombo.readthedocs.io/en/latest/api.html).
The following APIs are consistent for most of the models
(API Cheatsheet: https://pycombo.readthedocs.io/en/latest/api_cc.html).

* :func:`combo.models.base.BaseAggregator.fit`: Fit estimator. y is optional for unsupervised methods.
* :func:`combo.models.base.BaseAggregator.predict`: Predict on a particular sample once the estimator is fitted.
* :func:`combo.models.base.BaseAggregator.predict_proba`: Predict the probability of a sample belonging to each class once the estimator is fitted.
* :func:`combo.models.base.BaseAggregator.fit_predict`: Fit estimator and predict on X. y is optional for unsupervised methods.

Helpful functions:

* :func:`combo.models.base.BaseAggregator.get_params`: Get the parameters of the model.
* :func:`combo.models.base.BaseAggregator.set_params`: Set the parameters of the model.
* Each base estimator can be accessed by calling clf[i] where i is the estimator index.

For raw score combination (after the score matrix is generated),
use individual methods from
`"score_comb.py" <https://github.com/yzhao062/combo/blob/master/combo/models/score_comb.py>`_ directly.
Raw score combination API: (https://pycombo.readthedocs.io/en/latest/api.html#score-combination).


See base class definition below:

combo.models.base module
------------------------

.. automodule:: combo.models.base
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

