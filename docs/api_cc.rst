API CheatSheet
==============

Full API Reference: (https://pycombo.readthedocs.io/en/latest/api.html). The
following APIs are applicable for most models for easy use.

* :func:`combo.models.base.BaseAggregator.fit`: Fit estimator. y is optional for unsupervised methods.
* :func:`combo.models.base.BaseAggregator.predict`: Predict on a particular sample once the estimator is fitted.
* :func:`combo.models.base.BaseAggregator.predict_proba`: Predict the probability of a sample belonging to each class once the estimator is fitted.

Helpful functions:

* :func:`combo.models.base.BaseAggregator.get_params`: Get the parameters of the model.
* :func:`combo.models.base.BaseAggregator.set_params`: Set the parameters of the model.
* Each base estimator can be accessed by calling clf[i] where i is the estimator index.


See base class definition below:

combo.models.base module
------------------------

.. automodule:: combo.models.base
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

