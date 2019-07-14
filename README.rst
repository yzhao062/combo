combo: A python toolbox for combining machine learning models
=============================================================

**combo** is a comprehensive Python model combination toolkit for
fusing/aggregating/selecting multiple base ML estimators,
under supervised, unsupervised, and semi-supervised scenarios.
Model combination is an important task in
`ensemble learning <https://en.wikipedia.org/wiki/Ensemble_learning>`_,
but is often beyond the scope of ensemble learning. For instance, simple
averaging the results of the same classifiers with multiple runs is deemed as
a good way to eliminate the randomness in the classifier for a better stability.
Model combination has been widely used in data science competitions and
real-world tasks, such as Kaggle.


combo is featured for:

* **Unified APIs, detailed documentation, and interactive examples** across various algorithms.
* **Advanced models**, including dynamic classifier/ensemble selection.
* **Comprehensive coverage** for supervised, unsupervised, and semi-supervised scenarios.
* **Optimized performance with JIT and parallelization** when possible, using `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_.


----

combo will include various model combination frameworks:

* Simple methods: averaging, maximization, weighted averaging, thresholding
* Bucket methods: average of maximization, maximization of average
* Learning methods: stacking (build an additional classifier to learn base estimator weights)
* Selection methods: dynamic classifier/ensemble selection
* Other models


Development Status
^^^^^^^^^^^^^^^^^^

combo is currently **under development** as of July 14, 2019. A concrete plan has
been laid out and will be implemented in the next few months.

**Watch & Star** to get the latest update! Also feel free to send me an email (zhaoy@cmu.edu)
for suggestions and ideas.