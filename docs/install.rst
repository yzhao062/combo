Installation
============

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
* joblib
* matplotlib (**optional for running examples**)
* numpy>=1.13
* numba>=0.35
* pyod
* scipy>=0.19.1
* scikit_learn>=0.20


**Note on Python 2**\ :
The maintenance of Python 2.7 will be stopped by January 1, 2020 (see `official announcement <https://github.com/python/devguide/pull/344>`_).
To be consistent with the Python change and combo's dependent libraries, e.g., scikit-learn,
**combo only supports Python 3.5+** and we encourage you to use
Python 3.5 or newer for the latest functions and bug fixes. More information can
be found at `Moving to require Python 3 <https://python3statement.org/>`_.

