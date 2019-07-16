Quick Start
===========

`"examples/classifier_comb_example.py" <https://github.com/yzhao062/combo/blob/master/examples/classifier_comb_example.py>`_
demonstrates the basic API of predicting with multiple classifiers. **It is noted that the API across all other algorithms are consistent/similar**.

#. Initialize a group of classifiers as base estimators

   .. code-block:: python


       from combo.models.classifier_comb import BaseClassiferAggregator

       # initialize a group of classifiers
       classifiers = [DecisionTreeClassifier(random_state=random_state),
                      LogisticRegression(random_state=random_state),
                      KNeighborsClassifier(),
                      RandomForestClassifier(random_state=random_state),
                      GradientBoostingClassifier(random_state=random_state)]


#. Initialize an aggregator class and pass in initialized classifiers for training

   .. code-block:: python


       # combine by averaging
       clf = BaseClassiferAggregator(classifiers)
       clf.fit(X_train, y_train)


#. Predict by averaging base classifier results and then evaluate

   .. code-block:: python


       # combine by averaging

       y_test_predicted = clf.predict(X_test, method='average')
       evaluate_print('Combination by avg  |', y_test, y_test_predicted)


#. Predict by maximizing base classifier results and then evaluate

   .. code-block:: python


       # combine by maximization

       y_test_predicted = clf.predict(X_test, method='maximization')
       evaluate_print('Combination by max  |', y_test, y_test_predicted)


#. See a sample output of classifier_comb_example.py

   .. code-block:: python


       Decision Tree       | Accuracy:0.9386, ROC:0.9383, F1:0.9521
       Logistic Regression | Accuracy:0.9649, ROC:0.9615, F1:0.973
       K Neighbors         | Accuracy:0.9561, ROC:0.9519, F1:0.9662
       Gradient Boosting   | Accuracy:0.9605, ROC:0.9524, F1:0.9699
       Random Forest       | Accuracy:0.9605, ROC:0.961, F1:0.9693

       Combination by avg  | Accuracy:0.9693, ROC:0.9677, F1:0.9763
       Combination by max  | Accuracy:0.9518, ROC:0.9312, F1:0.9642


-----