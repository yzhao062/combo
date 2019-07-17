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


#. Initialize an aggregator class and pass in combination methods

   .. code-block:: python


       # combine by averaging
       clf = SimpleClassifierAggregator(classifiers, method='average')
       clf.fit(X_train, y_train)


#. Predict by SimpleClassifierAggregator and then evaluate

   .. code-block:: python


       y_test_predicted = clf.predict(X_test)
       evaluate_print('Combination by avg   |', y_test, y_test_predicted)


#. See a sample output of classifier_comb_example.py

   .. code-block:: python


       Decision Tree        | Accuracy:0.9386, ROC:0.9383, F1:0.9521
       Logistic Regression  | Accuracy:0.9649, ROC:0.9615, F1:0.973
       K Neighbors          | Accuracy:0.9561, ROC:0.9519, F1:0.9662
       Gradient Boosting    | Accuracy:0.9605, ROC:0.9524, F1:0.9699
       Random Forest        | Accuracy:0.9605, ROC:0.961, F1:0.9693

       Combination by avg   | Accuracy:0.9693, ROC:0.9677, F1:0.9763
       Combination by w_avg | Accuracy:0.9781, ROC:0.9716, F1:0.9833
       Combination by max   | Accuracy:0.9518, ROC:0.9312, F1:0.9642
       Combination by w_vote| Accuracy:0.9649, ROC:0.9644, F1:0.9728


-----