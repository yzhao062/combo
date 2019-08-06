Examples by Tasks
=================


**All implemented modes** are associated with examples, check
`"combo examples" <https://github.com/yzhao062/combo/blob/master/examples>`_
for more information.


----


Example of Stacking/DCS/DES
^^^^^^^^^^^^^^^^^^^^^^^^^^^


`"examples/classifier_stacking_example.py" <https://github.com/yzhao062/combo/blob/master/examples/classifier_stacking_example.py>`_
demonstrates the basic API of stacking (meta ensembling). `"examples/classifier_dcs_la_example.py" <https://github.com/yzhao062/combo/blob/master/examples/classifier_dcs_la_example.py>`_
demonstrates the basic API of Dynamic Classifier Selection by Local Accuracy. `"examples/classifier_des_la_example.py" <https://github.com/yzhao062/combo/blob/master/examples/classifier_des_la_example.py>`_
demonstrates the basic API of Dynamic Ensemble Selection by Local Accuracy.

It is noted **the basic API is consistent across all these models**.


#. Initialize a group of classifiers as base estimators

   .. code-block:: python


      # initialize a group of classifiers
      classifiers = [DecisionTreeClassifier(random_state=random_state),
                     LogisticRegression(random_state=random_state),
                     KNeighborsClassifier(),
                     RandomForestClassifier(random_state=random_state),
                     GradientBoostingClassifier(random_state=random_state)]


#. Initialize, fit, predict, and evaluate with Stacking

   .. code-block:: python


      from combo.models.classifier_stacking import Stacking

      clf = Stacking(base_estimators=classifiers, n_folds=4, shuffle_data=False,
                   keep_original=True, use_proba=False, random_state=random_state)

      clf.fit(X_train, y_train)
      y_test_predict = clf.predict(X_test)
      evaluate_print('Stacking | ', y_test, y_test_predict)


#. See a sample output of classifier_stacking_example.py

   .. code-block:: bash


      Decision Tree        | Accuracy:0.9386, ROC:0.9383, F1:0.9521
      Logistic Regression  | Accuracy:0.9649, ROC:0.9615, F1:0.973
      K Neighbors          | Accuracy:0.9561, ROC:0.9519, F1:0.9662
      Gradient Boosting    | Accuracy:0.9605, ROC:0.9524, F1:0.9699
      Random Forest        | Accuracy:0.9605, ROC:0.961, F1:0.9693

      Stacking             | Accuracy:0.9868, ROC:0.9841, F1:0.9899


----


Example of Classifier Combination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


`"examples/classifier_comb_example.py" <https://github.com/yzhao062/combo/blob/master/examples/classifier_comb_example.py>`_
demonstrates the basic API of predicting with multiple classifiers. **It is noted that the API across all other algorithms are consistent/similar**.

#. Initialize a group of classifiers as base estimators

   .. code-block:: python


      # initialize a group of classifiers
      classifiers = [DecisionTreeClassifier(random_state=random_state),
                     LogisticRegression(random_state=random_state),
                     KNeighborsClassifier(),
                     RandomForestClassifier(random_state=random_state),
                     GradientBoostingClassifier(random_state=random_state)]


#. Initialize, fit, predict, and evaluate with a simple aggregator (average)

   .. code-block:: python


      from combo.models.classifier_comb import SimpleClassifierAggregator

      clf = SimpleClassifierAggregator(classifiers, method='average')
      clf.fit(X_train, y_train)
      y_test_predicted = clf.predict(X_test)
      evaluate_print('Combination by avg   |', y_test, y_test_predicted)



#. See a sample output of classifier_comb_example.py

   .. code-block:: bash


      Decision Tree        | Accuracy:0.9386, ROC:0.9383, F1:0.9521
      Logistic Regression  | Accuracy:0.9649, ROC:0.9615, F1:0.973
      K Neighbors          | Accuracy:0.9561, ROC:0.9519, F1:0.9662
      Gradient Boosting    | Accuracy:0.9605, ROC:0.9524, F1:0.9699
      Random Forest        | Accuracy:0.9605, ROC:0.961, F1:0.9693

      Combination by avg   | Accuracy:0.9693, ROC:0.9677, F1:0.9763
      Combination by w_avg | Accuracy:0.9781, ROC:0.9716, F1:0.9833
      Combination by max   | Accuracy:0.9518, ROC:0.9312, F1:0.9642
      Combination by w_vote| Accuracy:0.9649, ROC:0.9644, F1:0.9728
      Combination by median| Accuracy:0.9693, ROC:0.9677, F1:0.9763


----


Example of Clustering Combination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


`"examples/cluster_comb_example.py" <https://github.com/yzhao062/combo/blob/master/examples/cluster_comb_example.py>`_
demonstrates the basic API of combining multiple base clustering estimators. `"examples/cluster_eac_example.py" <https://github.com/yzhao062/combo/blob/master/examples/cluster_eac_example.py>`_
demonstrates the basic API of Combining multiple clusterings using evidence accumulation (EAC).

#. Initialize a group of clustering methods as base estimators

   .. code-block:: python


      # Initialize a set of estimators
      estimators = [KMeans(n_clusters=n_clusters),
                    MiniBatchKMeans(n_clusters=n_clusters),
                    AgglomerativeClustering(n_clusters=n_clusters)]


#. Initialize a Clusterer Ensemble class and fit the model

   .. code-block:: python


      from combo.models.cluster_comb import ClustererEnsemble
      # combine by Clusterer Ensemble
      clf = ClustererEnsemble(estimators, n_clusters=n_clusters)
      clf.fit(X)


#. Get the aligned results

   .. code-block:: python


      # generate the labels on X
      aligned_labels = clf.aligned_labels_
      predicted_labels = clf.labels_



Example of Outlier Detector Combination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


`"examples/detector_comb_example.py" <https://github.com/yzhao062/combo/blob/master/examples/detector_comb_example.py>`_
demonstrates the basic API of combining multiple base outlier detectors.

#. Initialize a group of outlier detection methods as base estimators

   .. code-block:: python


      # Initialize a set of estimators
      detectors = [KNN(), LOF(), OCSVM()]


#. Initialize a simple averaging aggregator, fit the model, and make
   the prediction.

   .. code-block:: python


      from combo.models.detector combination import SimpleDetectorAggregator
      clf = SimpleDetectorAggregator(base_estimators=detectors)
      clf_name = 'Aggregation by Averaging'
      clf.fit(X_train)

      y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
      y_train_scores = clf.decision_scores_  # raw outlier scores

      # get the prediction on the test data
      y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
      y_test_scores = clf.decision_function(X_test)  # outlier scores


#. Evaluate the prediction using ROC and Precision @ Rank n.

   .. code-block:: python

      # evaluate and print the results
      print("\nOn Training Data:")
      evaluate_print(clf_name, y_train, y_train_scores)
      print("\nOn Test Data:")
      evaluate_print(clf_name, y_test, y_test_scores)

#. See sample outputs on both training and test data.

   .. code-block:: bash

      On Training Data:
      Aggregation by Averaging ROC:0.9994, precision @ rank n:0.95

      On Test Data:
      Aggregation by Averaging ROC:1.0, precision @ rank n:1.0


