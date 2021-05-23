# KNN
knn(classificated: dict, forClassification: list, k: int, distance: function, showPercentage = False)
meanKNN(classificated: dict, forClassification: list, k: int, distance: function, showPercentage = False)
hybridKNN(classificated: dict, forClassification: list, k: int, distance: function, showPercentage = False)
  classificated - dictionary of classificated objects (key - feature tuple of objects, value - class of object),
  forClassification - list of objects for classification features,
  k - number of nearest neighbours,
  distance - function which calculate distance between objects,
  showPercentage - showing the loading bar (default - False)
Returns list of getting classes.

createSets(df: pd.DataFrame, sizeOfTestSet: float, classCol: str, showPercentage = False) - devide preprocessed pandas.DataFrame to test and training set. DataFrame must consisits only features and classes
  df - DataFrame,
  sizeOfTestSet - size of test set relative to the entire DataFrame length,
  classCol - name of the column with classes,
  showPercentage - showing the loading bar (default - False)
Returns training dictionary, test list and list of classes in test list.

countAccuracy(classificated: list, testClasses: list) - count the accuracy of classification
  classificated - list of getted classes after classification,
  testClasses - list of real classes,
Returns accuracy of classification.
  
  

  
