# Decision Tree Learning
import itertools;
import pandas as pd;
import pylab as np
import matplotlib.pyplot as plt;
from sklearn.tree import DecisionTreeClassifier, export_graphviz;
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix;
from sklearn.model_selection import train_test_split;
from six import StringIO;
import pydotplus;
import matplotlib.image as mpimg;
from IPython.display import Image;

def getCleanDataframeAndGetAxes():
    # Import dataset to dataframe:
    dataFrame = pd.read_csv('dataset_einstein.csv', delimiter=';');
    # Removing empty rows:
    dataFrame = dataFrame.dropna();
    # Get all labels in Y variable:
    Y = dataFrame['SARS-Cov-2 exam result'].values;

    # Get all features in X variable:
    X = dataFrame[['Hemoglobin', 'Leukocytes', 'Basophils', 'Proteina C reativa mg/dL']].values;
    return dataFrame, X, Y;

def getTrainAndTestDataSet(X, Y): 
    # Partition into two parts: train and test (80/20)
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=.2, random_state=3);
    return XTrain, XTest, YTrain, YTest;

def getTreeAndModel(XTrain, YTrain): 
    treeLogic = DecisionTreeClassifier(criterion='entropy', max_depth=5);
    model = treeLogic.fit(XTrain, YTrain);
    return treeLogic, model;

def buildTreeImage(model, dotData, featuresNames, classNames, fileName = 'tree.png'):
    dotData = StringIO();
    # dotData = export_graphviz(my_tree_one=treeLogic, out_file=None, feature_names=featuresNames);
    export_graphviz(
        model,
        out_file=dotData,
        filled=True,
        feature_names=featuresNames,
        class_names=classNames,
        rounded=True,
        special_characters=True
    );
    graph = pydotplus.graph_from_dot_data(dotData.getvalue());
    Image(graph.create_png());
    graph.write_png(fileName);
    Image(fileName);

def getMoreImportantFeatures(model, X):
    # More important feature:
    importances = model.feature_importances_;
    index = np.argsort(importances)[::-1];
    print('Feature ranking: ');

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, index[f], importances[index[f]]));

    f, ax = plt.subplots(figsize=(11, 9));
    plt.title('Feature ranking', fontsize = 20);
    plt.bar(
        range(X.shape[1]),
        importances[index],
        color='b',
        align='center'
    );
    plt.xticks(range(X.shape[1]), index);
    plt.xlim([-1, X.shape[1]]);
    plt.ylabel('importance', fontsize = 18);
    plt.show();

    # Features index:
    # 0 - 'Hemoglobin'
    # 1 - 'Leukocytes'
    # 2 - 'Basophis'
    # 3 - 'Proteina C reativa mg/dL'

def validateModelWithTestDataset(model, XTest, YTest):
    # Apply model into test dataset
    YPredict = model.predict(XTest);
    # Model metrics:
    print('Acurácia da árvore: ', accuracy_score(YTest, YPredict));
    print(classification_report(YTest, YPredict));
    return YPredict;

def plotConfusionMatrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap=plt.cm.Blues):
    # This function prints and plots the confusion matrix.
    plt.imshow(cm, interpolation='nearest', cmap=cmap);
    plt.title(title);
    plt.colorbar();
    tickMarks = np.arange(len(classes));
    plt.xticks(tickMarks, classes, rotation=45);
    plt.yticks(tickMarks, classes);

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis];
        print('Normalized confusion matrix');
    else:
        print('Consusion matrix, without normalization');
    
    thresh = cm.max() / 2;
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black');

    plt.tight_layout();
    plt.ylabel('Rótulo real');
    plt.xlabel('Rótulo prevista');
    plt.show();

dataFrame, X, Y = getCleanDataframeAndGetAxes();
XTrain, XTest, YTrain, YTest = getTrainAndTestDataSet(X, Y);
treeLogic, model = getTreeAndModel(XTrain, YTrain);
featuresNames = ['Hemoglobin', 'Leukocytes', 'Basophils', 'Proteina C reativa mg/dL'];
classNames = model.classes_;
# getMoreImportantFeatures(model, X);

YPredict = validateModelWithTestDataset(model, XTest, YTest);

confusionMatrix = confusion_matrix(YTest, YPredict);
plt.figure();

plotConfusionMatrix(confusionMatrix, classes=classNames);
