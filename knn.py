import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from random import randint as rand
import sys


def knn(classificated, forClassification, k, distance, showPercentage = False):
    res = []
    classes = [i[1] for i in classificated.items()]
    classes = list(np.unique(classes))
    c = 1
    if showPercentage:
        print('Classification (knn):')
    for object in forClassification:
        k1 = k
        if showPercentage:
            percentage = round(c/len(forClassification)*100, 1)
            sys.stdout.write(f"\r\tIn progress: {percentage}% {'#' * int(percentage // 4)}{'_' * int(25 - percentage // 4)}")
        distances = []
        for classificatedObject in classificated:
            distances.append((distance(object, list(classificatedObject)), classificated[classificatedObject], classificatedObject))
        distances = sorted(distances, key=lambda x: x[0])
        classesCount = {}
        for i in classes:
            classesCount[i] = 0
        if k >= len(classificated):
            k1 = len(classificated)
        else:
            adding = 1
            while distances[k-1][0] == distances[k+adding-1][0]:
                adding += 1
            k1 += adding-1
        for i in range(k1):
            classesCount[distances[i][1]] += 1
        counts = [i[1] for i in classesCount.items()]
        definedClass = counts.index(max(counts))
        classificated[tuple(object)] = classes[definedClass]
        res.append(classes[definedClass])
        c += 1
    return res


def meanKNN(classificated, forClassification, k, distance, showPercentage = False):
    res = []
    classes = [i[1] for i in classificated.items()]
    classes = list(np.unique(classes))
    c = 1
    if showPercentage:
        print('Classification (mean knn):')
    for object in forClassification:
        k1 = k
        if showPercentage:
            percentage = round(c / len(forClassification) * 100, 1)
            sys.stdout.write(f"\r\tIn progress: {percentage}% {'#' * int(percentage // 4)}{'_' * int(25 - percentage // 4)}")
        distances = []
        for classificatedObject in classificated:
            distances.append((distance(object, list(classificatedObject)), classificated[classificatedObject], classificatedObject))
        distances = sorted(distances, key=lambda x: x[0])
        classesCount = {}
        for i in classes:
            classesCount[i] = 0
        if k >= len(classificated):
            k1 = len(classificated)
        else:
            adding = 1
            while distances[k-1][0] == distances[k + adding-1][0]:
                adding += 1
            k1 += adding - 1
        for i in range(k1):
            classesCount[distances[i][1]] += 1
        counts = [i[1] for i in classesCount.items()]
        for i in range(len(counts)):
            if counts[i] == 0:
                counts[i] = -1
        means = [0 for _ in range(len(classes))]
        for i in range(k1):
            means[classes.index(distances[i][1])] += distances[i][0]
        means = [means[i]/counts[i] for i in range(len(classes))]
        for i in range(len(means)):
            if means[i] == 0:
                means[i] = max(means)+1
        definedClass = means.index(min(means))
        classificated[tuple(object)] = classes[definedClass]
        res.append(classes[definedClass])
        c += 1
    return res


def hybridKNN(classificated, forClassification, k, distance, showPercentage = False):
    res = []
    classes = [i[1] for i in classificated.items()]
    classes = list(np.unique(classes))
    c = 1
    if showPercentage:
        print('Classification (hybrid knn):')
    for object in forClassification:
        k1 = k
        if showPercentage:
            percentage = round(c / len(forClassification) * 100, 1)
            sys.stdout.write(f"\rIn progress: {percentage}% {'#' * int(percentage // 4)}{'_' * int(25 - percentage // 4)}")
        distances = []
        for classificatedObject in classificated:
            distances.append(
                (distance(object, list(classificatedObject)), classificated[classificatedObject], classificatedObject))
        distances = sorted(distances, key=lambda x: x[0])
        classesCount = {}
        for i in classes:
            classesCount[i] = 0
        if k >= len(classificated):
            k1 = len(classificated)
        else:
            adding = 1
            while distances[k-1][0] == distances[k + adding-1][0]:
                adding += 1
            k1 += adding - 1
        for i in range(k1):
            classesCount[distances[i][1]] += 1
        counts = [i[1] for i in classesCount.items()]
        for i in range(len(counts)):
            if counts[i] == 0:
                counts[i] = -1
        if counts.count(max(counts)) > 1:
            maxClasses = []
            for i in range(len(counts)):
                maxClasses.append(i)
            means = [0 for _ in range(len(classes))]
            for i in range(len(counts)):
                if counts[i] == 0:
                    counts[i] = -1
            for i in range(k1):
                means[classes.index(distances[i][1])] += distances[i][0]
            means = [means[i] / max(counts) for i in range(len(maxClasses))]
            for i in range(len(means)):
                if means[i] == 0:
                    means[i] = max(means) + 1
            for i in range(len(means)):
                if means[i] == 0:
                    means[i] = max(means) + 1
            definedClassIndex = means.index(min(means))
            definedClass = maxClasses[definedClassIndex]
        else:
            definedClass = counts.index(max(counts))
        classificated[tuple(object)] = classes[definedClass]
        res.append(classes[definedClass])
        c += 1
    return res


def createFeatureList(df, classCol, id):
    columnList = df.columns
    res = []
    for column in columnList:
        if column != classCol:
            res.append(float(df[column][df.index == id]))
    return res


def getClassById(df, classCol, id):
    return df[classCol][df.index == id].item()


def createSets(df: pd.DataFrame, sizeOfTestSet: float, classCol: str, showPercentage = False):
    trainingSet = {}
    testSet, testClasses = [], []
    classes = list(set(df[classCol].tolist()))
    idList = df.index.tolist()
    size = len(idList)-1
    all = len(df)
    c = 1
    if showPercentage:
        print('Generating sets:')
    sizeOfTestSet = round(all * sizeOfTestSet)
    for i in range(sizeOfTestSet):
        if showPercentage:
            percentage = round(c / all * 100, 1)
            sys.stdout.write(f"\r\tIn progress: {percentage}% {'#' * int(percentage // 4)}{'_' * int(25 - percentage // 4)}")
        index = rand(0, size)
        testClasses.append(getClassById(df, classCol, idList[index]))
        testSet.append(createFeatureList(df, classCol, idList[index]))
        size -= 1
        del idList[index]
        c += 1
    for id in idList:
        if showPercentage:
            percentage = round(c / all * 100, 1)
            sys.stdout.write(f"\r\tIn progress: {percentage}% {'#' * int(percentage // 4)}{'_' * int(25 - percentage // 4)}")
        featureList = createFeatureList(df, classCol, id)
        clas = getClassById(df, classCol, id)
        trainingSet[tuple(featureList)] = clas
        c += 1
    return trainingSet, testSet, testClasses


def countAccuracy(classificated: list, testClasses: list):
    size = len(classificated)
    count = 0
    for i in range(size):
        if classificated[i] == testClasses[i]:
            count += 1
    return count/size
