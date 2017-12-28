import numpy as np
import pandas as pd

def main():
    #Pre-processing 数据预处理
    from sklearn.datasets import load_iris
    iris = load_iris()
    print(iris)
    print(len(iris["data"]))
    from sklearn.cross_validation import  train_test_split
    train_data, test_data,train_target,test_target = train_test_split(iris.data,iris.target,test_size=0.2,random_state=1)
    # Model 建模

    from sklearn import tree
    clf = tree.DecisionTreeClassifier(criterion = "entropy")
    clf.fit(train_data,train_target)
    y_pred = clf.predict(test_data)

    # Verify 验证--> 两种方式验证：　正确率　混淆矩阵
    from sklearn import metrics
    # 准确率
    print(metrics.accuracy_score(y_true=test_target,y_pred=y_pred))
    # 混淆矩阵
    print(metrics.confusion_matrix(y_true=test_target,y_pred=y_pred))

    with open("./data/tree.dot","w") as fw:
        tree.export_graphviz(clf,out_file =fw)
if __name__ == "__main__":
    main()