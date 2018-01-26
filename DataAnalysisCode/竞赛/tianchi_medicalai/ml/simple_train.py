from sklearn import svm, linear_model, tree
from sklearn.naive_bayes import GaussianNB
import csv
import numpy as np
import random


def main():
    data_num = 1918
    train_idx = random.sample(range(data_num), int(1 * data_num))
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open('/home/cangzhu/data/first_train_csv/train_data.csv', 'r') as data_file:
        data_reader = csv.reader(data_file, delimiter=',', quotechar='|')
        for i, row in enumerate(data_reader):
            if i in train_idx:
                x_train.append(np.asarray(row, 'int'))
            else:
                x_test.append(np.asarray(row, 'int'))
    with open('/home/cangzhu/data/first_train_csv/train_label.csv', 'r') as data_file:
        label_reader = csv.reader(data_file, delimiter=',', quotechar='|')
        for i, row in enumerate(label_reader):
            if i in train_idx:
                y_train.append(row[0])
            else:
                y_test.append(row[0])

    y_train = np.asarray(y_train, 'int')
    y_test = np.asarray(y_test, 'int')
    # clf = svm.SVC()
    # clf = linear_model.LogisticRegression()
    clf = GaussianNB()
    # clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    # result = clf.predict(x_test)
    # print((result == y_test).sum() / len(result))
    # idxx = (y_test == 1)
    # print((result[idxx] == y_test[idxx]).sum() / idxx.sum())

    # negative_x = []
    # negative_y = []
    # with open('/home/cangzhu/data/first_train_csv/val_data.csv', 'r') as data_file:
    #     data_reader = csv.reader(data_file, delimiter=',', quotechar='|')
    #     for row in data_reader:
    #         negative_x.append(np.asarray(row, 'int'))
    #
    # with open('/home/cangzhu/data/first_train_csv/val_label.csv', 'r') as data_file:
    #     label_reader = csv.reader(data_file, delimiter=',', quotechar='|')
    #     for i, row in enumerate(label_reader):
    #         negative_y.append(row[0])
    # negative_y = np.asarray(negative_y, 'int')
    # print(negative_x[0].shape)
    # result = clf.predict(negative_x)
    # print((result == negative_y).sum() / len(result))

    val_x = []
    val_y = []
    with open('/home/cangzhu/data/first_train_csv/val_data.csv', 'r') as data_file:
        data_reader = csv.reader(data_file, delimiter=',', quotechar='|')
        for row in data_reader:
            val_x.append(np.asarray(row, 'int'))

    with open('/home/cangzhu/data/first_train_csv/val_label.csv', 'r') as data_file:
        label_reader = csv.reader(data_file, delimiter=',', quotechar='|')
        for i, row in enumerate(label_reader):
            val_y.append(row[0])
    val_y = np.asarray(val_y, 'int')
    print(val_x[0].shape)
    result = clf.predict(val_x)
    prob = clf.predict_proba(val_x)
    print((result == val_y).sum() / len(result))
    tp = (result[result == val_y] == 1).sum()
    fp = (result[result != val_y] == 1).sum()
    tn = (result[result == val_y] == 0).sum()
    fn = (result[result != val_y] == 0).sum()
    print('tp:%d, fp:%d, tn:%d, fn:%d' % (tp, fp, tn, fn))
    print(prob)
    print(val_y)

    test_x = []
    with open('/home/cangzhu/data/first_test_csv/test_data.csv', 'r') as data_file:
        data_reader = csv.reader(data_file, delimiter=',', quotechar='|')
        for row in data_reader:
            test_x.append(row)
    test_x = np.asarray(test_x)
    test_x_name = test_x[:, 0]
    test_x = test_x[:, 1::].astype('int')
    print(test_x_name)
    result = clf.predict(test_x)
    prob = clf.predict_proba(test_x)
    selected_name = test_x_name[(result == 1)]
    selected_prob = prob[:, 1][(result == 1)]

    cord_info = []
    with open('/home/cangzhu/data/first_test_data/csv/c_out.csv', 'r') as cord_file:
        data_reader = csv.reader(cord_file, delimiter=',', quotechar='|')
        for row in data_reader:
            cord_info.append(row)
    cord_info = np.asarray(cord_info)
    cord_name = list(cord_info[:, 0])

    with open('/home/cangzhu/data/result.csv', 'w') as res:
        res_writer = csv.writer(res, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        res_writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
        for i, name in enumerate(selected_name):
            name += '.png'
            idx = cord_name.index(name)
            real_name = name.split('-')[0] + '-' + name.split('-')[1]
            w = list(cord_info[idx, 1:-1])
            w.insert(0, real_name)
            w.append(selected_prob[i])
            res_writer.writerow(w)


if __name__ == '__main__':
    main()
