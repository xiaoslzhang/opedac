# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from copy import copy
from sklearn import ensemble, preprocessing
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from sklearn.model_selection import KFold, cross_val_score
from keras.layers import Dense
from sklearn.pipeline import Pipeline
from keras.layers import Dropout
from sklearn.ensemble import RandomForestClassifier
from keras import backend as kb
from keras.callbacks import ModelCheckpoint
from collections import Counter
from sklearn import cross_validation
import sys

if len(sys.argv) != 6:
    print("python randomForest_classifier_deepLearning_regression.py tra_features_file outlier_file label_file val_\
    features_file output_file ")
    exit()

x_train_file = sys.argv[1]
outlier_file = sys.argv[2]
y_train_file = sys.argv[3]
x_validation_file = sys.argv[4]
result_file = sys.argv[5]

x_train_data = pd.read_csv(x_train_file, encoding='utf-8', index_col="author_name")
data = pd.DataFrame(x_train_data)
outlier = pd.read_csv(outlier_file,index_col="AUTHOR", encoding="utf-8").index
author_tmp_list = list(data.index)
for a in outlier:
    author_tmp_list.remove(a)
data = data[data.index.isin(author_tmp_list)]
test_x = data
x_train = test_x
x_train = test_x.as_matrix(columns=None)

y_train_data = pd.read_csv(y_train_file, encoding='utf-8', index_col="AUTHOR")
data2 = pd.DataFrame(y_train_data)
data2 = data2[data2.index.isin(author_tmp_list)]
y_train = data2
y_zero_train = copy(y_train)
y_train =y_train.as_matrix(columns=None)

x_validation_data = pd.read_csv(x_validation_file, encoding='utf-8', index_col="author_name")
data3 = pd.DataFrame(x_validation_data)
x_validation = data3
x_validation = x_validation.as_matrix(columns=None)

normal_func = preprocessing.StandardScaler().fit(x_train)
x_train = normal_func.transform(x_train)
x_validation = normal_func.transform(x_validation)


def my_loss(y_true, y_prediction):
    return kb.mean(kb.abs(y_prediction - y_true)/kb.clip(kb.maximum(y_prediction,y_true),kb.epsilon(), np.inf),axis=-1)


def baseline_model():
    b_model = Sequential()
    b_model.add(Dense(336, input_shape=(336,), init='normal', activation='relu'))
    b_model.add(Dropout(0.2))
    b_model.add(Dense(168, init='normal', activation='relu'))
    b_model.add(Dropout(0.2))
    b_model.add(Dense(1, init='normal',activation='relu'))
    b_model.compile(loss=my_loss, optimizer='adam', metrics=['accuracy'])
    return b_model

# Firstly, classify dataset to zero set and nonzero set:
# x_train,y_train,x_validation.
# while label is True, value is 1
y_zero_train = [v == 0 for v in y_zero_train.CITATION]

clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_split=20, min_weight_fraction_leaf=0.0,
            n_estimators=1000, n_jobs=30, oob_score=False, random_state=1600,
            verbose=0, warm_start=False)
# clf.fit(Cx_train,Cy_train)
# clf.fit(tmp_x_train,tmp_y_train)
clf.fit(x_train,y_zero_train)
# class_score = clf.score(Cx_test,Cy_test)
# validation_class = clf.predict(x_validation)
# print(class_score)

validation_class = clf.predict_proba(x_validation)

# Count of zero should be 67221
find_zero_split = list(copy(validation_class))
find_zero_split.sort(key=lambda x: x[0])
zero_split = find_zero_split[67221 - 1][0]
print("split number is " + str(zero_split))
validation_class = [False if i[0] > zero_split else True for i in validation_class]

# score = 0.7685 while number of estimator is 1000.
# score = 0.76849 while number of estimator is 2000.

print(Counter(validation_class))
# There are 10871 rows value 0

temp = pd.DataFrame(x_train,index=y_zero_train)
x_train = temp[temp.index == False].values
temp = pd.DataFrame(y_train,index=y_zero_train)
y_train = temp[temp.index == False].values

model = baseline_model()
model.fit(x_train,y_train,epochs=500,batch_size=2048)#,callbacks=callbacks_list)
y_prediction_validation = model.predict(x_validation)

for index,z in enumerate(validation_class):
    if z:
        y_prediction_validation[index] = 0

with open(result_file, 'wb') as wf:
    variable_name = 'authorname' + '\tcitation'
    wf.write(variable_name.encode(encoding = 'utf-8'))
    wf.write('\r\n'.encode(encoding = 'utf-8'))
    with open(x_validation_file,'rb') as rf:
        lines = rf.readlines()
        lines = lines[1:]
        cnt = 0
        for line in lines:
            line = line.decode(encoding = 'utf-8').strip().split(',')
            temp = int(y_prediction_validation[cnt])
            info = line[0] + '\t' + str(temp)
            wf.write(info.encode(encoding = 'utf-8'))
            wf.write('\r\n'.encode(encoding = 'utf-8'))
            cnt = cnt + 1
