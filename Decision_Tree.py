

"""

@author: Ziba Dehghani

"""

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
import pandas as pd
import numpy as np
# Prepare the data data


path = "E:\\Ziba_Dehghani\\Results\\"
name = "result_2-2.xlsx"
data = pd.read_excel(path + name)
Cols = list(data.columns)
data = np.array(data)
data = pd.DataFrame(data)
data.fillna(1000,inplace=True)


Missing_Rows = []
for i in range(0, data.shape[0]):
    for j in range(0, data.shape[1]):
        if data.iloc[i][j] == 1000:
            Missing_Rows.append(i)
Missing_Rows = list(set(Missing_Rows))

#dist = np.linalg.norm(a-b)
def EluDis(a, b):
    for i in range(0, len(a)):
        if a.iloc[i] == 1000:
            a.iloc[i] = 0
            b.iloc[i] = 0
        if b.iloc[i] == 1000:
            a.iloc[i] = 0
            b.iloc[i] = 0
    dist = np.linalg.norm(a-b)
    return(dist)

for i in range(0, len(Missing_Rows)):
    missed_row = Missing_Rows[i]
    
    distances = []
    index = []
    for j in range(0, data.shape[0]):
        if missed_row!=j:
            dist = EluDis(data.iloc[missed_row], data.iloc[j])
            distances.append(dist)
            index.append(j)

    min_dist = min(distances)
    L = distances.index(min_dist)

    
    for k in range(0, len(data.iloc[missed_row])):
        if data.iloc[missed_row][k] == 1000:
            data.iloc[missed_row][k] = data.iloc[L][k]

data.columns = Cols            
   
"------------------------------------------------------------"
# Data Loader
X = data.iloc[:,1:-1]
y = data["degree"]

"------------------------------------------------------------"
# Feature selection


"------------------------------------------------------------"
# L1-based feature selection
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

lsvc = LinearSVC(C=0.2, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape

X_new = pd.DataFrame(X_new)
Cols = []
for i in range(0, X.shape[1]):
    for j in range(0, X_new.shape[1]):
        if list(X.iloc[:,i]) == list(X_new.iloc[:,j]):  
            Cols.append(X.columns[i])


X_new.columns = Cols

X= X_new






"------------------------------------------------------------"
def Decision_Tree_Ecvaluation(X, y, depth, test_size):
    from sklearn.model_selection import train_test_split
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=test_size)

    # Fit the classifier with max_depth=3
    clf = tree.DecisionTreeClassifier(max_depth= depth, random_state=1234)
    model = clf.fit(X, y)
    
    y_pred = clf.predict(testX)
    
    from sklearn.metrics import precision_recall_fscore_support
    macro = precision_recall_fscore_support(testY, y_pred, average='macro')
    macro_pres = macro[0]
    macro_rec = macro[1]
    macro_fs = macro[2]
    
    micro = precision_recall_fscore_support(testY, y_pred, average='micro')
    micro_pres = micro[0]
    micro_rec = micro[1]
    micro_fs = micro[2]  
    
    weighted = precision_recall_fscore_support(testY, y_pred, average='weighted')
    weighted_pres = weighted[0]
    weighted_rec = weighted[1]
    weighted_fs = weighted[2]
    
    Evals = [macro_pres, macro_rec, macro_fs, micro_pres, micro_rec, micro_fs,
             weighted_pres, weighted_rec, weighted_fs]
    
    
    df = pd.DataFrame(Evals)
    df = df.T

    df.columns = ["macro_pres", "macro_rec", "macro_fs", "micro_pres", "micro_rec", "micro_fs",
                  "weighted_pres", "weighted_rec", "weighted_fs"]

    return df




Decision_Tree_Ecvaluation(X = X, y=y,depth=11, test_size = 0.25)


def Robust_Validation(X, y, test_size, min_depth, max_depth):
    
    Criteria = []
    Depth = []
    for i in range(min_depth, max_depth +1):
        macro_micro_weighted = Decision_Tree_Ecvaluation(X, y, depth = i, test_size = test_size)
        Criteria.append(macro_micro_weighted)
        Depth.append(i)
    
    df = pd.concat(Criteria)
    df.index = Depth
    
    return df


B = Robust_Validation(X, y, test_size=0.25, min_depth=3, max_depth=20)

def Multi_Run(X, y, test_size, min_depth, max_depth, epoch):
    DF = []
    for i in range(epoch):
        B = Robust_Validation(X, y, test_size=test_size, min_depth=min_depth, max_depth=max_depth)
        DF.append(B)
    
    DF = pd.concat(DF)
    
    return DF
    


Q = Multi_Run(X, y, test_size=0.25, min_depth=3, max_depth=20, epoch=25)

Q.to_excel(path + "validation_4-4.xlsx")


clf = tree.DecisionTreeClassifier(max_depth= 11, random_state=1234)
model = clf.fit(X, y)
text_representation = tree.export_text(clf)
print(text_representation)





