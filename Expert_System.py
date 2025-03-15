

"""

@author: Ziba Dehghani

"""

from Decision_Tree import *
# Real Data
import termcolor
import colorama
from colorama import Fore
from termcolor import colored



clf = tree.DecisionTreeClassifier(max_depth= 11, random_state=1234)
model = clf.fit(X, y)
text_representation = tree.export_text(clf)
print(text_representation)




def Label_Finder(y_predict):
    if y_predict == 1:
        print(colored('It is predicted that the patient will enter the Grade 1 pressure ulcer stage.', 'cyan'))
    elif y_predict == 2:
        print(colored('It is predicted that the patient will enter the Grade 2 pressure ulcer stage.', 'cyan'))
    elif y_predict == 3:
        print(colored('It is predicted that the patient will enter the Grade 3 pressure ulcer stage.', 'cyan'))
    elif y_predict == 4:
        print(colored('It is predicted that the patient will enter the Grade 4 pressure ulcer stage.', 'cyan'))
    elif y_predict == 5:
        print(colored('It is predicted that the patient will enter the DTI pressure ulcer stage.', 'cyan'))
    elif y_predict == 6:
        print(colored('It is predicted that the patient will enter the Ungradable pressure ulcer stage.', 'cyan'))



testPath = "E:\\Ziba_Dehghani\\"
testName = "Test.xlsx"
TestData = pd.read_excel(testPath+testName)
        
newPerson=TestData.iloc[0]
df=pd.DataFrame(newPerson)
df=df.T
y_predict = clf.predict(df)[0]

y_predict

def Pressure_sore_Prognosis(personal_features):
    pred = clf.predict(personal_features)[0]
    label = Label_Finder(pred)
    return label

Pressure_sore_Prognosis(df)








def Pressure_sore_Prognosis(personal_features):
    pred = clf.predict(personal_features)[0]
    label = Label_Finder(pred)
    return label

Pressure_sore_Prognosis(t)
    
    
    
    
    
    
    
    
    
    
    