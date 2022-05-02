#eece 490Project

#imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split



#Needed columns
all_columns=['FTHG','FTAG','HTHG','HTAG','HS','AS','HST','AST','HC','AC','HF','AF','HY','AY','HR','AR','Div','HomeTeam','AwayTeam','FTR','HTR','Date']
num_columns=['FTHG','FTAG','HTHG','HTAG','HS','AS','HST','AST','HC','AC','HF','AF','HY','AY','HR','AR']
cat_columns=['Div','HomeTeam','AwayTeam','FTR','HTR','Date']





#data
dfseason0910=pd.read_csv("C:\\Users\\Owner\\Desktop\\FolderJem3a\\Spring2022\\EECE490\\project\\laliga\\season-0910_csv.csv")
dfseason1011=pd.read_csv("C:\\Users\\Owner\\Desktop\\FolderJem3a\\Spring2022\\EECE490\\project\\laliga\\season-1011_csv.csv")
dfseason1213=pd.read_csv("C:\\Users\\Owner\\Desktop\\FolderJem3a\\Spring2022\\EECE490\\project\\laliga\\season-1213_csv.csv")
dfseason1314=pd.read_csv("C:\\Users\\Owner\\Desktop\\FolderJem3a\\Spring2022\\EECE490\\project\\laliga\\season-1314_csv.csv")
dfseason1415=pd.read_csv("C:\\Users\\Owner\\Desktop\\FolderJem3a\\Spring2022\\EECE490\\project\\laliga\\season-1415_csv.csv")
dfseason1516=pd.read_csv("C:\\Users\\Owner\\Desktop\\FolderJem3a\\Spring2022\\EECE490\\project\\laliga\\season-1516_csv.csv")
dfseason1617=pd.read_csv("C:\\Users\\Owner\\Desktop\\FolderJem3a\\Spring2022\\EECE490\\project\\laliga\\season-1617_csv.csv")
dfseason1718=pd.read_csv("C:\\Users\\Owner\\Desktop\\FolderJem3a\\Spring2022\\EECE490\\project\\laliga\\season-1718_csv.csv")
dfseason1819=pd.read_csv("C:\\Users\\Owner\\Desktop\\FolderJem3a\\Spring2022\\EECE490\\project\\laliga\\season-1819_csv.csv")
dfseason1920=pd.read_csv("C:\\Users\\Owner\\Desktop\\FolderJem3a\\Spring2022\\EECE490\\project\\laliga\\season-1920_csv.csv")





#Preparing the data

#drop unneded collumns from each data frame alone since they do not have the same collumns exactly
allData=[dfseason0910,dfseason1011,dfseason1213,dfseason1314,dfseason1415,dfseason1516,dfseason1617,dfseason1718,dfseason1819,dfseason1920]
for i in range (len (allData)):
    allData[i].drop(allData[i].columns.difference(all_columns), 1, inplace=True)

#Merge all data frames into only one
df=dfseason0910
for i in range (1,len(allData)):
    df=df.append(allData[i].iloc[:])

print(df)
#Needed new columns:
#HomeTeam FINIT
#AwayTeam FINIT
#FTR - Full Time Result (H=Home Win, D=Draw, A=Away Win) FINIT
#HTGD - Home team goal difference FINIT
#ATGD - Away team goal difference FINIT
#HTP - Home team points cumulative points FINIT
#ATP - Away team points cumulative points FINIT
#From - Last 10 match form FINIT

new_columns_num=['HTGD','ATGD','HTP','ATP','ATForm','HTForm']
new_columns_cat=['HomeTeam','AwayTeam','FTR']

#goald diffrence:
    
#HTGD - Home team goal difference cumulative
df['HTGD']=df[df['HomeTeam']==df['HomeTeam'].iloc[0]]['FTHG'].iloc[0]-df['FTAG'].iloc[0]
for i in range (1,len(df)):
    df['HTGD'].iloc[i]=-1

for i in range (1,len(df)):
    tempDF=df[df['HomeTeam']==df['HomeTeam'].iloc[i]]
    for j in range (1,len(tempDF)):
        if (tempDF['HTGD'].iloc[j]==-1):
            df['HTGD'].iloc[i]=tempDF['HTGD'].iloc[j-1]-df['FTAG'].iloc[i]+df['FTHG'].iloc[i]
            if (tempDF['HTGD'].iloc[j-1]==-1):
                df['HTGD'].iloc[i]=df['HTGD'].iloc[i]+1
            break
# print(df[df['HomeTeam']=='Real Madrid'][['HomeTeam','AwayTeam','FTHG','FTAG','HTGD']])
# print(df[df['HomeTeam']=='Mallorca'][['HomeTeam','AwayTeam','FTHG','FTAG','HTGD']])


#ATGD - away team goal difference cumulative
df['ATGD']=df[df['AwayTeam']==df['AwayTeam'].iloc[0]]['FTAG'].iloc[0]-df['FTHG'].iloc[0]
for i in range (1,len(df)):
    df['ATGD'].iloc[i]=0

for i in range (1,len(df)):
    tempDF=df[df['AwayTeam']==df['AwayTeam'].iloc[i]]
    for j in range (1,len(tempDF)):
        if (tempDF['ATGD'].iloc[j]==0):
            df['ATGD'].iloc[i]=tempDF['ATGD'].iloc[j-1]-df['FTHG'].iloc[i]+df['FTAG'].iloc[i]
            if (tempDF['ATGD'].iloc[j-1]==-1):
                df['ATGD'].iloc[i]=df['ATGD'].iloc[i]+1
            break
# print(df[df['AwayTeam']=='La Coruna'][['HomeTeam','AwayTeam','FTHG','FTAG','HTGD','ATGD']])
# print(df[df['AwayTeam']=='Mallorca'][['HomeTeam','AwayTeam','FTHG','FTAG','HTGD','ATGD']])



#Cumulative points:
#HTP - Home team  cumulative points
if (df[df['HomeTeam']==df['HomeTeam'].iloc[0]]['FTR'].iloc[0]=='H'):
    df['HTP']=3
elif (df[df['HomeTeam']==df['HomeTeam'].iloc[0]]['FTR'].iloc[0]=='A'):
    df['HTP']=0
elif (df[df['HomeTeam']==df['HomeTeam'].iloc[0]]['FTR'].iloc[0]=='D'):
    df['HTP']=1

for i in range (1,len(df)):
    df['HTP'].iloc[i]=-1


#From - Last 10 match form
#Home Team From - Last 10 match form
if (df[df['HomeTeam']==df['HomeTeam'].iloc[0]]['FTR'].iloc[0]=='H'):
    df['HTForm']=3
elif (df[df['HomeTeam']==df['HomeTeam'].iloc[0]]['FTR'].iloc[0]=='A'):
    df['HTForm']=0
elif (df[df['HomeTeam']==df['HomeTeam'].iloc[0]]['FTR'].iloc[0]=='D'):
    df['HTForm']=1
for i in range (1,len(df)):
    df['HTForm'].iloc[i]=-1

for i in range (1,len(df)):
    tempDF=df[df['HomeTeam']==df['HomeTeam'].iloc[i]]
    for j in range (0,len(tempDF)):
        if (tempDF['HTP'].iloc[j]==-1):
            if (df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['FTR'].iloc[j]=='H'):
                df['HTP'].iloc[i]=df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['HTP'].iloc[j-1]+3
                if(df['HTP'].iloc[i]==2):
                    df['HTP'].iloc[i]=3
                if (j>=10):
                     df['HTForm'].iloc[i]=df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['HTP'].iloc[j-1]-df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['HTP'].iloc[j-10]+3
                else:
                    if(j==0):
                        df['HTForm'].iloc[i]= df['HTP'].iloc[i]
                    else:
                        df['HTForm'].iloc[i]=df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['HTP'].iloc[j-1].sum()+3
            elif (df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['FTR'].iloc[j]=='A'):
                df['HTP'].iloc[i]=df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['HTP'].iloc[j-1]
                if(df['HTP'].iloc[i]==-1):
                    df['HTP'].iloc[i]=0
                if (j>=10):
                    df['HTForm'].iloc[i]=df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['HTP'].iloc[j-1]-df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['HTP'].iloc[j-10]
                else:
                    if(j==0):
                        df['HTForm'].iloc[i]= df['HTP'].iloc[i]
                    else:
                        df['HTForm'].iloc[i]=df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['HTP'].iloc[j-1].sum()
            elif (df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['FTR'].iloc[j]=='D'):
                df['HTP'].iloc[i]=df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['HTP'].iloc[j-1]+1
                if(df['HTP'].iloc[i]==0):
                    df['HTP'].iloc[i]=1
                if (j>=10):
                    df['HTForm'].iloc[i]=df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['HTP'].iloc[j-1]-df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['HTP'].iloc[j-10]+1
                else:
                    if(j==0):
                        df['HTForm'].iloc[i]= df['HTP'].iloc[i]
                    else:
                        df['HTForm'].iloc[i]=df[df['HomeTeam']==df['HomeTeam'].iloc[i]]['HTP'].iloc[j-1].sum()+1
            break
# print(df[df['HomeTeam']=='Mallorca'][['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTP']])
# print(df[df['HomeTeam']=='La Coruna'][['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTP']])
# print(df[df['HomeTeam']=='Villarreal'][['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTP']])
# print(df[df['HomeTeam']=='Xerez'][['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTP']])


#ATP - Away team points cumulative points
if (df[df['AwayTeam']==df['AwayTeam'].iloc[0]]['FTR'].iloc[0]=='H'):
    df['ATP']=0
elif (df[df['AwayTeam']==df['AwayTeam'].iloc[0]]['FTR'].iloc[0]=='A'):
    df['ATP']=3
elif (df[df['AwayTeam']==df['AwayTeam'].iloc[0]]['FTR'].iloc[0]=='D'):
    df['ATP']=1

for i in range (1,len(df)):
    df['ATP'].iloc[i]=-1

#From - Last 10 match form
#Away Team From - Last 10 match form
if (df[df['AwayTeam']==df['AwayTeam'].iloc[0]]['FTR'].iloc[0]=='H'):
    df['ATForm']=0
elif (df[df['AwayTeam']==df['AwayTeam'].iloc[0]]['FTR'].iloc[0]=='A'):
    df['ATForm']=3
elif (df[df['AwayTeam']==df['AwayTeam'].iloc[0]]['FTR'].iloc[0]=='D'):
    df['ATForm']=1
for i in range (1,len(df)):
    df['ATForm'].iloc[i]=-1

for i in range (1,len(df)):
    tempDF=df[df['AwayTeam']==df['AwayTeam'].iloc[i]]
    for j in range (0,len(tempDF)):
        if (tempDF['ATP'].iloc[j]==-1):
            if (df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['FTR'].iloc[j]=='H'):
                df['ATP'].iloc[i]=df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['ATP'].iloc[j-1]
                if(df['ATP'].iloc[i]==-1):
                    df['ATP'].iloc[i]=0
                if (j>=10):
                    df['ATForm'].iloc[i]=df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['ATP'].iloc[j-1]-df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['ATP'].iloc[j-10]
                else:
                    if(j==0):
                        df['ATForm'].iloc[i]= df['ATP'].iloc[i]
                    else:
                        df['ATForm'].iloc[i]=df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['ATP'].iloc[j-1].sum()
            elif (df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['FTR'].iloc[j]=='A'):
                df['ATP'].iloc[i]=df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['ATP'].iloc[j-1]+3
                if(df['ATP'].iloc[i]==2):
                    df['ATP'].iloc[i]=3
                if (j>=10):
                    df['ATForm'].iloc[i]=df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['ATP'].iloc[j-1]-df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['ATP'].iloc[j-10]+3
                else:
                    if(j==0):
                        df['ATForm'].iloc[i]= df['ATP'].iloc[i]
                    else:
                        df['ATForm'].iloc[i]=df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['ATP'].iloc[j-1].sum()+3
            elif (df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['FTR'].iloc[j]=='D'):
                df['ATP'].iloc[i]=df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['ATP'].iloc[j-1]+1
                if(df['ATP'].iloc[i]==0):
                    df['ATP'].iloc[i]=1
                if (j>=10):
                    df['ATForm'].iloc[i]=df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['ATP'].iloc[j-1]-df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['ATP'].iloc[j-10]+1
                else:
                    if(j==0):
                        df['ATForm'].iloc[i]= df['ATP'].iloc[i]
                    else:
                        df['ATForm'].iloc[i]=df[df['AwayTeam']==df['AwayTeam'].iloc[i]]['ATP'].iloc[j-1].sum()+1
            break
# print(df[df['AwayTeam']=='Mallorca'][['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTP','ATP']])
# print(df[df['AwayTeam']=='La Coruna'][['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTP','ATP']])
print(df[df['HomeTeam']=='Xerez'][['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTP','ATP','HTForm']])
    
    
    
#HomeTeam FINIT
#AwayTeam FINIT
#FTR - Full Time Result (H=Home Win, D=Draw, A=Away Win) FINIT
#HTGD - Home team goal difference FINIT
#ATGD - Away team goal difference FINIT
#HTP - Home team points cumulative points FINIT
#ATP - Away team points cumulative points FINIT
#ATForm - Last 10 match form FINIT
#HTForm - Last 10 match form FINIT
for i in range (len (df)):
     df['HTGD'].iloc[i]=df['HTGD'].iloc[i]-df['FTHG'].iloc[i]+df['FTAG'].iloc[i]
     df['ATGD'].iloc[i]=df['ATGD'].iloc[i]-df['FTAG'].iloc[i]+df['FTHG'].iloc[i]
     if (df['FTR'].iloc[i]=='H'):
         df['HTP'].iloc[i]=df['HTP'].iloc[i]-3
         df['HTForm'].iloc[i]=df['HTForm'].iloc[i]-3
     elif (df['FTR'].iloc[i]=='A'):
         df['ATP'].iloc[i]=df['ATP'].iloc[i]-3
         df['ATForm'].iloc[i]=df['ATForm'].iloc[i]-3
     elif (df['FTR'].iloc[i]=='D'):
         df['ATP'].iloc[i]=df['ATP'].iloc[i]-1
         df['HTP'].iloc[i]=df['HTP'].iloc[i]-1
         df['ATForm'].iloc[i]=df['ATForm'].iloc[i]-1
         df['HTForm'].iloc[i]=df['HTForm'].iloc[i]-1
         

# Visualising distribution of data
from pandas.plotting import scatter_matrix

#HTGD - Home team goal difference
#ATGD - away team goal difference
#HTP - Home team points
#ATP - Away team points
#HTForm Last 10 Games points of Home team
#ATForm Last 10 Games points of away team

#scatter_matrix(df[['HTGD','ATGD','HTP','ATP','HTForm','ATForm']], figsize=(10,10))
    

    
    

    
    
    
    
    
    
# # Convert the index of the DataFrame to datetime
# df.index = pd.to_datetime(df.Date)

#normalising numeric inputs
scaler = MinMaxScaler()
df[new_columns_num] = scaler.fit_transform(df[new_columns_num])
#df[num_columns].describe()

#Transforming categorical data into numerical using one-hot encoding
le = LabelEncoder()
for i in range (len (new_columns_cat)):
    df[new_columns_cat[i]]= le.fit_transform(df[new_columns_cat[i]])


import matplotlib.pyplot as plt
arFTHG=[0 for i in range (11)]
for i in range (len(df)):
    if (df['FTHG'].iloc[i]==0):
        arFTHG[0]=arFTHG[0]+1
    elif (df['FTHG'].iloc[i]==1):
        arFTHG[1]=arFTHG[1]+1
    elif (df['FTHG'].iloc[i]==2):
        arFTHG[2]=arFTHG[2]+1
    elif (df['FTHG'].iloc[i]==3):
        arFTHG[3]=arFTHG[3]+1
    elif (df['FTHG'].iloc[i]==4):
        arFTHG[4]=arFTHG[4]+1
    elif (df['FTHG'].iloc[i]==5):
        arFTHG[5]=arFTHG[5]+1
    elif (df['FTHG'].iloc[i]==6):
        arFTHG[6]=arFTHG[6]+1
    elif (df['FTHG'].iloc[i]==7):
        arFTHG[7]=arFTHG[7]+1
    elif (df['FTHG'].iloc[i]==8):
        arFTHG[8]=arFTHG[8]+1
    elif (df['FTHG'].iloc[i]==9):
        arFTHG[9]=arFTHG[9]+1
    elif (df['FTHG'].iloc[i]==10):
        arFTHG[10]=arFTHG[10]+1

arFTAG=[0 for i in range (9)]
for i in range (len(df)):
    if (df['FTAG'].iloc[i]==0):
        arFTAG[0]=arFTAG[0]+1
    elif (df['FTAG'].iloc[i]==1):
        arFTAG[1]=arFTAG[1]+1
    elif (df['FTAG'].iloc[i]==2):
        arFTAG[2]=arFTAG[2]+1
    elif (df['FTAG'].iloc[i]==3):
        arFTAG[3]=arFTAG[3]+1
    elif (df['FTAG'].iloc[i]==4):
        arFTAG[4]=arFTAG[4]+1
    elif (df['FTAG'].iloc[i]==5):
        arFTAG[5]=arFTAG[5]+1
    elif (df['FTAG'].iloc[i]==6):
        arFTAG[6]=arFTAG[6]+1
    elif (df['FTAG'].iloc[i]==7):
        arFTAG[7]=arFTAG[7]+1
    elif (df['FTAG'].iloc[i]==8):
        arFTAG[8]=arFTAG[8]+1

print(arFTHG)
print(arFTAG)
avHTwin=df[df['FTR']==2]['FTHG'].mean()
avATwin=df[df['FTR']==0]['FTHG'].mean()
avD=df[df['FTR']==1]['FTAG'].mean()

avHTwin2=df[df['FTR']==0]['FTAG'].mean()
avATwin2=df[df['FTR']==2]['FTAG'].mean()
avD2=df[df['FTR']==1]['FTHG'].mean()

print(avHTwin,avATwin,avD)
print(avHTwin2,avATwin2,avD2)

# colarrayH=["red","red","orange","orange","yellow","yellow","bleu","bleu","green","green","violet"]
# colarrayA=["red","red","orange","orange","yellow","yellow","bleu","bleu","green"]

#HTGD - Home team goal difference
#ATGD - away team goal difference
#HTP - Home team points
#ATP - Away team points
#HTForm Last 10 Games points of Home team
#ATForm Last 10 Games points of away team
# df.plot(x='FTHG', y='FTR', style='o')
# df.plot(x='FTAG', y='FTR', style='o')
# df.plot(x='HTP', y='FTR', style='o')
# df.plot(x='ATP', y='FTR', style='o')
# df.plot(x='HTGD', y='FTR', style='o')
# df.plot(x='ATGD', y='FTR', style='o')
# df.plot(x='HTForm', y='FTR', style='o')
# df.plot(x='ATForm', y='FTR', style='o')

print(df[['HomeTeam','AwayTeam','FTR','HTGD','ATGD','HTP','ATP','ATForm','HTForm']])







#why do we need time series why not conventianal aproashes aproashes?
#we are getting a perfect variance of 1 so predictions are perfect ?
#no!, we are giving the model the data of the match completed so its 
    #naturally knowing the outcome 100% since it only has to look at 
    #the goal diffrence. and this is visibile in the coeficient that 
    #it gave us: 1 for goal scored and -1 for goal conceded and nearly 
    #0 for everything else
   
    
   
#linear regression Model

#Training with the first 10% of data
print()
print()
import matplotlib.pyplot as plt
import numpy as np


gd=(df['FTHG']-df['FTAG'])
# defining feature matrix(X) and response vector(y)
X = df[['HTGD','ATGD','HTP','ATP','HomeTeam','AwayTeam','HTForm','ATForm']].iloc[0:int(len(df)/10)]
y = gd.iloc[0:int(len(df)/10)]

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
 													random_state=1)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: ', reg.coef_)

#variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plot for residual error

# #setting plot style
# plt.style.use('fivethirtyeight')

# #plotting residual errors in training data
# plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
# 			color = "red", s = 10, label = 'Train data')

#  plotting residual errors in test data
# plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
# 			color = "blue", s = 10, label = 'Test data')

# #plotting line for zero residual error
# #plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

# ## plotting legend
# plt.legend(loc = 'upper right')

# ## plot title
# plt.title("Residual errors")

# ## method call for showing the plot
# plt.show()



#Training with the rest of the data
varScorearr=[]
# defining feature matrix(X) and response vector(y)
for i in range (int(len(df)/10),len(df)):
    X = df[['HTGD','ATGD','HTP','ATP','HomeTeam','AwayTeam','HTForm','ATForm']].iloc[0:i+1]
    y = gd.iloc[0:i+1]
    X_train=X.iloc[0:i-3]
    y_train=y.iloc[0:i-3]
    X_test=X.iloc[i-2:i]
    y_test=y.iloc[i-2:i]
    
    # create linear regression object
    reg = linear_model.LinearRegression()
    
    # train the model using the training sets
    reg.fit(X_train, y_train)
    
    # variance score: 1 means perfect prediction
    score=reg.score(X_test, y_test)
    varScorearr.append(score)
    # if (i>len(df)-6):
    #     print("predicted:",reg.predict(X_test.iloc[0:1]))
    #     print("actuale:",y_test.iloc[0:1])

a=0
for i in range (len(varScorearr)): 
    a=a+varScorearr[i]
#print(a/len(varScorearr))


























import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from IPython.display import display
from time import time 
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print ("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    start = time()
    y_predic = clf.predict(features)
    end = time()
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target, y_predic, pos_label='H',average='micro'), sum(target == y_predic) / float(len(y_predic))

def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))  
    train_classifier(clf, X_train, y_train)  
    f1, acc = predict_labels(clf, X_train, y_train)
    print (acc)
    print ("Accuracy score of training set:",acc)   
    f1, acc = predict_labels(clf, X_test, y_test)
    print ("Accuracy score for test set: ",acc)

X = df[['HTGD','ATGD','HTP','ATP','HomeTeam','AwayTeam','HTForm','ATForm']]
y = df['FTR']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 200,random_state = 2,stratify = y)


clf_A = LogisticRegression(random_state = 42)
clf_B = SVC(random_state = 912, kernel='rbf')
clf_C = xgb.XGBClassifier(seed = 82, eta=10)


train_predict(clf_A, X_train, y_train, X_test, y_test)
print ('')
train_predict(clf_B, X_train, y_train, X_test, y_test)
print ('')
train_predict(clf_C, X_train, y_train, X_test, y_test)
print ('')
print()
print()

howmany=31
match= [i-(len(X_test)-howmany) for i in range (len(X_test)-howmany,len(X_test))]
#plots
plt.figure(1)
ac=[y_test.iloc[i] for i in range (len(X_test)-howmany,len(X_test))]
plt.plot(match,clf_A.predict(X_test.iloc[len(X_test)-howmany:]),"g")
plt.plot(match,ac,"r")
plt.xlabel('Match')
plt.ylabel('W/D/L')
plt.title('Actuale vs Predicted')
plt.figure(2)
plt.plot(match,clf_B.predict(X_test.iloc[len(X_test)-howmany:]),"g")
plt.plot(match,ac,"r")
plt.xlabel('Match')
plt.ylabel('W/D/L')
plt.title('Actuale vs Predicted')
plt.figure(3)
plt.plot(match,clf_C.predict(X_test.iloc[len(X_test)-howmany:]),"g")
plt.plot(match,ac,"r")
plt.xlabel('Match')
plt.ylabel('W/D/L')
plt.title('Actuale vs Predicted')




