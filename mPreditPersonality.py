#Importing the required modules
import pandas as pd
import numpy as np
#from scipy.sparse.construct import random
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model  import LogisticRegression
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.naive_bayes  import GaussianNB
from sklearn.svm  import SVC
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble  import RandomForestClassifier

data=pd.read_csv(r'training_dataset1.csv')

class Data_analysis:
    #Exploring the data analysis
    #Understanding the dataset
    def analysis (self):
        print(data.head())
        print(data.shape)
        print(data.info())
        print(data.describe())
        print(data.isnull().values.any())
   

class preprocessing():
    def pre(self):

        array = data.values
        for i in range(len(array)):
            if array[i][0]=="Male":
                array[i][0]=1
            else:
                array[i][0]=0
        pdata=pd.DataFrame(array)
        pdata=pdata.drop_duplicates()
        return(pdata)


class Data_visulalisation:
    #prepro=preprocessing()
    #pdata=pd.DataFrame(prepro.pre())
    #Count ploting the dependent data
    pdata=pd.read_csv(r'training_dataset1.csv')
    
    ##Histogram of the  data set
    def hist(self):
        self.pdata.hist(bins=8,figsize=(8,8))
        return plt.show()
   
    ##Analizing the data set and the dependent valriables
    def correlation(self):
        cor=self.pdata.corr()
        cor_index=cor.index
        plt.figure(figsize=(12,12))
        g=sns.heatmap(self.pdata[cor_index].corr(),annot=True,cmap="Blues")
        return plt.show()

    

#training the model
class seleting_model:
    prepro=preprocessing()
    pdata=prepro.pre()
    y=pdata[7]
    x=pdata[[0,1,2,3,4,5,6]]
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,stratify=y,random_state=7)
    
    def training_testing(self):
        #logical regression
        lr=LogisticRegression(solver='liblinear',multi_class='ovr')
        lr.fit(self.xtrain,self.ytrain)
        #kNeighbors classifier(KNN)
        knn=KNeighborsClassifier()
        knn.fit(self.xtrain,self.ytrain)
        #Navi bayes classifier
        nb=GaussianNB()
        nb.fit(self.xtrain,self.ytrain)
        #support vector machine(svm)
        svm=SVC()
        svm.fit(self.xtrain,self.ytrain)
        #Decision tree
        dt=DecisionTreeClassifier()
        dt.fit(self.xtrain,self.ytrain)
        #Random forest
        rf=RandomForestClassifier()
        rf.fit(self.xtrain,self.ytrain)
        
        #Model Evaluation
        print("Training set Accuracy ")
        #logical regression
        train_accuracy_lr=accuracy_score(lr.predict(self.xtrain),self.ytrain)
        print('Logical regression =',train_accuracy_lr*100)
        #kNeighbors classifier(KNN)
        train_accuracy_knn=accuracy_score(knn.predict(self.xtrain),self.ytrain)
        print('KNeighbors Classifier =',train_accuracy_knn*100)
        #Navi bayes classifier
        train_accuracy_nb=accuracy_score(nb.predict(self.xtrain),self.ytrain)
        print('Navi Bayes =',train_accuracy_nb*100)
        #support vector machine(svm)
        train_accuracy_svm=accuracy_score(svm.predict(self.xtrain),self.ytrain)
        print('Support vector Machine =',train_accuracy_svm*100)
        #Decision tree
        train_accuracy_dt=accuracy_score(dt.predict(self.xtrain),self.ytrain)
        print('Decision Tree =',train_accuracy_dt*100)
        #Random forest
        train_accuracy_rf=accuracy_score(rf.predict(self.xtrain),self.ytrain)
        print('Random Forest =',train_accuracy_rf*100)
        
        #Accuracy score evalution
        print("Testing the Accuracy")
        #logical regression
        test_accuracy_lr=accuracy_score(lr.predict(self.xtest),self.ytest)
        print('Logical regression =',test_accuracy_lr*100)
        #kNeighbors classifier(KNN)
        test_accuracy_knn=accuracy_score(knn.predict(self.xtest),self.ytest)
        print('KNeighbors Classifier =',test_accuracy_knn*100)
        #Navi bayes classifier
        test_accuracy_nb=accuracy_score(nb.predict(self.xtest),self.ytest)
        print('Navi Bayes =',test_accuracy_nb*100)
        #support vector machine(svm)
        test_accuracy_svm=accuracy_score(svm.predict(self.xtest),self.ytest)
        print('Support Vector Machine =',test_accuracy_svm*100)
        #Decision tree
        test_accuracy_dt=accuracy_score(dt.predict(self.xtest),self.ytest)
        print('Decision Tree =',test_accuracy_dt*100)
        #Random forest
        test_accuracy_rf=accuracy_score(rf.predict(self.xtest),self.ytest)
        print('Random Forest =',test_accuracy_rf*100)
        print(f'The Accuracy of Random Forest model is {test_accuracy_lr*100} , higher so for this dataset we are going to use Logical Regression model ')
    
    
class predicte:
    
    def pre(self,Gender,Age,openness,neuroticism,conscientiousness,agreeableness,extraversion):
        prepro=preprocessing()
        pdata=prepro.pre()
        maindf =pdata[[0,1,2,3,4,5,6]]
        mainarray=maindf.values
        temp=pdata[7]
        train_y =temp.values
        lr =LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
        lr.fit(mainarray, train_y)
        indata=(Gender,Age,openness,neuroticism,conscientiousness,agreeableness,extraversion)
        inarray=np.array(indata)
        inreshape=inarray.reshape(1,-1)
        instd=inreshape
        print(lr.predict(instd))
        return lr.predict(instd)

if __name__=='__main__':
    b=Data_visulalisation()
    b.pie()
    #b.pre(1,4,5,6,7,8,6)
    #b.hist()
    #b.correlation()
    #c=seleting_model()
    #b.analysis()
    #d=predicte()
    #d.pre(1,18,6,5,3,5,3)
    #d.analysis()
    #d.pie()
    #d.hist()
    #d.pair_plot()
    #d.correlation()
