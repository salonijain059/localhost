import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn
import itertools
#import numpy as np
#import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
#import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
#from sklearn.metrics import f1_score
#from sklearn.metrics import jaccard_similarity_score
#from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns

from sklearn.preprocessing import StandardScaler
st.title("LOAN PREDICTION APPLICATION")
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style.css")
@st.cache
def loadData():
    df=pd.read_csv("Final.csv")
    df.drop("Unnamed: 0",axis=1,inplace=True)
    return df
def process(data):
    X=data.drop("loan_status",axis=1)
    y=data["loan_status"]
    X= preprocessing.StandardScaler().fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=0)
    return X_train, X_test, y_train, y_test
 
def knn_classifier(X_train,X_test,y_train,y_test):
    
    Principal=st.number_input("Principal")
    terms=st.number_input("terms")
    age=st.number_input("age")
    Gender=st.number_input("Gender")
    weekend=st.number_input("weekend")
    Bechalor=st.number_input("Bechalor")
    School=st.number_input("School")
    college=st.number_input("college")
    neigh=KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train,y_train)
    y_hat=neigh.predict(X_test)
    prediction=neigh.predict([[Principal,terms,age,Gender,weekend,Bechalor,School,college]])
    score=metrics.accuracy_score(y_test,y_hat)*100
    report = classification_report(y_test, y_hat)
    cm=confusion_matrix(y_test,y_hat,labels=['PAIDOFF','COLLECTION'])
    np.set_printoptions(precision=2)
    
    #plot_confusion_matrix(cm, classes=['PAIDOFF(0)','COLLECTION(1)'],normalize= False,  title='Confusion matrix')
    return prediction,score,report,cm,y_hat




def logisticreg(X_train,X_test,y_train,y_test):
    Principal=st.number_input("Principal")
    terms=st.number_input("Terms")
    age=st.number_input("Age")
    Gender=st.number_input("Gender")
    weekend=st.number_input("Weekend")
    Bechalor=st.number_input("Bachelor")
    School=st.number_input("School")
    college=st.number_input("College")
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
    yhat = LR.predict(X_test)
    pre=LR.predict([[Principal,terms,age,Gender,weekend,Bechalor,School,college]])
    score=metrics.accuracy_score(y_test,yhat)*100
    report = classification_report(y_test, yhat)
    cm=confusion_matrix(y_test,yhat,labels=['PAIDOFF','COLLECTION'])
    return pre,score,report,cm,yhat
    
def svm(X_train,X_test,y_train,y_test):
    Principal=st.number_input("Principal")
    terms=st.number_input("Terms")
    age=st.number_input("Age")
    Gender=st.number_input("Gender")
    weekend=st.number_input("Weekend")
    Bechalor=st.number_input("Bachelor")
    School=st.number_input("School")
    college=st.number_input("College")
    
    clf = svm.SVC(kernel='rbf',C=1,gamma=1)
    clf.fit(X_train, y_train) 
    ypre=clf.predict(X_test)
    we=clf.predict([[Principal,terms,age,Gender,weekend,Bechalor,School,college]])
    score=metrics.accuracy_score(y_test,ypre)*100
    report = classification_report(y_test, ypre)
    cm=confusion_matrix(y_test,ypre,labels=['PAIDOFF','COLLECTION'])
    return we,score,report,cm,ypre
    
    
def decisiontree(X_train,X_test,y_train,y_test):
         
         
    Principal=st.number_input("Principal")
    terms=st.number_input("terms")
    age=st.number_input("age")
    Gender=st.number_input("Gender")
    weekend=st.number_input("weekend")
    Bechalor=st.number_input("Bechalor")
    School=st.number_input("School")
    college=st.number_input("college")

    
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 8)
    drugTree.fit(X_train,y_train)
    y_pred=drugTree.predict(X_test)
    pred=drugTree.predict([[Principal,terms,age,Gender,weekend,Bechalor,School,college]])
    score=metrics.accuracy_score(y_test,y_pred)*100
    report = classification_report(y_test, y_pred)
    cm=confusion_matrix(y_test,y_pred,labels=['PAIDOFF','COLLECTION'])
    
    return pred,score,report,cm,y_pred

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            



def main():
    #st.header("LOAN PREICTION")
    data=loadData()

    X_train,X_test,y_train,y_test=process(data)

    
    choose_model=st.sidebar.selectbox("Choose ML model",["Welcome","K-Nearest Neighbour","Decision Tree","Logistic Regression"])
    if(choose_model=="Welcome"):
       
        st.subheader("INTRODUCTION")
        st.write("")
        st.write("")
        st.write("This is the Loan Prediction application. The various model used in this application helps us to predict whether the customer would be able to pay the loan or not. We will do this by predicting two labels namely ""PAIDOFF"" and ""COLLECTION"".") 
        st.subheader("OVERVIEW OF DATA")
        st.write(" Initially the data consist of several labels which might not be that useful. So we cleaned the data and used only the required useful labels which are important to predict whether or not to give the loan as the user would be able to pay the loan or not. See the dataset below!")
        
        check_data = st.checkbox("See the simple data")
        if check_data:
            st.write(data.tail(10))
            st.write(data.shape)
        st.subheader("BUILDING PREDICTIVE MODELS")
        st.write("After data cleaning had mainly used 3 predictive models with different accuracy. In this application the user will have to enter his details and bank will be able to predict whether the customer will be able to pay it or not.")
        st.write("")
        
        st.write("1.K-Nearest Neighbour")
        st.write("2.Decision Tree")
        st.write("3.Logistic Regression")
        
        st.title("Thank You!!!!")
        
        
        
    elif (choose_model=="K-Nearest Neighbour"):
        st.header("Model:K-Nearest Neighbour")
        local_css("style.css")
        prediction,score,report,cm,y_hat=knn_classifier(X_train,X_test,y_train,y_test)
        check_data = st.checkbox("See the simple data")
        if check_data:
            st.write(data.head(10))
            st.write(data.shape)
        if st.button("PREDICTION"):
            st.header("1.Loan Prediction for the input values is:")
            st.subheader(prediction)
            st.header("2.Accuracy of the model is:")
            st.subheader(score)
            #st.subheader("The Classification Report is")
            #st.write(report)
            st.header("3. The Confusion Matrix is ")
            #st.write(cm)
            cm = confusion_matrix(y_test, y_hat, labels=['PAIDOFF','COLLECTION'])
            np.set_printoptions(precision=2)
                    
            st.pyplot(plot_confusion_matrix(cm, classes=['PAIDOFF(0)','COLLECTION(1)'],normalize= False,  title='Confusion matrix'))
        
    #elif(choose_model=="Support Vector Machine"):
     #   we,score,report,cm,ypre=svm(X_train,X_test,y_train,y_test)
     #   check_data = st.checkbox("See the simple data")
      #  if check_data:
       #     st.write(data.tail(10))
        #    st.write(data.shape)
        #if st.button("PREDICTION"):
         #   st.header("1.Loan Prediction for the input values is:")
          #  st.subheader(we)
          #  st.header("2.Accuracy of the model is:")
           # st.subheader(score)
            #st.subheader("The Classification Report is")
            #st.write(report)
            #st.header("3. The Confusion Matrix is ")
            #st.write(cm)
       # cm = confusion_matrix(y_test, ypre, labels=['PAIDOFF','COLLECTION'])
        #np.set_printoptions(precision=2)
                    
        #st.pyplot(plot_confusion_matrix(cm, classes=['PAIDOFF(0)','COLLECTION(1)'],normalize= False,  title='Confusion matrix'))
        
        
        
            
    elif (choose_model=="Decision Tree"):
        local_css("style.css")
        st.header("Model: Decision Tree")
        pred,score,report,cm,y_pred=decisiontree(X_train,X_test,y_train,y_test)
        check_data = st.checkbox("See the simple data")
        if check_data:
            st.write(data.tail(10))
            st.write(data.shape)
        if st.button("Run me"):
            st.subheader("1. The Prediction is:")
            st.header(pred)
            st.header("2. Accuracy of the model is:")
            st.subheader(score)
            #st.text("The report is")
            #st.write(report)
            #st.write(cm)
            st.header("3. The Confusion Matrix is")
            cm = confusion_matrix(y_test, y_pred, labels=['PAIDOFF','COLLECTION'])
            np.set_printoptions(precision=2)
                    
            st.pyplot(plot_confusion_matrix(cm, classes=['PAIDOFF(0)','COLLECTION(1)'],normalize= False,  title='Confusion matrix'))
        
    elif(choose_model=="Logistic Regression"):
        st.header("Model:Logistic Regression")
        local_css("style.css")
        pre,score,report,cm,yhat=logisticreg(X_train,X_test,y_train,y_test)
        check_data = st.checkbox("See the simple data")
        if check_data:
            st.write(data.tail(10))
            st.write(data.shape)
        if st.button("Prediction"):
            st.header("1. The prediction is:")
            st.subheader(pre)
            st.header("2.Accuracy of the model is:")
            st.write(score,"%")
            #st.text("The report is")
            #st.write(report)
            #st.write(cm)
            st.header("3. The Confusion Matrix is:")
            cm = confusion_matrix(y_test, yhat, labels=['PAIDOFF','COLLECTION'])
            np.set_printoptions(precision=2)
                    
            st.pyplot(plot_confusion_matrix(cm, classes=['PAIDOFF(0)','COLLECTION(1)'],normalize= False,  title='Confusion matrix'))
        
        
            
    else:        
        
        st.header("Yes")



if __name__=="__main__":
    main()



    
