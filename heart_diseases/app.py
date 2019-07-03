from flask import Flask, render_template, url_for,request
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    heart_df=pd.read_csv("framingham.csv")
    heart_df.drop(['education'],axis=1,inplace=True)
    heart_df.rename(columns={'male':'Sex_male'},inplace=True)
    heart_df.dropna(axis=0,inplace=True)
    from statsmodels.tools import add_constant as add_constant
    heart_df_constant = add_constant(heart_df)
    st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
    cols=heart_df_constant.columns[:-1]
    model=sm.Logit(heart_df.TenYearCHD,heart_df_constant[cols])
    result=model.fit()
    import sklearn
    new_features=heart_df[['age','Sex_male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]
    x=new_features.iloc[:,:-1]
    y=new_features.iloc[:,-1]
    from sklearn.cross_validation import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)

    from sklearn.linear_model import LogisticRegression
    logreg=LogisticRegression()
    logreg.fit(x_train,y_train)
    y_pred=logreg.predict(x_test)
    logreg.score(x_test,y_test)

    if request.method == 'POST':
        Sex_male = request.form['Sex_male']
        age = request.form['age']
        education = request.form['education']
        currentSmoker = request.form['currentSmoker']
        cigsPerDay = request.form['cigsPerDay']
        BPMeds = request.form['BPMeds']
        prevalentStroke = request.form['prevalentStroke']
        prevalentHyp = request.form['prevalentHyp']
        diabetes = request.form['diabetes']
        totChol = request.form['totChol']
        sysBP = request.form['sysBP']
        diaBP = request.form['diaBP']
        BMI = request.form['BMI']
        heartRate = request.form['heartRate']
        glucose = request.form['glucose']
        TenYearCHD = request.form['TenYearCHD']


        data = [age,Sex_male,cigsPerDay,totChol,sysBP,glucose,TenYearCHD]
        cv = CountVectorizer()
        vect = cv.transform(data).toarray()
        newdata = logreg.predict(vect)
        fitmodel=model.fit(logreg)
        fitted.results <- predict(fitmodel,newdata=test)
        fitted.results <- ifelse(fitted.results > 0.5,1,0)
        result<-fitted.results
        Predictions<-data.frame(result)
    return render_template('result.html',result)

if __name__ == '__main__':
	app.run(debug=True)
	
