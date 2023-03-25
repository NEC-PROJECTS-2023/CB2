from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import os
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

app.secret_key = 'welcome'

with open('model/rf.txt', 'rb') as file:
    rf = pickle.load(file)
file.close()

with open('model/encoder.txt', 'rb') as file:
    label_encoder = pickle.load(file)
file.close()

with open('model/scaler.txt', 'rb') as file:
    scaler = pickle.load(file)
file.close()

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST' and 't1' in request.form:
        data = request.form['t1']
        arr = data.split(",")
        columns = "id,code_module,code_presentation,id_assessment,assessment_type,date,weight,module_presentation_length,id_student,date_submitted,is_banked,score,gender,region,highest_education,imd_band,age_band,num_of_prev_attempts,studied_credits,disability"
        col_arr = columns.split(",")
        values = []
        for i in range(len(arr)):
            if i == 1 or i == 2 or i == 4 or i == 12 or i == 13 or i == 14 or i == 19:
                values.append(arr[i])
            else:
                values.append(float(arr[i]))
        temp = []
        temp.append(values)
        temp = np.asarray(temp)
        dataset = pd.DataFrame(temp, columns = col_arr)
        dataset['id'] = dataset['id'].astype(float)
        dataset['id_assessment'] = dataset['id_assessment'].astype(float)
        dataset['date'] = dataset['date'].astype(float)
        dataset['weight'] = dataset['weight'].astype(float)
        dataset['module_presentation_length'] = dataset['module_presentation_length'].astype(float)
        dataset['id_student'] = dataset['id_student'].astype(float)
        dataset['date_submitted'] = dataset['date_submitted'].astype(float)
        dataset['is_banked'] = dataset['is_banked'].astype(float)
        dataset['score'] = dataset['score'].astype(float)
        dataset['imd_band'] = dataset['imd_band'].astype(float)
        dataset['age_band'] = dataset['age_band'].astype(float)
        dataset['num_of_prev_attempts'] = dataset['num_of_prev_attempts'].astype(float)
        dataset['studied_credits'] = dataset['studied_credits'].astype(float)        
        dataset.drop(['id'], axis = 1,inplace=True)
        data = dataset
        data = data.values
        columns = dataset.columns
        types = dataset.dtypes.values
        print(dataset.info())
        index = 0
        for i in range(len(types)):
            name = types[i]
            if name == 'object':
                dataset[columns[i]] = pd.Series(label_encoder[index].fit_transform(dataset[columns[i]].astype(str)))
                index = index + 1
        dataset.fillna(0, inplace = True)
        dataset = dataset.values
        dataset = dataset[:,0:dataset.shape[1]]
        dataset = scaler.transform(dataset)
        predict = rf.predict(dataset)
        print(predict)
        labels = ['Distinction', 'Fail', 'Pass', 'Withdrawn']
        output = "Test Data = "+str(data[0])+"<br/> ====> Predicted As : "+labels[int(predict[0])]
        return render_template('Predict.html', msg=output)
        

@app.route('/Predict', methods=['GET', 'POST'])
def predict():
    return render_template('Predict.html', msg='')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')

@app.route('/AdminLogin', methods=['GET', 'POST'])
def AdminLogin():
   return render_template('AdminLogin.html', msg='')

@app.route('/AdminLoginAction', methods=['GET', 'POST'])
def AdminLoginAction():
    if request.method == 'POST' and 't1' in request.form and 't2' in request.form:
        user = request.form['t1']
        password = request.form['t2']
        if user == "admin" and password == "admin":
            return render_template('AdminScreen.html', msg="Welcome "+user)
        else:
            return render_template('AdminLogin.html', msg="Invalid login details")

@app.route('/Logout')
def Logout():
    return render_template('index.html', msg='')



if __name__ == '__main__':
    app.run()










