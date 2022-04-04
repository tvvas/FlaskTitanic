from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def titanic_form():
    if request.method == 'GET':
        return render_template('titanic_form.html')
    else:
        sex = request.form['sex']

        pclass = request.form['pclass']
        if pclass.isnumeric():
            pclass = int(pclass)
            if pclass not in [1, 2, 3]:
                pclass = np.nan
        else:
            pclass = np.nan

        age = request.form['age']
        if age.isnumeric():
            age = float(age)
            if age < 0 or age > 120:
                age = np.nan
        else:
            age = np.nan

        print("sex = ", sex)
        print("pclass = ", pclass)
        print("age = ", age)

        with open("titanic_pipeline.pickle", "rb") as infile:
            titanic_pipeline = pickle.load(infile)

        prediction = titanic_pipeline.predict(
            pd.DataFrame(
                {
                    'Sex': [sex],
                    'Pclass': [pclass],
                    'Age': [age]
                }
            )
        )

        print("prediction:", prediction)

        return render_template('result.html', result=prediction[0])




if __name__ == '__main__':
    app.run()
