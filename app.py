from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__, template_folder='template')
adaboost = pickle.load(open('Final_model', 'rb'))

@app.route('/')
def home():
    return render_template("homepage.html")




def get_data():
    word_freq_our = request.form.get('word_freq_our')
    word_freq_our = int(word_freq_our)
    word_freq_remove = request.form.get('word_freq_remove')
    word_freq_remove = int(word_freq_remove)
    word_freq_free = request.form.get('word_freq_free')
    word_freq_free = int(word_freq_free)
    word_freq_your = request.form.get('word_freq_your')
    word_freq_your = int(word_freq_your)
    word_freq_hp = request.form.get('word_freq_hp')
    word_freq_hp = int(word_freq_hp)
    char_freq_not = request.form.get('char_freq_not')
    char_freq_not = int(char_freq_not)
    char_freq_dollar = request.form.get('char_freq_dollar')
    char_freq_dollar = int(char_freq_dollar)
    capital_run_length_average = request.form.get('capital_run_length_average')
    capital_run_length_average = int(capital_run_length_average)    
    capital_run_length_longest = request.form.get('capital_run_length_longest')
    capital_run_length_longest = int(capital_run_length_longest)
	#capital_run_length_total = request.form.get('capital_run_length_total')
    #capital_run_length_total = int(capital_run_length_total)
    capital_run_length_total = request.form.get('capital_run_length_total')
    char_freq_ = int(capital_run_length_total)

    d_dict = {'word_freq_our': [word_freq_our],'word_freq_remove': [word_freq_remove],'word_freq_free': [word_freq_free],'word_freq_your': [word_freq_your], 'word_freq_hp': [word_freq_hp],
              'char_freq_not': [char_freq_not],'char_freq_dollar': [char_freq_dollar],'capital_run_length_average': [capital_run_length_average],'capital_run_length_longest': [capital_run_length_longest],
              'capital_run_length_total': [capital_run_length_total]}


    #
    return pd.DataFrame.from_dict(d_dict, orient='columns')



@app.route('/send', methods=['POST'])
def show_data():
    df = get_data()
    prediction = adaboost.predict(df)
    if prediction[0] == 0:
    	prediction = 'Non Spam'
    else:
    	prediction = 'Spam Email'

    return render_template('results.html', tables = [df.to_html(classes='data', header=True)],result = prediction)


if __name__=="__main__":
    app.run(debug=True)