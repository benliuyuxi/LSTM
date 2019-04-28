from flask import render_template, request
import os

from app import portfolio_optimization

from app import app

@app.route('/')
# @app.route('/index')
# def index():
#     user = {'username': 'Scott'}
#     return render_template('index.html', title='Home', user=user, value=None)

@app.route('/index', methods=['GET', 'POST'])
def index():

    file_status = 'n/a'
    sigma_opt = 'n/a'
    mu_opt = 'n/a'

    if request.method == 'POST':
        raw_data = request.form['raw_data'].lower()

        if os.path.exists(raw_data):
            sigma_opt, mu_opt = portfolio_optimization.start(raw_data)
            file_status = 'Raw data file is {0}'.format(raw_data)
        else:
            file_status = 'This raw data file does not exist!'
        
        return render_template('index.html', title='Home', file_status=file_status, sigma_opt=sigma_opt, mu_opt=mu_opt)

    elif request.method == 'GET':
        return render_template('index.html', title='Home', file_status=file_status, sigma_opt=sigma_opt, mu_opt=mu_opt)