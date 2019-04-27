from flask import render_template, request

from app import portfolio_optimization

from app import app

@app.route('/')
# @app.route('/index')
# def index():
#     user = {'username': 'Scott'}
#     return render_template('index.html', title='Home', user=user, value=None)

@app.route('/index', methods=['GET', 'POST'])
def index():

    raw_data = 'n/a'
    sigma_opt = 'n/a'
    mu_opt = 'n/a'

    if request.method == 'POST':
        raw_data = request.form['raw_data'].lower()
        sigma_opt, mu_opt = portfolio_optimization.start(raw_data)
        return render_template('index.html', title='Home', raw_data=raw_data, sigma_opt=sigma_opt, mu_opt=mu_opt)

    elif request.method == 'GET':
        return render_template('index.html', title='Home', raw_data=raw_data, sigma_opt=sigma_opt, mu_opt=mu_opt)