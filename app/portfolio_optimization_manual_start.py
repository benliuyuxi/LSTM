import portfolio_optimization

portfolio_optimization.start('../ai-funds-raw-data.csv', portfolio_types=['markowitz', 'minimum_variance'], monitors=['sharpe_ratio'], plot=True)