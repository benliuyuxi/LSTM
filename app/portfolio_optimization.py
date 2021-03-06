import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set up logging
import logging

logger = logging.getLogger('portfolio-optimization')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('portfolio-optimization-debug.log', mode='w') # mode='w' to overwrite existing log file
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s') # output method name too
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)







# Set numpy print options
np.set_printoptions(precision=3)


def get_raw_data(raw_data, plot=False):
    ''' Get dataframe of raw data
    Args:
        
        raw_data (string): path to the raw data CSV file
        plot (boolean): optional, default False. whether plots should be made.

    Raises:

    Returns:
        df_raw_data (dataframe): 

    '''


    logger.info('Get raw data from CSV file: {0}'.format(raw_data))

    df_raw_data = pd.read_csv(raw_data)
    df_raw_data.set_index('date', inplace=True)

    if 'Unnamed: 0' in df_raw_data.columns:
        df_raw_data.drop(columns=['Unnamed: 0'], inplace=True)

    logger.debug('\ndf_raw_data\n{0}\nwith dimensions {1}x{2}'.format(df_raw_data, df_raw_data.shape[0], df_raw_data.shape[1]))

    if plot is True:
        df_raw_data.plot(figsize=(15,10), title='Closing prices of all assets')
    else:
        pass

    return df_raw_data




def get_r(df_raw_data, plot=False):
    ''' Get rate of return matrix from dataframe of raw data
    Args:
        
        df_raw_data (dataframe): 
        plot (boolean): optional, default False. whether plots should be made.

    Raises:

    Returns:
        r (ndarray): rate of return matrix with dimensions n x t
                        n: number of assets
                        t: number of timestamps

                        This is calculated and converted from DataFrame of raw data
    '''


    logger.info('Get rates of return from dataframe of raw data ...')

    df_r = df_raw_data.pct_change().dropna() # Calculate rates of return using percentage change method

    r = df_r.values.transpose()

    logger.debug('\nr\n{0}\nwith dimensions {1}x{2}'.format(r, r.shape[0], r.shape[1]))

    if plot is True:
        plt.figure(figsize=(15,10))

        x = np.arange(r.shape[1])
        for i in np.arange(r.shape[0]):

            # plt.plot(<X AXIS VALUES HERE>, <Y AXIS VALUES HERE>, 'line type', label='label here')
            plt.plot(x, r[i, :], label='Asset {0}'.format(i))
        
        plt.title('Rates of return for all assets')
        plt.xlabel('Time')
        plt.ylabel('Rate of return')
        plt.legend(loc='best')
    else:
        pass

    return r


def get_efficient_frontier_params(mu=None, Sigma=None, r=None):
    ''' 
    Args:
        
        mu (ndarray): optional, mean rate of return matrix with dimensions n x 1
        Sigma (ndarray): optional, dimensions n x n
        r (ndarray): optional, rate of return matrix with dimensions n x t
                                n: number of assets
                                t: number of timestamps
                                
                                r is required when mu and Sigma are not present
        

    Raises:

    Returns:
        decayed_alphas (array): 
    '''

    if mu is None and Sigma is None and r is not None: # calculate Sigma and mu given r
        logger.debug('\nr\n{0}'.format(r))

        Sigma = np.cov(r)
        logger.debug('\nSigma\n{0}'.format(Sigma))

        mu = np.mean(r, axis=1)[np.newaxis].transpose() # mu with dimensions n x 1
        logger.debug('\nmu\n{0}'.format(mu))

    elif mu is not None and Sigma is not None and r is None:
        logger.debug('\nSigma\n{0}'.format(Sigma))
        logger.debug('\nmu\n{0}'.format(mu))

    elif mu is not None and Sigma is not None and r is not None:
        logger.info('Detected mu, Sigma and r. Selecting mu and Sigma ...\n')
        logger.debug('\nSigma\n{0}'.format(Sigma))
        logger.debug('\nmu\n{0}'.format(mu))

    else:
        raise ValueError('Need mu and Sigma, or r')


    n = mu.shape[0]
    logger.info('Number of assets = {0}'.format(n))

    Sigma_inv = np.linalg.inv(Sigma)
    logger.debug('\nSigma_inv\n{0}'.format(Sigma_inv))

    ones_matrix = np.ones((n, 1))
    logger.debug('\nones_matrix\n{0}'.format(ones_matrix))

    a = mu.transpose().dot(Sigma_inv).dot(mu).item() # .item() converts 1x1 matrix to scalar at the end
    logger.info( 'a = {0}'.format(a) )

    b = ones_matrix.transpose().dot(Sigma_inv).dot(mu).item()
    logger.info( 'b = {0}'.format(b) )

    c = ones_matrix.transpose().dot(Sigma_inv).dot(ones_matrix).item()
    logger.info( 'c = {0}'.format(c) )

    d = a*c - b**2
    logger.info( 'd = {0}'.format(d) )

    return a, b, c, d, mu, Sigma_inv


def efficient_frontier(a, b, c, d, mu, Sigma_inv, C_0, mu_p):
    ''' Given an array of mean portfolio return values, calculate an array of portfolio 
        standard deviation/risk values, as well as 

    Args:
        a (scalar): 
        b (scalar): 
        c (scalar): 
        d (scalar): 
        mu (ndarray): dimensions n x 1
        Sigma_inv (ndarray): dimensions n x n
        C_0 (scalar): capital that can be invested
        mu_p (scalar or ndarray): mean portfolio return, provided as an array for plotting purpose
        

    Raises:

    Returns:
        sigma_p (ndarray): 
        theta_ef (ndarray, or array of ndarray): 
    '''

    n = mu.shape[0]
    logger.debug('Number of assets = {0}'.format(n))

    ones_matrix = np.ones((n, 1))
    logger.debug('\nones_matrix\n{0}'.format(ones_matrix))

    #   sigma = 1/d * (c * np.square(mean) - 2*b*C_0 * mean + a*C_0**2)
    sigma_p = np.sqrt( 1/d * (c * np.square(mu_p) - 2*b*C_0 * mu_p + a*C_0**2) )
    logger.info( '\nsigma_p\n{0}'.format(sigma_p) )

    if isinstance(mu_p, (list, tuple, np.ndarray)):
        logger.debug('Detected mu_p as an array')
        theta_ef = []

        for x in mu_p:
            y = 1/d * Sigma_inv.dot( (a*ones_matrix - b*mu)*C_0 + (c*mu - b*ones_matrix)*x )

            logger.debug('\ntheta_ef at mu_p = {0}\n{1}'.format(x, y))
            theta_ef.append( 1/d * Sigma_inv.dot( (a*ones_matrix - b*mu)*C_0 + (c*mu - b*ones_matrix)*x ) )


    else:
        logger.debug('Detected mu_p as a scalar')

        theta_ef = 1/d * Sigma_inv.dot( (a*ones_matrix - b*mu)*C_0 + (c*mu - b*ones_matrix)*mu_p )
        logger.debug('\ntheta_ef\n{0}'.format(theta_ef))

    return sigma_p, theta_ef


def portfolio_minimum_variance(b, c, Sigma_inv, C_0):
    ''' Calculate sigma_mv, mu_mv and theta_mv

    Args:
        b (scalar): 
        c (scalar): 
        Sigma_inv (ndarray): dimensions n x n
        C_0 (scalar): capital that can be invested

    Raises:

    Returns:
        sigma_mv (scalar): 
        mu_mv (scalar): 
        theta_mv (ndarray): 
    '''

    n = Sigma_inv.shape[0]
    logger.debug('Number of assets = {0}'.format(n))

    ones_matrix = np.ones((n, 1))
    logger.debug('\nones_matrix\n{0}'.format(ones_matrix))

    sigma_mv = np.sqrt( C_0**2 / c )
    mu_mv = b/c * C_0

    logger.info('(sigma_mv, mu_mv) = ({0}, {1})'.format(sigma_mv, mu_mv))

    theta_mv = C_0/c * Sigma_inv.dot(ones_matrix)
    logger.info('\ntheta_mv\n{0}'.format(theta_mv))

    return sigma_mv, mu_mv, theta_mv


def portfolio_tangency(a, b, c, d, mu, Sigma_inv, C_0):
    ''' Calculate sigma_tg, mu_tg and theta_tg

    Args:
        a (scalar): 
        b (scalar): 
        c (scalar): 
        d (scalar):
        mu (ndarray): dimensions n x 1
        Sigma_inv (ndarray): dimensions n x n
        C_0 (scalar): capital that can be invested

    Raises:

    Returns:
        sigma_tg (scalar): 
        mu_tg (scalar): 
        theta_tg (ndarray): 
    '''


    sigma_tg = np.sqrt(a)/b * C_0
    mu_tg = a/b * C_0

    logger.info('(sigma_tg, mu_tg) = ({0}, {1})'.format(sigma_tg, mu_tg))

    theta_tg = Sigma_inv.dot(mu) * C_0/b
    logger.info('\ntheta_tg\n{0}'.format(theta_tg))
    
    logger.debug('1/d * (c*mu_tg**2 - 2*b*C_0*mu_tg + a*C_0**2) at ({0}, {1}) is {2}'.format(sigma_tg, mu_tg, 1/d * (c*mu_tg**2 - 2*b*C_0*mu_tg + a*C_0**2) ) )
    logger.debug('Inverse slope of the Sharpe line at ({0}, {1}) is {2}'.format(sigma_tg, mu_tg, np.sqrt(1/d * (c*mu_tg**2 - 2*b*C_0*mu_tg + a*C_0**2))/mu_tg ) )
    logger.debug('Inverse slope of the efficient frontier at ({0}, {1}) is {2}'.format(sigma_tg, mu_tg, (c*mu_tg - b*C_0)/np.sqrt(d * (c*mu_tg**2 - 2*b*C_0*mu_tg + a*C_0**2)) ) )

    mu_tg_1 = (b*C_0 + np.sqrt(b**2 * C_0**2 - 4 * b**2 * c + 4 * a * c**2)) / (2*c)
    mu_tg_2 = (b*C_0 - np.sqrt(b**2 * C_0**2 - 4 * b**2 * c + 4 * a * c**2)) / (2*c)

    logger.debug('mu_tg_1 = {0}, mu_tg_2 = {1}'.format(mu_tg_1, mu_tg_2))

    sigma_tg_1 = np.sqrt(1/d * (c*mu_tg_1**2 - 2*b*C_0*mu_tg_1 + a*C_0**2))
    sigma_tg_2 = np.sqrt(1/d * (c*mu_tg_2**2 - 2*b*C_0*mu_tg_2 + a*C_0**2))

    logger.debug('(sigma_tg_1, mu_tg_1) = ({0}, {1})'.format(sigma_tg_1, mu_tg_1))
    logger.debug('(sigma_tg_2, mu_tg_2) = ({0}, {1})'.format(sigma_tg_2, mu_tg_2))


    return sigma_tg_2, mu_tg_2, theta_tg


def portfolio_markowitz(a, b, c, d, mu, Sigma_inv, C_0, gamma):
    ''' Calculate sigma_opt, mu_opt and theta_opt

    Args:
        a (scalar): 
        b (scalar): 
        c (scalar): 
        d (scalar): 
        mu (ndarray): dimensions n x 1
        Sigma_inv (ndarray): dimensions n x n
        C_0 (scalar): capital that can be invested
        gamma (scalar or ndarray): absolute risk aversion parameter

    Raises:

    Returns:
        sigma_opt (scalar, or ndarray): 
        mu_opt (scalar, or ndarray): 
        theta_opt (ndarray, or array of ndarray): 

    '''

    n = mu.shape[0]
    logger.debug('Number of assets = {0}'.format(n))

    ones_matrix = np.ones((n, 1))
    logger.debug('\nones_matrix\n{0}'.format(ones_matrix))
    

    if isinstance(gamma, (list, tuple, np.ndarray)):
        logger.debug('Detected gamma as an array')

        sigma_opt = np.sqrt( (a*c - b**2 + gamma**2 * C_0**2) / (c * gamma**2) )
        mu_opt = d/(c*gamma) + b*C_0/c

        for i in np.arange(sigma_opt.shape[0]):
            logger.info('With gamma = {0}, (sigma_opt, mu_opt) = ({1}, {2})'.format(gamma[i], sigma_opt[i], mu_opt[i]))

        theta_opt = []

        for x in gamma:
            y = Sigma_inv.dot(mu)/x + Sigma_inv.dot(ones_matrix) * (C_0 - b/x)/c

            logger.debug('\nWith gamma = {0}, theta_opt = \n{1}'.format(x, y))
            theta_opt.append( y )


    else:
        logger.debug('Detected gamma as a scalar')

        sigma_opt = np.sqrt( (a*c - b**2 + gamma**2 * C_0**2) / (c * gamma**2) )
        mu_opt = d/(c*gamma) + b*C_0/c

        logger.info('With gamma = {0}, (sigma_opt, mu_opt) = ({1}, {2})'.format(gamma, sigma_opt, mu_opt))

        theta_opt = Sigma_inv.dot(mu)/gamma + Sigma_inv.dot(ones_matrix) * (C_0 - b/gamma)/c
        logger.debug('\nWith gamma = {0}, theta_opt = \n{1}'.format(gamma, theta_opt))

    return sigma_opt, mu_opt, theta_opt






def monitor_sharpe_ratio(sigma, mu):
    ''' Calculate Sharpe ratio of given return and standard deviation

    Args:
        sigma (scalar or ndarray): 
        mu (scalar or ndarray): 

    Raises:

    Returns:
        sharpe_ratio (scalar, or ndarray): Sharpe Ratio
    '''

    if isinstance(sigma, (list, tuple, np.ndarray)) and isinstance(mu, (list, tuple, np.ndarray)):
        logger.debug('Detected sigma and mu as arrays')

        sharpe_ratio = np.divide(mu, sigma)

        for i in np.arange(sigma.shape[0]):
            logger.info('At ({0}, {1}), Sharpe Ratio = {2}'.format(sigma[i], mu[i], sharpe_ratio[i]))


    elif np.ndim(sigma) == 0 and np.ndim(mu) == 0:
        logger.debug('Detected sigma and mu as scalars')

        sharpe_ratio = np.divide(mu, sigma)

        logger.info('At ({0}, {1}), Sharpe Ratio = {2}'.format(sigma, mu, sharpe_ratio))

    else:
        if isinstance(sigma, (list, tuple, np.ndarray)) and np.ndim(mu) == 0:
            raise ValueError('sigma is an array and mu is a scalar!')

        else:
            raise ValueError('sigma is a scalar and mu is an array!')


    return sharpe_ratio


def start(raw_data, portfolio_types=['markowitz', 'minimum_variance'], monitors=['sharpe_ratio'], plot=False):
    ''' Start portfolio optimization

    Args:
        raw_data (string): 

    Raises:

    Returns:
        sigma_opt (scalar, or ndarray): 
        mu_opt (scalar, or ndarray): 
        theta_opt (ndarray, or array of ndarray): 
        sr_opt (scalar, or ndarray): Sharpe Ratio
    '''

    df_raw_data = get_raw_data(raw_data, plot)
    r = get_r(df_raw_data, plot)

    mu_p = np.arange(start=-0.005, stop=0.01, step=0.0001) # mu_p defines a range of possible portfolio returns
    C_0 = 1

    gamma = np.array([1, 2, 3, 4])
    a, b, c, d, mu, Sigma_inv = get_efficient_frontier_params(mu=None, Sigma=None, r=r)
    sigma_p, theta_ef = efficient_frontier(a, b, c, d, mu, Sigma_inv, C_0, mu_p)

    portfolios = []

    for portfolio_type in portfolio_types:
        if portfolio_type.lower() == 'minimum_variance':
            sigma, mu, theta = portfolio_minimum_variance(b, c, Sigma_inv, C_0)

            portfolio = {'portfolio_type': 'minimum_variance',
                        'sigma': sigma,
                        'mu': mu,
                        'theta': theta,
                        'data_type': 'array' if isinstance(sigma, (list, tuple, np.ndarray)) else 'scalar'}

        elif portfolio_type.lower() == 'tangency':
            sigma, mu, theta = portfolio_tangency(a, b, c, d, mu, Sigma_inv, C_0)

            portfolio = {'portfolio_type': 'tangency',
                        'sigma': sigma,
                        'mu': mu,
                        'theta': theta,
                        'data_type': 'array' if isinstance(sigma, (list, tuple, np.ndarray)) else 'scalar'}

        elif portfolio_type.lower() == 'markowitz':
            sigma, mu, theta = portfolio_markowitz(a, b, c, d, mu, Sigma_inv, C_0, gamma)

            portfolio = {'portfolio_type': 'markowitz',
                        'sigma': sigma,
                        'mu': mu,
                        'theta': theta,
                        'gamma': gamma,
                        'data_type': 'array' if isinstance(sigma, (list, tuple, np.ndarray)) else 'scalar'}

        else:
            raise ValueError('Currently support these portfolio types only: minimum_variance, tangency, markowitz')

        for monitor in monitors:
            if monitor.lower() == 'sharpe_ratio':
                portfolio['sharpe_ratio'] = monitor_sharpe_ratio(sigma, mu)
            
            else:
                raise ValueError('Currently support these monitors only: sharpe_ratio')

        
        portfolios.append(portfolio)





    if plot is True:
        plt.figure(figsize=(15,10))
        # plt.plot(<X AXIS VALUES HERE>, <Y AXIS VALUES HERE>, 'line type', label='label here')
        plt.plot(sigma_p, mu_p, label='Efficient Frontier')

        for portfolio in portfolios:
            if portfolio['data_type'] == 'scalar':
                plt.plot(portfolio['sigma'], portfolio['mu'], 'o', label=portfolio['portfolio_type'])

                s = '({0:.3f}, {1:.3f})'.format(portfolio['sigma'], portfolio['mu'])
                for key, value in portfolio.items():
                    if key not in ['portfolio_type', 'sigma', 'mu', 'theta', 'data_type']:
                        s = s + ', {0} = {1:.3f}'.format(key, value)

                plt.annotate(xy=[portfolio['sigma'], portfolio['mu']], s=s)
            
            elif portfolio['data_type'] == 'array':
                plt.plot(portfolio['sigma'], portfolio['mu'], 'o', label=portfolio['portfolio_type'])

                for i in np.arange(portfolio['sigma'].shape[0]):

                    s = '({0:.3f}, {1:.3f})'.format(portfolio['sigma'][i], portfolio['mu'][i])
                    for key, value in portfolio.items():
                        if key not in ['portfolio_type', 'sigma', 'mu', 'theta', 'data_type']:
                            s = s + ', {0} = {1:.3f}'.format(key, value[i])

                    plt.annotate(xy=[portfolio['sigma'][i], portfolio['mu'][i]], s=s)


        plt.title('Overview')
        plt.xlabel('Portfolio standard deviation (i.e. risk)')
        plt.ylabel('Mean portfolio return (i.e. return)')
        plt.legend(loc='best')


        plt.show()

    return 0