import numpy as np  
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import requests
import yfinance as yf
from datetime import datetime
import math

#-------------------------------------------------------------------------------------#
# GET_DATA FUNCTIONS

def get_stock_data(ticker, period):
    return yf.download(ticker, period=period)


def get_options_chain(underlying: str, exp_date: str) -> pd.DataFrame:
    url = f'https://opcoes.net.br/listaopcoes/completa?idAcao={underlying}&listarVencimentos=False&cotacoes=True&Vencimentos={exp_date}'
    r = requests.get(url).json()
    l = [ [ i[0].split('_')[0], i[2], i[3], i[4] ,i[5], i[6]*100, i[8], i[10]] for i in r['data']['cotacoesOpcoes'] ] 
    chain = pd.DataFrame(l, columns=['Option', 'Type', 'E/A', 'Moneyness', 'Strike', 'Distance', 'Premium', 'volume'] )
    chain['abs_Distance'] = chain['Distance'].abs() 
    return chain

#-------------------------------------------------------------------------------------#
# BLACK_SCHOLES FUNCTIONS

def black_scholes(S, K, T, r, sigma, type='C') -> float:
    """ Calculates the BS option price for a call/put

        S: stock price
        K: strike price
        T: time to maturity in years
        r: interest rate
        market_price: option price in market
        type: call or put
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    type = type.lower()
    
    try:
        if type == 'c':
            price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == 'p':
            price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
        
        return price
    
    except:
        print(f'You entered type = {type}, Please enter type C or P') 


def delta(S, K, T, r, sigma, type='p') -> float:
    """ Calculates the delta of an european option"""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    type = type.lower()
    
    try:
        if type == 'c':
            delta = norm.cdf(d1, 0, 1) 
        elif type == 'p':
            delta = -norm.cdf(-d1, 0, 1)
        return delta
    
    except:
        print(f'You entered type = {type}, Please enter type C or P')
    
        
def gamma(S, K, T, r, sigma) -> float:
    """ Calculates the gamma of an european option"""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    gamma = norm.pdf(d1, 0, 1) / (S*sigma*np.sqrt(T))
    return gamma        
   
         
def vega(S, K, T, r, sigma) -> float:
    """ Calculates the vega of an european option"""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    vega = S*norm.pdf(d1, 0, 1)*np.sqrt(T)
    return vega*0.01 ## sensitivity to 1% change in volatility 
  
    
def theta(S, K, T, r, sigma, type='p') -> float:
    """ Calculates the theta of an european option"""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    type = type.lower()
    
    try:
        if type == 'c':
            theta = (-S*norm.pdf(d1, 0, 1)*sigma) / (2*np.sqrt(T)) - (r*K*np.exp(-r*T)*norm.cdf(d2, 0, 1))
            # theta=-S*norm.pdf(d1, 0, 1)*sigma  / (2*np.sqrt(T)) -  r*K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == 'p':
            theta = (-S*norm.pdf(d1, 0, 1)*sigma) / (2*np.sqrt(T)) + (r*K*np.exp(-r*T)*norm.cdf(-d2, 0, 1))
        return theta/365 ## time decays in days
    
    except:
        print(f'You entered type = {type}, Please enter type C or P') 
  
    
def rho(S, K, T, r, sigma, type='p') -> float:
    """ Calculates the rho of an european option"""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    type = type.lower()
    
    try:
        if type == 'c':
            rho = K*T*np.exp(-r*T)*norm.cdf(d2)
        elif type == 'p':
            rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)
        
        return rho*0.01 ## sensitivity to 1% change in interest rate 
    
    except:
        print(f'You entered type = {type}, Please enter type C or P') 
  
    
def implied_vol(S, K, T, r, market_price, type='C', tol=1e-5) -> float:
    """ Calculates the implied volatility of an european option

        S: stock price
        K: strike price
        T: time to maturity in years
        r: interest rate
        sigma: volatility 
        market_price: option price in market
        type: call or put
        tol: Defaults to 1e-5.
    """
    max_iter = 200 # max no. of iterations
    vol_old = 0.3 # initial guess
    type = type.lower()
    
    for k in range(max_iter):                  
        bs_price = black_scholes(S, K, T, r, vol_old, type)               
        Cprime = vega(S, K, T, r, vol_old) * 100
        C = bs_price - market_price
        
        vol_new = vol_old - C/Cprime
        # bs_price_new = bs(type, S, K, T, r, vol_new)
        bs_price_new = black_scholes(S, K, T, r, vol_new, type='C')     
        
        if (abs(vol_old-vol_new) < tol or abs(bs_price_new-market_price) < tol):
            break
        
        vol_old = vol_new
        
    implied_vol = vol_new
    
    return implied_vol   

#-------------------------------------------------------------------------------------#
## VOLATILITIES/RETURN FUCTIONSand EXPECTED_MOVE FUNCTIONS

def annualize_rets(r, periods_per_year=252):
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return (compounded_growth**(periods_per_year/n_periods)-1).iloc[0]
    
def realized_cc_vol(prices, periods_per_year=252):
    returns = prices.pct_change()
    realized_volatility = np.std(returns.dropna()) * np.sqrt(periods_per_year)  
    return realized_volatility

def parkinson_volatility(df):
     N = df.shape[0]
     k = (1/(4 * N * np.log(2)))
     hi_lo = np.sum(np.log(df['High'] / df['Low']) ** 2)
     
     return np.sqrt(k*hi_lo) * np.sqrt(252)

def yz_volatility(df, dte): ## FIX
    df = df[-dte:-1]
    N = df.shape[1]
    open = df['Open']
    close = df['Close']
    hi = df['High']
    lo = df['Low']
    
    sigma_open = (1 / N -1 ) * np.sum([np.log(open.iloc[i] / open.iloc[i-1]) - 1 / N * np.sum(np.log(open.iloc[i] / open.iloc[i-1]) ) for i, o in enumerate(open)]) ** 2
    sigma_close = (1 / N -1 ) * (np.sum([np.log(close.iloc[i] / close.iloc[i-1]) - 1 / N * np.sum(np.log(close.iloc[i] / close.iloc[i-1]) ) for i, o in enumerate(close)]) ** 2)
    sigma_hilo = (1 / N ) * np.sum([np.log(hi.iloc[i] / close.iloc[i-1]) * np.log(hi.iloc[i] / open.iloc[i-1]) + np.log(lo.iloc[i] / close.iloc[i-1]) * np.log(lo.iloc[i] / open.iloc[i-1]) for i, o in enumerate(close)])
    k = 0.34 / (1.34 + (N+1) / (N-1) )
    
    yz_vol = np.sqrt(sigma_open + k*sigma_close + (1 - k)*sigma_hilo) * np.sqrt(252)
    
    return yz_vol 

#-------------------------------------------------------------------------------------#
## EXPECTED_MOVE FUNCTIONS

def atm_straddle_short(options: pd.DataFrame, S: float):
    atm_strike = options.iloc[(options['Distance']-0).abs().argsort()[:1]].reset_index()['Strike'][0]
    atm_straddle = options[options['Strike'] == atm_strike]['Premium'].sum()

    lower = (S - atm_straddle) 
    upper = (S + atm_straddle)

    print(f'ATM Straddle: {atm_straddle:.2f}')
    print(f'Range Expectation using ATM Straddle {lower:.2f} e {upper:.2f}')
    
    return atm_straddle, lower, upper 


def atm_straddle_approximation_short(S, ann_sigma, dte): 
    atm_straddle = S * ann_sigma * np.sqrt(dte/252)
    lower = (S - atm_straddle) 
    upper = (S + atm_straddle)

    print(f'ATM Straddle: {atm_straddle:.2f}')
    print(f'Range Expectation using ATM Straddle Approximation {lower:.2f} e {upper:.2f}')
    
    return atm_straddle, lower, upper 


def mad_straddle_approximation_short(S, ann_sigma, dte): 
    """This is the aproxximation Formula of ATM Straddle"""
    mad_straddle = (4/5) * S * ann_sigma * np.sqrt(dte/252)
    lower = (S - mad_straddle) 
    upper = (S + mad_straddle)

    print(f'MAD Straddle: {mad_straddle:.2f}')
    print(f'Range Expectation using MAD Straddle {lower:.2f} e {upper:.2f}')
    
    return mad_straddle, lower, upper 

#-------------------------------------------------------------------------------------#
# IRON_CONDOR FUNCTIONS

def select_iron_condor_strikes(function, options, wing_width, S):

    puts = options[options['Type'] == 'PUT']
    calls = options[options['Type'] == 'CALL']

    atm_straddle, lower, upper = function(options, S)
    
    # Select the short put and call closest to the range defined by volatility
    short_put = puts.iloc[(puts['Strike']-lower).abs().argsort()[:1]].reset_index(drop=True)
    short_call = calls.iloc[(upper- calls['Strike']).abs().argsort()[:1]].reset_index(drop=True)
    
    # Select the long put and call to create the wings of the iron condor
    long_put = puts[puts['Strike'] < short_put['Strike'].values[0]].iloc[(-puts[puts['Strike'] < short_put['Strike'].values[0]]['Strike'] + (short_put['Strike'].values[0] - wing_width)).abs().argsort()[:1]]
    long_call = calls[calls['Strike'] > short_call['Strike'].values[0]].iloc[(calls[calls['Strike'] > short_call['Strike'].values[0]]['Strike'] - (short_call['Strike'].values[0] + wing_width)).abs().argsort()[:1]]
    
    long_put_strike = long_put['Strike'].iloc[0]
    short_put_strike = short_put['Strike'].iloc[0] 
    short_call_strike = short_call['Strike'].iloc[0] 
    long_call_strike = long_call['Strike'].iloc[0] 
    
    return long_put_strike, short_put_strike, short_call_strike, long_call_strike


def iron_condor(options, long_put_strike, short_put_strike, short_call_strike, long_call_strike, qty=10, take_profit=.65, taxes_cost=.2):
    long_put = options[(options['Strike'] == long_put_strike) & (options['Type'] == 'PUT')]
    short_put = options[(options['Strike'] == short_put_strike) & (options['Type'] == 'PUT')]
    short_call = options[(options['Strike'] == short_call_strike) & (options['Type'] == 'CALL')]
    long_call = options[(options['Strike'] == long_call_strike) & (options['Type'] == 'CALL')] 
    
    ## Collecting the premiuns from the options
    long_put_premium = long_put['Premium'].iloc[0] 
    short_put_premium = short_put['Premium'].iloc[0] 
    short_call_premium = short_call['Premium'].iloc[0] 
    long_call_premium = long_call['Premium'].iloc[0] 

    ## Calculates the credit received
    credit_received = (-long_put_premium +short_put_premium +short_call_premium -long_call_premium)
    credit_received

    ## Checking
    if (short_put_strike - long_put_strike) == (long_call_strike - short_call_strike):
        leg_width = (short_put_strike - long_put_strike)
        gain_range = (short_call_strike - short_put_strike)
        max_loss = (leg_width - ((credit_received*qty) / qty)) * qty
        roc_cost = ((credit_received / (max_loss)) * take_profit * (1-taxes_cost)) * qty
        ## trade is closed when 65% of max profit is reached
        profit = (credit_received*take_profit*(1-taxes_cost)) *qty
        
        print(f'Position Risk: {max_loss:.2f}')
        print(f'Gain Range: {gain_range:.2f}')
        print(f'Credit Received/Max Profit: ${credit_received*qty:.2f}')
        print(f'Managed Take Profit: ${profit*qty:.2f}')
        print(f'Managed ROIC (net): {roc_cost:.2%}')

    else:
        leg_width = 'Distances not equal'
        print(f"""{leg_width}\nCall distance: {(short_put_strike - long_put_strike)}\nPut distance: {(long_call_strike - short_call_strike)}""")
    
    return max_loss, gain_range, credit_received, profit, roc_cost, leg_width

#-------------------------------------------------------------------------------------#
# MONTE_CARLO and KELLY_CRITERION FUNCTIONS

def gbm(n_years=45/252, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=252, s_0=100.0):
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=(mu * dt + 1), scale=(sigma * np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    prices = s_0 * pd.DataFrame(rets_plus_1).cumprod()
    return prices


def plot_paths(gbm_df, ceiling=None, floor=None):
    final_prices = gbm_df.iloc[-1]
    n_scenarios = gbm_df.shape[1]
    
    # Create a figure with two subplots: one for the paths and one for the sideways distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True
                                   , gridspec_kw={'width_ratios': [3, 2]}
                                   )
    
    # Plot the paths
    ax1.plot(gbm_df, linewidth=1, alpha=.75, color='lightblue')
    ax1.set_ylabel('Preços')
    ax1.set_xlabel('Dias')
    ax1.set_title('Caminhos Possiveis')
    if ceiling is not None:
        ax1.axhline(y=ceiling, color='red', linestyle='--', label=f'Limite Sup.: {ceiling:.2f}', alpha=.35)
    if floor is not None:
        ax1.axhline(y=floor, color='red', linestyle='--', label=f'Limite Inf.: {floor:.2f}', alpha=.35)

    if ceiling is not None or floor is not None:
        ax1.legend()
    
    # Count paths outside the limits
    if ceiling is not None:
        above_ceiling = final_prices > ceiling
        num_above_ceiling = above_ceiling.sum()
    else:
        num_above_ceiling = 0
    
    if floor is not None:
        below_floor = final_prices < floor
        num_below_floor = below_floor.sum()
    else:
        num_below_floor = 0
    
    num_paths_inside_limits = n_scenarios - (num_above_ceiling + num_below_floor)
 
    # Plot the distribution of final prices sideways (horizontal histogram)
    ax2.hist(final_prices, bins=30, orientation='horizontal', color='lightblue', edgecolor='blue', alpha=.65)
    ax2.set_xlabel('Frequencia')
    ax2.set_title('Distribuição final do Preço')
    
    if ceiling is not None:
        ax2.axhline(y=ceiling, color='red', linestyle='--', label=f'Limite Sup.: {ceiling:.2f}', alpha=.35)
    if floor is not None:
        ax2.axhline(y=floor, color='red', linestyle='--', label=f'Limite Inf.: {floor:.2f}', alpha=.35)
    # if ceiling is not None or floor is not None:
    #     ax2.legend()
    
    # Add text annotations to the histogram plot
    loss_above = num_above_ceiling/n_scenarios
    win_pct = num_paths_inside_limits/n_scenarios
    loss_bellow = num_below_floor/n_scenarios
    
    textstr = '\n'.join((
        f'acima do limite: {loss_above:.2%} ({num_above_ceiling})',
        f'nos limites: {win_pct:.2%} ({num_paths_inside_limits})',
        f'abaixo do limite: {loss_bellow:.2%} ({num_below_floor})',
    ))  
    # Position the text box on the bottom-right of the histogram plot
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax2.text(0.95, 0.05, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()  # Adjust layout to make sure there's no overlap
    plt.show()

    return win_pct, loss_above, loss_bellow


def kelly_criterion(p, b, bet_factor=0.5, portfolio_size=10000):
    """
    Calculate the position size fraction based on a fraction of the Kelly Criterion.
    p: Probability of the strategy being profitable.
    b: Profit-to-loss ratio of the strategy.
    portfolio_size: total amount of money
    
    returns:
    kelly_fraction: kelly_fraction
    max_risk: maximum monetary value to risk  
    """

    q = 1 - p  # Probability of strategy being unprofitable

    # Calculate the Kelly fraction
    kelly_fraction = ((b * p - q) / b) * bet_factor
    max_risk = portfolio_size*kelly_fraction

    # Return a fraction of the Kelly fraction
    return kelly_fraction, max_risk


#-------------------------------------------------------------------------------------#
# DEPRECATED 

# def sd_straddle_short(S: float, sigma: float, sd: float=2): ## same as atm_straddle_approximation_short
#     lower = ((1 - sd  * sigma) * S)
#     upper = ((1 + sd * sigma) * S)
#     print(f'Range Expectation using {sd} sd Straddle {lower:.2f} e {upper:.2f}')
#     return lower, upper