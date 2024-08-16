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


def delta(S, K, T, r, sigma, type='C') -> float:
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
  
    
def theta(S, K, T, r, sigma, type='C') -> float:
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
  
    
def rho(S, K, T, r, sigma, type='C') -> float:
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
## RETURN and ANNUALIZED VOLATILITY FUCTIONS

def annualized_returns(r: pd.DataFrame, periodos=252):
    r = r[-periodos:]
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    per_rets = (compounded_growth**(periodos/n_periods)-1).iloc[0]
    return per_rets
    
def cc_volatility(precos_fechamento: pd.DataFrame, periodos: int=20, annualized=True):
    rets = precos_fechamento / precos_fechamento.shift(1)-1
    cc_vol = rets.rolling(periodos).std()
    if annualized:
        return cc_vol * np.sqrt(252)
    else:
        return cc_vol  * np.sqrt(periodos)

def parkinson_volatility(df: pd.DataFrame, periodos: int=20, annualized=True):
    hi_lo = (np.log(df['High'] / df['Low'])) ** 2
    pk_vol = np.sqrt(hi_lo.rolling(periodos).mean() / 4 * np.log(2)) 
    if annualized:
        return pk_vol * np.sqrt(252)
    else: 
        return pk_vol * np.sqrt(periodos)

def garman_klass_volatility(df: pd.DataFrame, periodos: int=20, annualized=True):
    hi_lo = (np.log(df['High'] / df['Low'])) ** 2
    cl_op = (np.log(df['Adj Close'] / df['Open'])) ** 2
    gk = 0.5 * hi_lo - (2 * np.log(2) - 1) * cl_op
    gk_vol = (np.sqrt(gk.rolling(periodos).mean())) 
    if annualized:
        return gk_vol * np.sqrt(252)
    else:
        return gk_vol * np.sqrt(periodos)

def rogers_satchell_volatility(df: pd.DataFrame, periodos: int=20, annualized=True):
    hi_cl = np.log(df['High'] / df['Adj Close'])
    hi_op = np.log(df['High'] / df['Open'])
    lo_cl = np.log(df['Low'] / df['Adj Close'])
    lo_op = np.log(df['Low'] / df['Open'])
    rs = np.sqrt((hi_cl * hi_op + lo_cl * lo_op).rolling(periodos).mean())
    rs_vol = rs 
    if annualized:
        return rs_vol * np.sqrt(252)
    else:
        return rs_vol * np.sqrt(periodos)

def yang_zhang_volatility(df: pd.DataFrame, periodos: int=20, annualized=True): 
    hi_cl = np.log(df['High'] / df['Adj Close'])
    hi_op = np.log(df['High'] / df['Open'])
    lo_cl = np.log(df['Low'] / df['Adj Close'])
    lo_op = np.log(df['Low'] / df['Open'])
    rs = np.sqrt((hi_cl * hi_op + lo_cl * lo_op).rolling(periodos).mean())
    
    op_cl_1 = np.log(df['Open'] / df['Adj Close'].shift(1) )
    # hi_op = np.log(df['High'] / df['Open'] )
    # lo_op = np.log(df['Low'] / df['Open'] )
    cl_op = np.log(df['Adj Close'] / df['Open'] )

    open_close_vol = (cl_op.rolling(periodos).std())
    overnight_vol = (op_cl_1.rolling(periodos).std())

    k = 0.34 / (1.34 + (periodos+1) / (periodos-1) )
 
    yz = np.sqrt(overnight_vol**2 + k*open_close_vol**2 + (1 - k)*rs**2)
    yz_vol = yz 
    if annualized:
        return yz_vol * np.sqrt(252)
    else: 
        return yz_vol * np.sqrt(periodos)

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


def atm_straddle_approximation_short(S: float, ann_sigma: float, dte: int): 
    atm_straddle = S * ann_sigma * np.sqrt(dte/252)
    lower = (S - atm_straddle) 
    upper = (S + atm_straddle)

    print(f'ATM Straddle: {atm_straddle:.2f}')
    print(f'Range Expectation using ATM Straddle Approximation {lower:.2f} e {upper:.2f}')
    
    return atm_straddle, lower, upper 


def mad_straddle_approximation_short(S: float, ann_sigma: float, dte: int): 
    """This is the aproxximation Formula of ATM Straddle"""
    mad_straddle = (4/5) * S * ann_sigma * np.sqrt(dte/252)
    lower = (S - mad_straddle) 
    upper = (S + mad_straddle)

    print(f'MAD Straddle: {mad_straddle:.2f}')
    print(f'Range Expectation using MAD Straddle {lower:.2f} e {upper:.2f}')
    
    return mad_straddle, lower, upper 

#-------------------------------------------------------------------------------------#
# IRON_CONDOR FUNCTIONS

def select_iron_condor_strikes(lower: float, upper: float, options: pd.DataFrame, wing_width: float):
    puts = options[options['Type'] == 'PUT']
    calls = options[options['Type'] == 'CALL'] 
    
    short_put = puts.iloc[(puts['Strike'] - lower).abs().argsort().iloc[0]]['Strike']
    short_call = calls.iloc[(calls['Strike'] - upper).abs().argsort().iloc[0]]['Strike']
    
    long_put = puts[(puts['Strike'] == short_put - wing_width )]['Strike'].iloc[0]
    long_call = calls[(calls['Strike'] == short_call + wing_width )]['Strike'].iloc[0]

    return long_put, short_put, short_call, long_call


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
        roc = ((credit_received / (max_loss)) * take_profit * (1-taxes_cost)) * qty
        ## trade is closed when 65% of max profit is reached
        profit = (credit_received*take_profit*(1-taxes_cost)) 
        
        print(f'Position Risk: {max_loss:.2f}')
        print(f'Gain Range: {gain_range:.2f}')
        print(f'Credit Received/Max Profit: ${credit_received*qty:.2f}')
        print(f'Managed Take Profit: ${profit*qty:.2f}')
        print(f'Managed ROIC (net): {roc:.2%}')

    else:
        leg_width = 'Distances not equal'
        print(f"""{leg_width}\nCall distance: {(short_put_strike - long_put_strike)}\nPut distance: {(long_call_strike - short_call_strike)}""")
    
    return max_loss, gain_range, credit_received, profit, roc, leg_width

#-------------------------------------------------------------------------------------#
# MONTE_CARLO and KELLY_CRITERION FUNCTIONS

def gbm(n_years: float, n_scenarios: int, mu: float, sigma: float, steps_per_year: int, S: float):
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=(mu * dt + 1), scale=(sigma * np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    prices = S * pd.DataFrame(rets_plus_1).cumprod()
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


def kelly_criterion(p, b, kelly=0.5, portfolio_size=10000):
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
    kelly_fraction = ((b * p - q) / b) * kelly
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