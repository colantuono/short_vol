import pandas as pd
import numpy as np
import requests
import yfinance as yf

def get_options_chain(underlying: str, exp_date: str) -> pd.DataFrame:
    url = f'https://opcoes.net.br/listaopcoes/completa?idAcao={underlying}&listarVencimentos=False&cotacoes=True&Vencimentos={exp_date}'
    r = requests.get(url).json()
    l = [ [ i[0].split('_')[0], i[2], i[3], i[4] ,i[5], i[6]*100, i[8], i[10]] for i in r['data']['cotacoesOpcoes'] ] 
    return pd.DataFrame(l, columns=['Option', 'Type', 'E/A', 'Moneyness', 'Strike', 'Distance', 'Premium', 'volume'] )

def annualize_vol(r, periods_per_year=252):
    """
    Annualizes the vol of a set of returns
    """
    return r.std()*(periods_per_year**0.5)


def vol_based_short(last_price: float, vol: float, sd=2):
    lower = ((1 - sd * vol) * last_price)
    upper = ((1 + sd * vol) * last_price)
    
    print(f'Range Expectation using ATM Straddle {lower:.2f} e {upper:.2f}')
    
    return lower, upper


def atm_short_straddle(options, last_price: float):
    atm_strike = options.iloc[(options['Distance']-0).abs().argsort()[:1]].reset_index()['Strike'][0]
    atm_straddle = options[options['Strike'] == atm_strike]['Premium'].sum()

    lower = (last_price - atm_straddle) 
    upper = (last_price + atm_straddle)

    print(f'ATM Straddle: {atm_straddle:.2f}')
    print(f'Range Expectation using ATM Straddle {lower:.2f} e {upper:.2f}')
    
    return atm_straddle, lower, upper 

    
def select_iron_condor_strikes(options, underlying_price, wing_width=5):   
    puts = options[options['Type'] == 'PUT']
    calls = options[options['Type'] == 'CALL']

    atm_straddle, lower, upper = atm_short_straddle(options, underlying_price)

    
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
        # pop = 1 - (prob_exercicio) ## highest delta between short options
        profit = (credit_received*take_profit*(1-taxes_cost))
        
        print(f'Position Risk: {max_loss:.2f}')
        print(f'Gain Range: {gain_range:.2f}')
        print(f'Credit Received/Max Profit: ${credit_received*qty:.2f}')
        print(f'Managed Take Profit: ${profit*qty:.2f}')
        print(f'Managed ROIC (net): {roc_cost:.2%}')
        # print(f'Probability of Profit: {pop:.2%}')
    else:
        leg_width = 'Distances not equal'
        print(f"""{leg_width}\nCall distance: {(short_put_strike - long_put_strike)}\nPut distance: {(long_call_strike - short_call_strike)}""")
    
    return max_loss, gain_range, credit_received, profit, roc_cost, leg_width
 
## KELLY CRITERION 

# p = pop
# q = 1 - pop
# b = profit / risco

def kelly_criterion(p,q,b):
    kc = (p - q)/b
    return round(kc, 2) 

def trade_capital(portfolio, trade_aloc, kc):
    alloc1 = portfolio * trade_aloc
    print(alloc1)
    return round(alloc1 * kc, 2)

# risco_max = trade_capital(1000, 0.025, kc);risco_max

def lotes(risco_max, risco):
    return round(risco_max / risco)

# lotes_trade = lotes(risco_max, risco) 
# risco_max, lotes_trade





def iron_condor_old(long_put, short_put, short_call, long_call, credito, prob_exercicio, lotes=1, take_profit=.65, ir_taxas =.20):
    intervalo = (short_call - short_put)
    distancia_pernas = []
    if (long_call - short_call) == (short_put - long_put):
        distancia_pernas.append((long_call - short_call))
        risco = (distancia_pernas[0] - ((credito*lotes) / lotes)) * lotes
        roc_ir = ((credito / (risco)) * take_profit * (1-ir_taxas)) * lotes
        ## trade is closed when 65% of max profit is reached
        pop = 1 - (prob_exercicio) ## highest delta between short options
        # max_profit = 
        profit = (credito*take_profit*(1-ir_taxas))
        
        print(f'Risco da operação: {risco:.2f}')
        print(f'Intervalo de Ganho: {intervalo:.2f}')
        print(f'Max Profit: ${credito*lotes:.2f}')
        print(f'Managed Take Profit: ${profit*lotes:.2f}')
        print(f'Managed ROIC (net): {roc_ir:.2%}')
        # print(f'Probability of Profit: {pop:.2%}')
    else:
        perna_call = (long_call - short_call)
        perna_put  = (long_put - short_put)
        print(f'Distancias Diferentes, ajuste os Strikes\nCall: {perna_call}\nPut:{perna_put}')
        
    return pop, risco, credito