import numpy as np
from scipy.stats import norm

class Black_and_Scholes:
    def __init__(self, S, K, T, r, sigma, type):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.type = type.lower()
        
        d1 = (np.log(self.S/self.K) + (self.r + self.sigma**2/2)*self.T)/(self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        
        self.d1 = d1
        self.d2 = d2
        
    def black_scholes(self, new_vol=None) -> float:
        """Calcula o PREÇO de uma opção europeia usando a formula de Black-Scholes"""
        try:
            if self.type == 'c':
                price = self.S*norm.cdf(self.d1, 0, 1) - self.K*np.exp(-self.r*self.T)*norm.cdf(self.d2, 0, 1)
            elif self.type == 'p':
                price = self.K*np.exp(-self.r*self.T)*norm.cdf(-self.d2, 0, 1) - self.S*norm.cdf(-self.d1, 0, 1)
            
            return price
    
        except:
            print(f'You entered self.type = {self.type}, Please enter self.type C or P') 


    def delta(self) -> float:
        """Calcula o DELTA de uma opção europeia usando a formula de Black-Scholes"""
        self.type = self.type.lower()
        
        try:
            if self.type == 'c':
                delta = norm.cdf(self.d1, 0, 1) 
            elif self.type == 'p':
                delta = -norm.cdf(-self.d1, 0, 1)
            return delta
        
        except:
            print(f'You entered self.type = {self.type}, Please enter type c or c')
        
            
    def gamma(self) -> float:
        gamma = norm.pdf(self.d1, 0, 1) / (self.S*self.sigma*np.sqrt(self.T))
        return gamma        
    
            
    def vega(self) -> float:
        """Calcula o VEGA de uma opção europeia usando a formula de Black-Scholes"""
        vega = self.S*norm.pdf(self.d1, 0, 1)*np.sqrt(self.T)
        return vega*0.01 ## sensibilidade á mudandça de 1% na volatilidade
    
        
    def theta(self) -> float:
        """Calcula o THETA de uma opção europeia usando a formula de Black-Scholes"""
        try:
            if self.type == 'c':
                theta = (-self.S*norm.pdf(self.d1, 0, 1)*self.sigma) / (2*np.sqrt(self.T)) - (self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(self.d2, 0, 1))
            elif self.type == 'p':
                theta = (-self.S*norm.pdf(self.d1, 0, 1)*self.sigma) / (2*np.sqrt(self.T)) + (self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(-self.d2, 0, 1))
            return theta/365 ## theta em dias
        
        except:
            print(f'You entered self.type = {self.type}, Please enter self.type C or P') 
    
        
    def rho(self) -> float:
        """Calcula o RHO de uma opção europeia usando a formula de Black-Scholes"""        
        try:
            if self.type == 'c':
                rho = self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(self.d2)
            elif self.type == 'p':
                rho = -self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(-self.d2)
            
            return rho*0.01 
        
        except:
            print(f'You entered self.type = {self.type}, Please enter self.type C or P') 
    
        
    def implied_vol(self, market_price, tol=1e-5) -> float:
        """Calcula a Volatilidade Implicita de uma opção europeia"""
        max_iter = 200 # iterações
        vol_old = 0.3 # vol inicial
        
        for k in range(max_iter):            
            bs_price = self.black_scholes()
            Cprime = self.vega() * 100
            C = bs_price - market_price
            
            vol_new = vol_old - C / Cprime  
            ## atualiza sigma
            self.sigma = vol_new
            ## atualiza d1 e d2 com novo sigma e pega o novo preço da opção
            self.d1 = (np.log(self.S/self.K) + (self.r + self.sigma**2/2)*self.T) / (self.sigma*np.sqrt(self.T))
            self.d2 = self.d1 - self.sigma*np.sqrt(self.T)
            bs_price_new = self.black_scholes()    
            
            if (abs(vol_old-vol_new) < tol or abs(bs_price_new-market_price) < tol):
                break
            
            vol_old = vol_new
            
        implied_vol = vol_new
        
        return implied_vol   


