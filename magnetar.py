import numpy as np
from numpy import pi,exp,power
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import astropy.units as u
import astropy.constants as const

#Defining gauss in cgs base units
cgsgauss = (u.g)**0.5*(u.s)**(-1)*(u.cm)**(-0.5)

#Universal Constants with units
c = (const.c).cgs
G = (const.G).cgs
N_A = const.N_A
sigma = (const.sigma_sb).cgs
k_B = (const.k_B).to(u.keV/u.K)

def I_fun(M,R_NS):
    return 2*M*(power(R_NS,2))/5

def B_fun(t,B_0,B_decay=True):

    return np.where(B_decay==False,B_0,
                    np.maximum(B_0*exp(-(t/3.154e+07)/1e+06)/(1+((1e+06/(1e+19/B_0))*(1 - exp(-(t/3.154e+07)/1e+06)))),np.minimum(B_0/2,2e+13)))

def mu_fun(t,B_0,R_NS,B_decay=True):
    return B_fun(t,B_0,B_decay)*(power(R_NS,3))

def Rm_fun(t,Omega,delta,B_0,rho_0,v_0,M,R_NS,grav_correction=True,B_decay=True):

    Rm = power((1/(8*pi))*(power(mu_fun(t,B_0,R_NS,B_decay),2))*(power(v_0,delta-2))*(power(Omega,-delta))*(power(rho_0,-1)),1/(delta+6))
    
    return np.where(grav_correction == False or Rm>(2*G.to_value(u.cm**3/(u.g*u.s**2))*M/(power(v_0,2))),Rm,
                    power((1/(8*pi))*power(2*G.to_value(u.cm**3/(u.g*u.s**2))*M,(delta-5)/2)*power(mu_fun(t,B_0,R_NS,B_decay),2)*power(v_0,3)*power(Omega,-delta)*power(rho_0,-1),2/(3*delta+7)))

#def vm_fun(t,Omega,delta,B_0,rho_0,v_0,M,R_NS,grav_correction=True,B_decay=True):
    #Rm = Rm_fun(t,Omega,delta,B_0,rho_0,v_0,M,R_NS,grav_correction,B_decay)
    #return np.where(grav_correction == False or Rm>(2*G.to_value(u.cm**3/(u.g*u.s**2))*M/(power(v_0,2))),v_0,np.sqrt(2*G.to_value(u.cm**3/(u.g*u.s**2))*M/min(Rm,R_NS)))

#def rhom_fun(t,Omega,delta,B_0,rho_0,v_0,M,R_NS,grav_correction=True,B_decay=True):
    #Rm = Rm_fun(t,Omega,delta,B_0,rho_0,v_0,M,R_NS,grav_correction,B_decay)
    #return np.where(grav_correction == False or Rm>(2*G.to_value(u.cm**3/(u.g*u.s**2))*M/(power(v_0,2))),rho_0,(rho_0/(v_0**3))*(2*G.to_value(u.cm**3/(u.g*u.s**2))*M/min(Rm,R_NS))**(3/2))


#Dipole Phase spindown function
def dipole_spindown_fun(t,Omega,gamma,delta,B_0,v_0,rho_0,M,R_NS,grav_correction=True,B_decay=True):
    dOmega = -(2*power(mu_fun(t,B_0,R_NS,B_decay),2))/(3*power((c.to_value(u.cm/u.s)),3)*I_fun(M,R_NS))*power(Omega,3)
    return dOmega

#Dipole Phase termination event
def dipole_prop_transition(t,Omega,gamma,delta,B_0,v_0,rho_0,M,R_NS,grav_correction=True, B_decay=True):
    return Rm_fun(t,Omega,delta,B_0,rho_0,v_0,M,R_NS,grav_correction,B_decay)-(c.to_value(u.cm/u.s)/Omega)

#Propeller Phase spindown function
def propeller_spindown_fun(t,Omega,gamma,delta, B_0, v_0, rho_0, M, R_NS, grav_correction=True, B_decay=True):
    Rm = Rm_fun(t,Omega,delta,B_0,rho_0,v_0,M,R_NS,grav_correction,B_decay)
    return np.where(grav_correction == False or Rm>(2*G.to_value(u.cm**3/(u.g*u.s**2))*M/(power(v_0,2))),
                    -power(mu_fun(t,B_0,R_NS,B_decay),2)*power(Omega/v_0,gamma-delta)*power(Rm,gamma-delta-3)/(8*I_fun(M,R_NS)),
                    -power(2*G.to_value(u.cm**3/(u.g*u.s**2))*M,(delta-gamma)/2)*power(mu_fun(t,B_0,R_NS,B_decay),2)*power(Omega,gamma-delta)*power(Rm,(3*(gamma-delta)-6)/2)/(8*I_fun(M,R_NS)))
    
#Propeller Phase termination event
def prop_accretion_transition(t,Omega,gamma, delta, B_0, v_0, rho_0, M, R_NS, grav_correction=True, B_decay=True):    
    return Rm_fun(t,Omega,delta,B_0,rho_0,v_0,M,R_NS,grav_correction,B_decay)-power(G.to_value(u.cm**3/(u.g*u.s**2))*M/(Omega**2),1/3)

class magnetar:
    '''
    Defining the magnetar object which houses the necessary functions for running a single instance of spindown.
    
    Parameters:
    gamma: gamma exponent parameter (-1,0,1,2)
    delta: delta exponent parameter (0,1,2)
    B_0: initial value for the NS surface magnetic (in G)
    v_0: translational NS velocity (in km/s)
    rho_0: number density of material around the NS (in molecules/cm^3) 
    P0: initial spin period of the NS at t0
    t_obvs: age of the NS at the time of "observation" (in yr) 
    M: mass of the NS (in M_sun)
    R_NS: radius of the NS (in km)
    t0: time to begin integration
    grav_correction: toggle for enabling gravitational corrections (relevant for slower moving NSs)
    B_decay: toggle for enabling magnetic field decay
    dipole_only: toggle for dipole spindown only vs. dipole + propeller spindowns together
    '''

    def __init__(self, gamma, delta, B_0, v_0, n_0, P0, t_obvs, M=1.4, R_NS=10, t0=0, 
                 grav_correction=True, B_decay=True, dipole_only=False):
        self.gamma = gamma
        self.delta = delta
        self.B_0 = B_0*cgsgauss
        self.v_0 = v_0*(u.km/u.s)
        self.n_0 = n_0*(1/(u.cm**3))
        self.rho_0 = n_0*(1/(u.cm)**3)*(1*(u.g/u.mol))/N_A
        self.P0 = P0*(u.s)
        self.Omega0 = 2*pi/self.P0
        self.t0 = t0*(u.yr)
        self.t_obvs = t_obvs*(u.yr)
        self.M = M*const.M_sun
        self.R_NS = R_NS*u.km
        self.grav_correction = grav_correction
        self.B_decay = B_decay
        self.dipole_only = dipole_only
        
        self.obvs_type = '0'

    def run(self):

        if self.dipole_only==True:
            dipole_prop_transition.terminal=False
            prop_accretion_transition.terminal=True
        else:
            dipole_prop_transition.terminal=True
            prop_accretion_transition.terminal=True
        
        dipole_integrator = solve_ivp(fun=dipole_spindown_fun,t_span=(self.t0.to_value(u.s),self.t_obvs.to_value(u.s)),y0=np.array([self.Omega0.to_value(1/u.s)]),
                                      args=(self.gamma,self.delta,self.B_0.to_value(cgsgauss),self.v_0.to_value(u.cm/u.s),self.rho_0.to_value(u.g/(u.cm**3)),self.M.to_value(u.g),self.R_NS.to_value(u.cm),self.grav_correction,self.B_decay),
                                      t_eval=np.logspace(-1,0.99*np.log10(self.t_obvs.to_value(u.s)),100000),events=dipole_prop_transition,method='RK45',max_step=1E+12,rtol=1E-5,atol=1E-26)
        
        if self.dipole_only == True or dipole_integrator.t[-1]==self.t_obvs.to_value(u.s):
            self.obvs_type = '1'
            self.t_arr = dipole_integrator.t
            self.Omega_arr = dipole_integrator.y[0]
            self.dOmega_arr = dipole_spindown_fun(self.t_arr,self.Omega_arr,self.gamma,self.delta,self.B_0.to_value(cgsgauss),self.v_0.to_value(u.cm/u.s),self.rho_0.to_value(u.g/(u.cm**3)),self.M.to_value(u.g),self.R_NS.to_value(u.cm),self.grav_correction,self.B_decay)
            
        else:
            propeller_integrator = solve_ivp(fun=propeller_spindown_fun,t_span=(dipole_integrator.t[-1],self.t_obvs.to_value(u.s)),y0=np.array([dipole_integrator.y[0][-1]]),
                                             args=(self.gamma,self.delta,self.B_0.to_value(cgsgauss),self.v_0.to_value(u.cm/u.s),self.rho_0.to_value(u.g/(u.cm**3)),self.M.to_value(u.g),self.R_NS.to_value(u.cm),self.grav_correction,self.B_decay),
                                             t_eval=np.logspace(np.log10(dipole_integrator.t[-1]),0.99*np.log10(self.t_obvs.to_value(u.s)),100000),events=prop_accretion_transition,method='RK45',max_step=1E+12,rtol=1E-5,atol=1E-26)
            
            self.t_arr = np.append(dipole_integrator.t,propeller_integrator.t[1:])
            self.Omega_arr = np.append(dipole_integrator.y[0],propeller_integrator.y[0][1:])
            dipole_dOmega_arr = dipole_spindown_fun(dipole_integrator.t,dipole_integrator.y[0],self.gamma,self.delta,self.B_0.to_value(cgsgauss),self.v_0.to_value(u.cm/u.s),self.rho_0.to_value(u.g/(u.cm**3)),self.M.to_value(u.g),self.R_NS.to_value(u.cm),self.grav_correction,self.B_decay)
            prop_dOmega_arr = propeller_spindown_fun(propeller_integrator.t[1:],propeller_integrator.y[0][1:],self.gamma,self.delta,self.B_0.to_value(cgsgauss),self.v_0.to_value(u.cm/u.s),self.rho_0.to_value(u.g/(u.cm**3)),self.M.to_value(u.g),self.R_NS.to_value(u.cm),self.grav_correction,self.B_decay)
            self.dOmega_arr = np.append(dipole_dOmega_arr,prop_dOmega_arr)

            if propeller_integrator.t[-1]==self.t_obvs.to_value(u.s):
                self.obvs_type = '2'

            else:
                self.obvs_type = '3'
        
        self.B_arr = (B_fun(self.t_arr,self.B_0.to_value(cgsgauss),B_decay=self.B_decay))*cgsgauss
        self.t_arr = (self.t_arr*u.s).to(u.yr)
        self.Omega_arr = self.Omega_arr*(1/u.s)
        self.dOmega_arr = self.dOmega_arr*(1/(u.s**2))
        self.P_arr = 2*pi/(self.Omega_arr)
        self.P_dot_arr = -(1/(2*pi))*self.dOmega_arr*(self.P_arr**2)
        self.dP_arr = -2*pi*self.dOmega_arr/(self.Omega_arr**2)
        self.distance = (self.v_0*self.t_arr[-1]).to(u.kpc)

        if self.obvs_type == '3':
            mu_end = mu_fun(self.t_arr[-1].to_value(u.s),self.B_0.to_value(cgsgauss),self.R_NS.to_value(u.cm),self.B_decay)
            self.M_dot = (9.6*(10**7)*(u.g/u.s))*((mu_end/(10**33))**(2/3))*((self.v_0.to_value(u.cm/u.s)/(10**7))**(1/3))*((self.rho_0.to_value(u.g/(u.cm**3))*N_A.to_value(1/u.mol)/(0.1))**(2/3))
            self.L_X = (1.8*(10**28)*(u.erg/u.s))*((mu_end/(10**33))**(2/3))*((self.v_0.to_value(u.cm/u.s)/(10**7))**(1/3))*((self.rho_0.to_value(u.g/(u.cm**3))*N_A.to_value(1/u.mol)/(0.1))**(2/3))*(self.M/const.M_sun)*((self.R_NS.to_value(u.cm)/(10**6))**(-1))
            self.kT_eff = (0.39*(u.keV))*((mu_end/(10**33))**(1/4))*((self.rho_0.to_value(u.g/(u.cm**3))*N_A.to_value(1/u.mol)/(0.1))**(1/8))*((self.M/const.M_sun)**(1/4))*((self.R_NS.to_value(u.cm)/(10**6))**(-1))
        else:
            self.M_dot = 0*(u.g/u.s)
            self.L_X = 0*(u.erg/u.s)
            self.kT_eff = 0*(u.keV)