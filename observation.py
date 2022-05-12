#>>> import observation as ob
#>>> me = ob.observation().classical().new_RGB().new_MS().new_fdnu()
import os
os.chdir('/home/life/asfgrid/')
import asfgrid
from math import e
from uncertainties import ufloat
from uncertainties.unumpy import isnan
import numpy as np
from warnings import warn
from lightkurve import seismology as seis

class observation():
    def __init__(self):
        message = [[  232623464,          97.17,          1.264,           8.24,          0.058,      4846.77,    33.83,       0.165,       0.030],
                   [  224292441,          59.36,          0.464,           5.67,          0.043,      4610.80,     7.09,       0.074,       0.006],
                   [  149399573,         125.73,          0.829,          10.82,          0.051,      4690.95,    30.70,       0.150,       0.028],
                   [   27002591,         106.00,          2.624,           8.40,          0.118,      5063.55,    25.68,      -0.136,       0.021],
                   [  458690359,          80.72,          0.766,           7.43,          0.038,      4469.92,    37.04,       0.287,       0.034],
                   [  459769560,          71.15,          0.431,           6.86,          0.035,      4705.28,    30.66,      -0.219,       0.028],
                   [    3716092,         170.91,          0.852,          14.42,          0.656,      5004.67,    18.33,      -0.682,       0.016],
                   [    4753562,          84.51,          0.547,           8.53,          0.066,      4851.22,    22.65,      -0.341,       0.018],
                   [   11938650,         152.06,          1.298,          13.95,          0.104,      4861.84,    17.37,      -0.063,       0.014],
                   [  105489715,         107.59,          0.857,           9.40,          0.125,      4809.83,    22.91,       0.028,       0.019],
                   [  302750867,         156.99,          0.607,          12.96,          0.131,      4681.75,    43.54,       0.204,       0.035],
                   [  252796304,         159.45,          0.827,          13.02,          0.094,      4713.17,    46.47,       0.032,       0.045],
                   [    8901893,         108.15,          0.657,           9.19,          0.116,      4650.52,    28.23,       0.209,       0.025],
                   [  137649123,         414.72,         20.983,          25.18,          0.869,      4761.02,    30.34,       0.269,       0.027] ]
        self.id   = [str(i[0]) for i in message]
        self.nu   = [ufloat(i[1],i[2]) for i in message] 
        self.dnu  = [ufloat(i[3],i[4]) for i in message] 
        self.teff = [ufloat(i[5],i[6]) for i in message] 
        self.fe_h = [ufloat(i[7],i[8]) for i in message] 
        '''
        message = np.array([['TIC232623464', ufloat( 99.59,  1.71 ), ufloat(  8.20,  0.08 ), ufloat(4846.77,33.83), ufloat( 0.165, 0.03 ) ],
                            ['TIC224292441', ufloat( 59.21,  0.42 ), ufloat(  5.76,  0.05 ), ufloat(4610.8 , 7.09), ufloat( 0.074, 0.006) ],
                            ['TIC149399573', ufloat(126.64,  0.76 ), ufloat( 11.06,  0.12 ), ufloat(4690.95,30.7 ), ufloat( 0.15 , 0.028) ],
                            ['TIC27002591 ', ufloat(106.19,  1.66 ), ufloat( 14.46,  0.22 ), ufloat(5063.55,25.68), ufloat(-0.136, 0.021) ],
                            ['TIC458690359', ufloat( 79.72,  5.97 ), ufloat(  8.10,  0.32 ), ufloat(4469.92,37.04), ufloat( 0.287, 0.034) ],
                            ['TIC459769560', ufloat( 70.09,  2.03 ), ufloat(  6.83,  0.16 ), ufloat(4705.28,30.66), ufloat(-0.219, 0.028) ],
                            ['TIC3716092  ', ufloat(168.59,  3.38 ), ufloat( 14.44,  0.76 ), ufloat(5004.67,18.33), ufloat(-0.682, 0.016) ],
                            ['TIC4753562  ', ufloat( 83.95,  2.82 ), ufloat(  8.52,  0.17 ), ufloat(4851.22,22.65), ufloat(-0.341, 0.018) ],
                            ['TIC11938650 ', ufloat(150.55,  3.30 ), ufloat( 17.17,  0.98 ), ufloat(4861.84,17.37), ufloat(-0.063, 0.014) ],
                            ['TIC105489715', ufloat(124.16,  4.56 ), ufloat( 12.76,  0.62 ), ufloat(4809.83,22.91), ufloat( 0.028, 0.019) ],
                            ['TIC302750867', ufloat(156.80,  3.54 ), ufloat( 12.74,  0.55 ), ufloat(4681.75,43.54), ufloat( 0.204, 0.035) ],
                            ['TIC252796304', ufloat(163.62,  4.26 ), ufloat( 13.17,  0.55 ), ufloat(4713.17,46.47), ufloat( 0.032, 0.045) ],
                            ['TIC8901893  ', ufloat(108.66,  3.86 ), ufloat(  8.09,  0.43 ), ufloat(4650.52,28.23), ufloat( 0.209, 0.025) ],
                            ['TIC137649123', ufloat(414.56, 13.19 ), ufloat( 25.15,  0.67 ), ufloat(4761.02,30.34), ufloat( 0.269, 0.027) ]])
        self.id   = [i[0] for i in message]
        self.nu   = [i[1] for i in message]
        self.dnu  = [i[2] for i in message]
        self.teff = [i[3] for i in message]
        self.fe_h = [i[4] for i in message]
        '''
    def new_MS(self):
        # Enter some data, for example a solar twin
        # ufloat holds a measurement and its uncertainty
        nu_max   = ufloat(3090,  3090  * 0.01) # muHz, with 1% uncertainty
        Delta_nu = ufloat(135.1, 135.1 * 0.001) # muHz, with 1% uncertainty
        delta_nu = ufloat(8.957, 8.957 * 0.04) # muHz, with 1% uncertainty
        Teff     = ufloat(5772,  5772  * 0.01) #K,     with 1% uncertainty
        Fe_H     = ufloat(0,             0.1) #dex,   0.1 dex uncertainty
        
        # Take the powers from Table 1, here given with more precision
        # P        =[      alpha,        beta,       gamma,       delta,     epsilon]
        P_age      =[-6.55598425,  9.05883854, -1.29229053, -4.24528340, -0.42594767]
        P_mass     =[ 0.97531880, -1.43472745,  0,           1.21647950,  0.27014278]
        P_radius   =[ 0.30490057, -1.12949647,  0,           0.31236570,  0.10024562]
        P_R_seis   =[ 0.88364851, -1.85899352,  0,           0,           0         ]
        # Apply the scaling relation
        def scaling(nu_max, Delta_nu,delta_nu , Teff, exp_Fe_H, P=P_age,
                     nu_max_Sun = ufloat(3090,  30),   # muHz
                   Delta_nu_Sun = ufloat(135.1, 0.1),  # muHz
                   delta_nu_Sun = ufloat(8.957, 0.059),# muHz
                       Teff_Sun = ufloat(5772,  0.8),  # K 
                       Fe_H_Sun = ufloat(0,     0)):    #dex
                    
            alpha, beta, gamma, delta, epsilon = P
            
            #Equation 5
            return((nu_max    /   nu_max_Sun) ** alpha *
                   (Delta_nu  / Delta_nu_Sun) ** beta  *
                  ((delta_nu  / delta_nu_Sun) ** gamma)/((delta_nu  / delta_nu_Sun) ** gamma) *
                   (Teff      /     Teff_Sun) ** delta *
                   (e**Fe_H   /  e**Fe_H_Sun) ** epsilon)
        self.new_MS_mass   = ['{:.3f}'.format(scaling(self.nu[i], self.dnu[i], 1, self.teff[i], self.fe_h[i], P=P_mass  )) for i in range(len(self.id))]
        self.new_MS_radius = ['{:.3f}'.format(scaling(self.nu[i], self.dnu[i], 1, self.teff[i], self.fe_h[i], P=P_radius)) for i in range(len(self.id))]
        self.new_MS_age    = ['{:.3f}'.format(scaling(self.nu[i], self.dnu[i], 1, self.teff[i], self.fe_h[i], P=P_age   ) * ufloat(4.569, 0.006)) for i in range(len(self.id))]
        return self
    def new_RGB(self):
        refs = [4.3, 1.33, 6.6] # age [Gyr]; mass [solar masses]; radius [solar radii]
        
        # Calibrated exponents from Table 2 文章表二对数据进行了说明
        # P = [   alpha,    beta,  gamma,  delta] 数据对应
        P_age_full = np.array([
              [- 9.760 , 13.08  , -6.931, 0.4894],  #  1
              [- 7.778 , 10.77  , -11.05,      0],  #  2
              [-12.19  , 15.86  ,      0,  1.027],  #  3
              [       0,  1.396 , -22.32, -1.046],  #  4
              [  1.084 ,       0, -23.28, -1.165],  #  5
              [- 8.837 , 11.73  ,      0,      0],  #  6
              [       0,  0.9727, -14.64,      0],  #  7
              [  0.6424,       0, -13.82,      0],  #  8
              [       0,       0,      0,      0],  #  9
              [       0,       0,      0,      0]]) # 10
        sigma_sys_age = np.array([0.25, 0.32, 0.34, 0.82, 0.92, 
                                  0.86, 1.2,  1.3,  0,    0])
        
        # Calibrated exponents from Table 3
        # P = [  alpha,    beta, gamma,   delta]
        P_mass_full = np.array([
              [ 2.901 , -3.876 , 1.621,       0],  #  1
              [ 2.901 , -3.876 , 1.621,       0],  #  2
              [ 3.546 , -4.619 ,     0, -0.1457],  #  3
              [      0, -0.3845, 5.740,  0.4290],  #  4
              [-0.2976,       0, 5.935,  0.4594],  #  5
              [  3.056, -4.015 ,     0,       0],  #  6
              [      0,       0,     0,       0],  #  7
              [      0,       0,     0,       0],  #  8
              [      0,       0,     0,       0],  #  9
              [      0,       0,     0,       0]]) # 10
        sigma_sys_M = np.array([0.023, 0.023, 0.10, 0.11, 0.046, 
                                0.15,  0,     0,    0,    0])
        
        # Calibrated exponents from Table 4
        # P = [  alpha,    beta,  gamma,  delta]
        P_radius_full = np.array([
              [ 0.9570, -1.955 , 0.6288,      0],  #  1
              [ 0.9570, -1.955 , 0.6288,      0],  #  2
              [ 1.008 , -1.999 ,      0,      0],  #  3
              [      0, -0.8048, 2.062 , 0.1378],  #  4
              [-0.6593,       0, 2.953 , 0.2283],  #  5
              [ 1.008 , -1.999 ,      0,      0],  #  6
              [      0, -0.7362, 0.8088,      0],  #  7
              [-0.5591,       0, 0.7857,      0],  #  8
              [      0, -0.7038,      0,      0],  #  9
              [-0.5353,       0,      0,      0]]) # 10
        sigma_sys_R = np.array([0.037, 0.037, 0.075, 0.16, 0.25, 
                                0.075, 0.24,  0.38,  0.24, 0.36])
        
        # now stack all these tables together 
        P         = np.array([P_age_full,    P_mass_full, P_radius_full])
        sigma_sys = np.array([sigma_sys_age, sigma_sys_M, sigma_sys_R])

        # Apply the scaling relation. 
        # Returns age, mass, and radius in Gyr, solar masses, and solar radii. 
        def scaling_giants(nu_max   = ufloat(0,0), 
                           Delta_nu = ufloat(0,0), 
                           Teff     = ufloat(0,0), 
                           Fe_H     = ufloat(0,0), 
                           nu_max_ref   =  104.5, 
                           Delta_nu_ref =    9.25, 
                           Teff_ref     = 4790, 
                           check_bounds=True, 
                           warn_bounds=True, 
                           warn_combo=True, 
                           star_name=''):
            
            result = [np.nan, np.nan, np.nan] # nannannannannan batman! 
            
            # check that the data are in bounds 
            if check_bounds or warn_bounds: 
                if (nu_max   != 0 and nu_max   <   27.9  or nu_max   >   255.6  or
                    Delta_nu != 0 and Delta_nu <    3.73 or Delta_nu >    17.90 or 
                    Teff     != 0 and Teff     < 4520    or Teff     > 5120     or 
                    Fe_H     != 0 and Fe_H     <   -1.55 or Fe_H     >     0.50): 
                    if warn_bounds:
                        warn("Input data out of range of training data. " + star_name)
                    if check_bounds:
                        return result 
            
            # Determine which row of the table to use by checking which entries are 0
            star = np.array([nu_max!=0, Delta_nu!=0, Teff!=0, Fe_H!=0])
            found = False
            for combo_idx in range(len(P_age_full)):
                exponents = P_age_full[combo_idx,]
                if not np.any(exponents): # relation 9 or 10
                    exponents = P_radius_full[combo_idx,]
                found = np.array_equal(np.nonzero(exponents)[0], np.nonzero(star)[0])
                if found: 
                    break 
            
            if not found: # No applicable scaling relation 
                if warn_combo:
                    warn("No applicable relation for input combination. " + star_name)
                return result 
            
            # Equation 1, plus the systematic error of the corresponding relation
            zi = []
            for var_idx, exponents in enumerate(P):
                if not np.any(exponents[combo_idx,]):
                    continue 
                alpha, beta, gamma, delta = exponents[combo_idx,]
                sigma = sigma_sys[var_idx, combo_idx]
                result[var_idx] = (
                    (nu_max   /   nu_max_ref) ** alpha  * 
                    (Delta_nu / Delta_nu_ref) ** beta   * 
                    (Teff     /     Teff_ref) ** gamma  * refs[var_idx] * 
                    (np.e**Fe_H             ) ** delta) + ufloat(0, sigma)
                zi.append([alpha, beta, gamma, delta])
            print(zi)
            return result
        
        def scaling_printer(age, mass, radius):
            if not isnan(age):
                print('Age:', '{:.2u}'.format(age), '[Gyr]')
        
            if not isnan(mass):
                print('Mass:', '{:.2u}'.format(mass), '[solar masses]')
        
            if not isnan(radius):
                print('Radius:', '{:.2u}'.format(radius), '[solar radii]')
        all_m = [scaling_giants(self.nu[i], self.dnu[i], self.teff[i], self.fe_h[i]) for i in range(len(self.id))]          
        self.new_RGB_age    = ['{:.3f}'.format(i[0]) for i in all_m] 
        self.new_RGB_mass   = ['{:.3f}'.format(i[1]) for i in all_m]
        self.new_RGB_radius = ['{:.3f}'.format(i[2]) for i in all_m]
        return self
    def new_fdnu(self):
        evstate = [1 for i in range(len(self.id))]
        s = asfgrid.Seism()
        logz = [i.n+np.log10(0.019) for i in self.fe_h] 
        mass, radius = s.get_mass_radius(evstate,logz, [i.n for i in self.teff], [i.n for i in self.dnu], [i.n for i in self.nu])
        logg = s.mr2logg(mass,radius)
        dnu, nu, fdnu = s.get_dnu_numax(evstate, logz, [i.n for i in self.teff], mass,mass,logg)
        
        logg_solar = 4.43796037457 # cgs unit
        teff_solar=ufloat(5777.0,0.8)        # kelvin
        numax_solar=ufloat(3090.0,30) # micro Hz 3090+-30
        dnu_solar=ufloat(135.1,0.1)    # micro Hz 135.1
        z_solar=0.019
        Mass = []
        Radius = []
        for i in range(len(self.id)):
            M = (self.nu[i]/numax_solar)**3*(self.dnu[i]/(fdnu[i]*dnu_solar))**(-4)*(self.teff[i]/teff_solar)**(3/2)
            R = (self.nu[i]/numax_solar)*(self.dnu[i]/(fdnu[i]*dnu_solar))**(-2)*(self.teff[i]/teff_solar)**(1/2)
            Mass.append('{:.3f}'.format(M))
            Radius.append('{:.3f}'.format(R))
        self.fdnu_mass = Mass
        self.fdnu_radius = Radius
        self.fdnu = fdnu
        return self
    def classical(self):
        for i in range(len(self.id)):
            classical_mass   = [seis.estimate_mass(  numax=self.nu[i].n, deltanu=self.dnu[i].n, teff=self.teff[i].n, numax_err=self.nu[i].s, deltanu_err=self.dnu[i].s, teff_err=self.teff[i].s) for i in range(len(self.id))] 
            classical_radius = [seis.estimate_radius(numax=self.nu[i].n, deltanu=self.dnu[i].n, teff=self.teff[i].n, numax_err=self.nu[i].s, deltanu_err=self.dnu[i].s, teff_err=self.teff[i].s) for i in range(len(self.id))]
            classical_logg   = [seis.estimate_logg(  numax=self.nu[i].n,                        teff=self.teff[i].n, numax_err=self.nu[i].s,                            teff_err=self.teff[i].s) for i in range(len(self.id))]
            self.classical_mass   = ['{:.3f}'.format(ufloat(i.value, i.error.value)) for i in classical_mass  ]
            self.classical_radius = ['{:.3f}'.format(ufloat(i.value, i.error.value)) for i in classical_radius]
            self.classical_logg   = ['{:.3f}'.format(ufloat(i.value, i.error.value)) for i in classical_logg  ]
        return self

