# Calls meteo preproc to compute some profiles

from Pre_Proc.prosat.meteo_saturne import Fic_meteo_saturne

import numpy as np

def compute_meteo_profile(lmo, z0, uref, zref, t0, z):
    """
    Monin-Obhukov similarity profiles
    Based on what is done in code_saturne
    Implementation modified from Pre_Proc.prosat.meteo_saturne.prof_saturne_mo2
      Args:
          lmo: float, Monin-obukhov length
          z0: float, ground rugosity
          uref: float, reference speed at height zref
          zref: float, Reference height (for uref)
          t0: float, temperature at 2m above ground, in kelvin
          z: Array-like, height at which the profile will be computed 
    """
    kappa = 0.42
    g = 9.81
    cp=1005.
    rair=287.
    tkelvin=273.15
    pref=1e5
    lmoneutre=1e50
    rscp=rair/cp
    pmer=101325.
    sigmae=1.3
    sigmak=1.
    ce1=1.44
    ce2=1.92
    Cmu = 0.03 #constante C_mu du modèle k-epsilon
    

    ##############################
    ##############################
    ### Fonctions universelles ###
    ##############################
    ##############################

    ################
    ###  Neutre  ###
    ################

    #Formes dérivées
    def phim_n(z,l):
      return np.ones_like(z)

    def phih_n(z,l):
      return np.zeros_like(z)

    #Formes intégrées
    def psim_n(z,l):
      return np.log(z/z0)

    def psih_n(z,l):
      return np.zeros_like(z)

    ##Hogstrom 1988

    #Formes dérivées

    def phim_u(z,l):
      a=1.
      b=19.3
      e=-1./4.
      return a*(1.-b*z/l)**e

    def phih_u(z,l):
      a=0.95
      b=11.6
      e=-1./2.
      return a*(1.-b*z/l)**e

    #Formes intégrées
    def psim_u(z,l):
      a=1.
      b=19.3
      e=1./4.
      x0=z0/l
      x=z/l
      psi0=(1.-b*x0)**e
      psi=(1.-b*x)**e
      return a*(np.log(z/z0)
               -2*np.log((1.+psi)/(1.+psi0))
               -np.log((1.+psi**2)/(1.+psi0**2))
               +2*(np.arctan(psi)-np.arctan(psi0)))

    def psih_u(z,l):
      a=0.95
      b=11.6
      e=1./2.
      x0=z0/l
      x=z/l
      psi0=(1.-b*x0)**e
      psi=(1.-b*x)**e
      return a*(np.log(z/z0)
                -2*np.log((1+psi)/(1+psi0)))

    ##############
    ### Stable ###
    ##############

    ##Cheng and Brutsaert 2005

    #Formes dérivées
    def phim_s(z,l):
      a=6.1
      b=2.5
      x=z/l
      return 1.+a*(x+(x**b)*((1.+x**b)**((1.-b)/b)))/(x+(1.+x**b)**(1./b))

    def phih_s(z,l):
      a=5.3
      b=1.1
      x=z/l
      return 1.+a*(x+(x**b)*((1.+x**b)**((1.-b)/b)))/(x+(1.+x**b)**(1./b))

    #Formes intégrées
    def psim_s(z,l):
      a=6.1
      b=2.5
      x=z/l
      x0=z0/l
      return np.log(z/z0)+a*(np.log(x+(1.+x**b)**(1./b))-np.log(x0+(1.+x0**b)**(1./b)))

    def psih_s(z,l):
      a=5.3
      b=1.1
      x=z/l
      x0=z0/l
      return np.log(z/z0)+a*(np.log(x+(1.+x**b)**(1./b))-np.log(x0+(1.+x0**b)**(1./b)))

    #Forme dérivées ++
    def ff(x):
      b=2.5
      return x+(x**b)*((1.+x**b)**((1.-b)/b))

    def gg(x):
      b=2.5
      return x+(1.+x**b)**(1./b)

    def dz_ff(x):
      b=2.5
      return (1/x) * (ff(x) + (b-1.)*(x**b)*((1+(x**b))**(1./b-2)))

    def dz_gg(x):
      return ff(x)/x

    def d2z_ff(x):
      b=2.5
      return (b-1.)*(x**(b-2.))*((1+x**b)**(1./b-3.))*(b+(1.-b)*x**b)

    def d2z_gg(x):
      return 1./x*(dz_ff(x)-dz_gg(x))

    def dz_phim(z,l):
      a=6.1
      x=z/l
      return a/l/gg(x)*(dz_ff(x)-ff(x)*dz_gg(x)/gg(x))

    def d2z_phim(z,l):
      a=6.1
      x=z/l
      return a/(l**2.)/gg(x)*(d2z_ff(x)-ff(x)*d2z_gg(x)/gg(x)-2.*dz_gg(x)*l*dz_phim(z,l)/a)

    ##################################
    ##################################
    ### Fonctions universelles fin ###
    ##################################
    ##################################

    #Choix des fonctions universelles
    if lmo!=None:
      #Neutre
      if abs(lmo)>abs(lmoneutre):
        phim=phim_n
        phih=phih_n
        psim=psim_n
        psih=psih_n
      #Stable
      elif lmo>0.:
        phim=phim_s
        phih=phih_s
        psim=psim_s
        psih=psih_s
      #Instable
      else:
        phim=phim_u
        phih=phih_u
        psim=psim_u
        psih=psih_u

    #Calcul de ustar
    ustar0=uref*kappa/psim(zref+z0,lmo)

    #Calcul thetastar et flux de chaleur
    theta0=t0*(pref/pmer)**rscp
    tstar=(ustar0**2)*theta0/(kappa*g*lmo)

    #Profil de vitesse
    uh=ustar0/kappa*psim(z+z0,lmo)

    #Profil de temperature potentielle
    theta=theta0+tstar/kappa*psih(z+z0,lmo)

    #Profil de Rif (Richardson de flux)
    Rif=(z+z0)/lmo/phim(z+z0,lmo)

    #Profil tke
    ect=(ustar0**2)/np.sqrt(Cmu)*np.sqrt(1.-np.minimum(Rif,1.))

    #Profil epsilon
    eps=(ustar0**3)/kappa/(z+z0)*phim(z+z0,lmo)*(1.-Rif)

    #stack des profils
    profiles = np.stack([uh, theta, ect, eps], axis = 1)

    return profiles