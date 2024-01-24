"""
Series of tests and example code to run elecsus via the API

Last updated 2018-07-04 MAZ
"""
# py 2.7 compatibility
from __future__ import (division, print_function, absolute_import)

import matplotlib.pyplot as plt
import numpy as np
import time
import os
import seaborn as sns
import scipy as sp
from scipy import constants as c
results_folder = 'LME_plots'
os.makedirs(results_folder, exist_ok=True)
sns.set_palette(sns.color_palette('mako'))

import sys
sys.path.append('../')
import elecsus_methods as EM
sys.path.append('../libs')
from spectra import get_spectra
import LindbladMasterEq as LME
import BasisChanger as bc



groundState = LME.state(5, 0, 1/2)	 # 5S1/2
excitedState_D1 = LME.state(5, 1, 1/2)	# 5P1/2
excitedState_D2 = LME.state(5, 1, 3/2)	# 5P3/2

E_LCP = bc.lrz_to_xyz(np.array([1,0,0]))
E_RCP = bc.lrz_to_xyz(np.array([0,1,0]))
E_LP = np.array([1,1,0])

###############################################################################
# Rb85 D1
###############################################################################
########## LCP ##########
def Rb85_D1_LCP_B_0G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 0, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb85 D1 LCP B=0G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb85_D1_LCP_B_0G.png', dpi=200)

def Rb85_D1_LCP_B_100G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 100, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	fig = plt.figure(tight_layout=True)
	plt.title('Rb85 D1 LCP B=100G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb85_D1_LCP_B_100G.png', dpi=200)

def Rb85_D1_LCP_B_1000G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 1000, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb85 D1 LCP B=1000G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb85_D1_LCP_B_1000G.png', dpi=200)

########## RCP ##########
def Rb85_D1_RCP_B_0G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 0,
	   'Bfield': 0, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb85 D1 RCP B=0G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb85_D1_RCP_B_0G.png', dpi=200)

def Rb85_D1_RCP_B_100G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 0,
	   'Bfield': 100, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb85 D1 RCP B=100G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb85_D1_RCP_B_100G.png', dpi=200)

def Rb85_D1_RCP_B_1000G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 0,
	   'Bfield': 1000, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb85 D1 RCP B=1000G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb85_D1_RCP_B_1000G.png', dpi=200)
###############################################################################
# Rb85 D2
###############################################################################
########## LCP ##########
def Rb85_D2_LCP_B_0G():
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 0, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb85 D2 LCP B=0G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb85_D2_LCP_B_0G.png', dpi=200)

def Rb85_D2_LCP_B_100G():
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 100, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb85 D2 LCP B=100G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb85_D2_LCP_B_100G.png', dpi=200)

########## RCP ##########
def Rb85_D2_RCP_B_100G():
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 20., 'Pol': 0,
	   'Bfield': 100, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb85 D2 RCP B=100G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb85_D2_RCP_B_100G.png', dpi=200)

###############################################################################
# Rb85 D1
###############################################################################
########## LCP ##########
def Rb87_D1_LCP_B_0G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 0, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(3500, 5000, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 LCP B=0G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_LCP_B_0G.png', dpi=200)

def Rb87_D1_LCP_B_100G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 100, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	# x = np.linspace(3600, 5000, 500)
	x = np.linspace(4500, 4750, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 LCP B=100G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_LCP_B_100G.png', dpi=200)

def Rb87_D1_LCP_B_1000G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 1000, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(4250, 6000, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 LCP B=1000G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_LCP_B_1000G.png', dpi=200)

def Rb87_D1_LCP_B_6000G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 6000, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(5500, 9000, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 LCP B=6000G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_LCP_B_6000G.png', dpi=200)

########## RCP ##########
def Rb87_D1_RCP_B_0G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 0,
	   'Bfield': 0, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(3500, 5000, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 RCP B=0G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_RCP_B_0G.png', dpi=200)

def Rb87_D1_RCP_B_100G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 0,
	   'Bfield': 100, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(3500, 5000, 2000)
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 RCP B=100G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_RCP_B_100G.png', dpi=200)

def Rb87_D1_RCP_B_1000G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 0,
	   'Bfield': 1000, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(3500, 6000, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 RCP B=1000G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_RCP_B_1000G.png', dpi=200)

def Rb87_D1_RCP_B_6000G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 0,
	   'Bfield': 6000, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(8000, 15000, 10000)
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 RCP B=6000G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_RCP_B_6000G.png', dpi=200)

########## LP ##########
def Rb87_D1_LP_B_0G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 50,
	   'Bfield': 0, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(3500, 5000, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 LP B=0G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_LP_B_0G.png', dpi=200)

def Rb87_D1_LP_B_100G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 50,
	   'Bfield': 100, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(3500, 5000, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 LP B=100G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_LP_B_100G.png', dpi=200)

def Rb87_D2_LCP_B_0G():
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 50., 'Pol': 100,
	   'Bfield': 0, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(3900, 4400, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D2 LCP B=0G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D2_LCP_B_0G.png', dpi=200)

def Rb87_D2_LCP_B_100G():
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 100, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(3500, 4500, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D2 LCP B=100G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D2_LCP_B_100G.png', dpi=200)


def Rb87_D1_LCP_B_100G_power_scan():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 10., 'Pol': 100,
	   'Bfield': 100, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.1499}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(4500, 4550, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 LCP B=100G')
	for p in [1e-10, 1e-8, 1e-7]:
		p_dict_bwf['laserPower'] = p
		[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
		plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_LCP_B_100G_power_scan.png', dpi=200)

def Rb87_D1_LCP_B_100G_high_T():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 100, 'Bfield': 100, 'rb85frac': 0}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 5e-3}
	x = np.linspace(3000, 6000, 1000)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 LCP B=100G high T')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_LCP_B_100G_high_T.png', dpi=200)

def Rb87_D2_RCP_B_6000G_high_T():
	from datetime import datetime
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 50., 'Pol': 0, 'Bfield': 6000, 'rb85frac': 0, 'GammaBuf': 0}#, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 2e-3, 'collisions': 'decay'}#, 'symbolic_transit': True}
	x = np.linspace(4000, 20000, 2000)
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])
	t = datetime.now()
	[y_bwf] = get_spectra(x, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	# y_elecsus = np.abs(np.log(y_elecsus) / 2e-3)
	# y_bwf = np.abs(np.log(y_bwf) / 2e-3)
	# print(f'Fractional difference: {y_elecsus.max()/y_bwf.max()}')
	# y_elecsus /= y_elecsus.max()
	# y_bwf /= y_bwf.max()
	print(f'Total time of call: {datetime.now()-t}')
	# print(get_spectra(np.array([6000, 6001]), E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0']))

	fig = plt.figure(tight_layout=True)
	# size = fig.get_size_inches()
	# fig.set_size_inches(size[0], 2)
	plt.title('Rb87 D2 RCP B=6000G high T')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	# plt.plot(x, y_bwf-y_elecsus, c='C1')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	# plt.ylabel('Residual')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D2_RCP_B_6000G_high_T.png', dpi=200)

def Rb87_D2_RCP_B_6000G_high_T_custom_transit():
	from datetime import datetime
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 50., 'Pol': 0, 'Bfield': 6000, 'rb85frac': 0, 'GammaBuf': 0}#, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 2e-3, 'collisions': 'decay', 'symbolic_transit': True}

	def maxwell_boltzmann(v):
		# https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution
		return 4 * np.pi * np.sqrt(Rb87_D2.atom.mass / (2 * c.pi * c.k * Rb87_D2.DoppT))**3 \
		* np.exp(-Rb87_D2.atom.mass * v**2 / (2 * c.k * Rb87_D2.DoppT)) * v**2

	def rayleigh(v):
		# https://en.wikipedia.org/wiki/Rayleigh_distribution
		return 2 * np.pi * Rb87_D2.atom.mass / (2 * c.pi * c.k * Rb87_D2.DoppT) \
		* np.exp(-Rb87_D2.atom.mass * v**2 / (2 * c.k * Rb87_D2.DoppT)) * v

	x = np.linspace(9500, 10900, 600)
	t = datetime.now()
	Rb87_D2 = LME.atomicSystem('Rb87', [groundState, excitedState_D2], p_dict=p_dict_bwf)
	beam = LME.beam(w=x, P=1e-15, D=2e-3, profile='flat')
	doppler = True

	import scipy as sp
	from scipy import constants as c
	RbDen = Rb87_D2.atom.getNumberDensity(Rb87_D2.T)
	k = Rb87_D2.f_resonance / c.c #/ 1e6

	v = np.linspace(0, 1000, 100000)
	v_dist_rayleigh = rayleigh(v)
	v_dist_maxwell_boltzmann = maxwell_boltzmann(v)
	v_avg_rayleigh = (v*v_dist_rayleigh).sum() / v_dist_rayleigh.sum()
	v_avg_maxwell_boltzmann = (v*v_dist_maxwell_boltzmann).sum() / v_dist_maxwell_boltzmann.sum()
	print(f'Average in 2D: {v_avg_rayleigh:.1f}')
	print(f'Average in 3D: {v_avg_maxwell_boltzmann:.1f}')
	plt.figure()
	plt.plot(v, v_dist_rayleigh, c='C1', label='Rayleigh distribution')
	plt.plot(v, v_dist_maxwell_boltzmann, '--', c='C4', label='MB distribution')
	plt.legend()
	plt.xlabel('Velocity [m/s]')
	plt.ylabel('P(v)')
	plt.savefig(f'{results_folder}/compare_velocity_distributions.png', dpi=200)

	Rb87_D2.update_transit(v_avg_rayleigh)
	y_avg_rayleigh = Rb87_D2.optical_depth([beam], doppler=doppler)
	y_rayleigh = np.zeros_like(y_avg_rayleigh, dtype=np.complex128)

	v = np.linspace(0, 900, 75)
	dv = v[1] - v[0]
	for i, vi in enumerate(v):
		Rb87_D2.update_transit(vi)
		if not doppler:
			_, chi = Rb87_D2.solve([beam])
		else:
			_, chi = Rb87_D2.solve_w_doppler([beam])
		y_rayleigh += chi * rayleigh(vi) * dv
	y_rayleigh = 4 * np.pi * k * np.sqrt(1.0 + y_rayleigh * RbDen).imag
	print(f'Total time of call: {datetime.now()-t}')

	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, tight_layout=True, gridspec_kw={'height_ratios': [3, 1]})
	ax1.plot(x, y_avg_rayleigh, c='C1', label='Rayleigh, avg')
	ax1.plot(x, y_rayleigh, '--', c='C4', label='Rayleigh, int')
	ax2.plot(x, (y_avg_rayleigh - y_rayleigh)/y_avg_rayleigh*100, c='C1')
	plt.xlabel('Detuning [MHz]')
	ax1.set_ylabel('Optical depth')
	ax2.set_ylabel('Residual [%]')
	ax1.legend(frameon=False)
	plt.savefig(f'{results_folder}/Compare_velocity_distributions_doppler_1e-15_power.png', dpi=200)

def Rb87_D2_RCP_B_6000G_high_T_custom_transit_check_velocity_classes():
	from datetime import datetime
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 50., 'Pol': 0, 'Bfield': 6000, 'rb85frac': 0, 'GammaBuf': 0}#, 'DoppTemp': -273.14999}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 2e-3, 'collisions': 'decay', 'symbolic_transit': True}

	def rayleigh(v):
		# https://en.wikipedia.org/wiki/Rayleigh_distribution
		return 2 * np.pi * Rb87_D2.atom.mass / (2 * c.pi * c.k * Rb87_D2.DoppT) \
		* np.exp(-Rb87_D2.atom.mass * v**2 / (2 * c.k * Rb87_D2.DoppT)) * v

	x = np.linspace(10206.6, 10207, 2)
	t = datetime.now()
	Rb87_D2 = LME.atomicSystem('Rb87', [groundState, excitedState_D2], p_dict=p_dict_bwf)
	beam = LME.beam(w=x, P=1e-1, D=2e-3, profile='flat')
	doppler = True

	RbDen = Rb87_D2.atom.getNumberDensity(Rb87_D2.T)
	k = Rb87_D2.f_resonance / c.c #/ 1e6

	v = np.linspace(0, 1000, 100000)
	v_dist_rayleigh = rayleigh(v) * (v[1] - v[0])
	v_dist_rayleigh /= v_dist_rayleigh.sum()
	v_avg_rayleigh = (v*v_dist_rayleigh).sum() / v_dist_rayleigh.sum()
	print(f'Average in 2D: {v_avg_rayleigh:.1f}')

	Rb87_D2.update_transit(v_avg_rayleigh)
	y_avg_rayleigh = Rb87_D2.optical_depth([beam], doppler=doppler)

	N = np.arange(3, 41, dtype=int)
	y_rayleigh = np.zeros(N.size, dtype=np.complex128)

	for ni, n in enumerate(N):
		v = np.linspace(0, 900, n)
		dv = v[1] - v[0]
		tmp = 0
		for i, vi in enumerate(v):
			Rb87_D2.update_transit(vi)
			if not doppler:
				_, chi = Rb87_D2.solve([beam])
			else:
				_, chi = Rb87_D2.solve_w_doppler([beam])
			tmp += chi[0] * rayleigh(vi) * dv
		y_rayleigh[ni] = 4 * np.pi * k * np.sqrt(1.0 + tmp * RbDen).imag
		print(f'Total time of call: {datetime.now()-t}')

	plt.figure(tight_layout=True)
	plt.axhline(y_avg_rayleigh[0], c='k')
	plt.plot(N, y_rayleigh, c='C1', label='Rayleigh, int')
	plt.xlabel('Number of velocity samples')
	plt.ylabel('Line-centre optical depth')
	plt.savefig(f'{results_folder}/Convergence_number_velocity_classes.png', dpi=200)

def Rb87_D2_RCP_B_6000G_high_T_custom_doppler():
	from datetime import datetime
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 50., 'Pol': 0, 'Bfield': 6000, 'rb85frac': 0, 'GammaBuf': 0}#, 'DoppTemp': -273.14999, 'Constrain': False}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 2e-3, 'collisions': 'decay'}
	x = np.linspace(8000, 9000, 30)
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])

	t = datetime.now()
	groundState = LME.state(5, 0, 1/2)     # 5S1/2
	excitedState = LME.state(5, 1, 3/2)    # 5P3/2
	rb87 = LME.atomicSystem('Rb87', [groundState, excitedState], p_dict=p_dict)
	print(f'Generating atom: {datetime.now()-t}')
	t = datetime.now()

	import scipy.constants as c
	binsize = 1000
	epsilon = 1e-6
	p_bounds = np.linspace(epsilon, 1-epsilon, binsize + 1)
	p = (p_bounds[1:] + p_bounds[:-1]) / 2
	v_bounds = rb87.cdfinv(p_bounds)
	v = rb87.cdfinv(p)
	dv = np.diff(v_bounds)

	chi = np.zeros((len(v), len(x)), dtype=np.complex128)
	k = rb87.f_resonance[0] / c.c #/ 1e6
	n = rb87.atom.getNumberDensity(rb87.T)

	for i, vi in enumerate(v):
		print(i)
		# rb87.update_transit(vi)
		chi[i] = rb87.solve([LME.beam(w=x-k*vi/1e6, P=1e-15, D=5e-3, profile='flat')])[1] * rb87.v_dist(vi) * dv[i]
	chi_doppler = np.sum(chi, axis=0, dtype=np.complex128)
	n_imag = np.sqrt(1.0 + chi_doppler * n).imag
	od =  -4 * c.pi * rb87.f_resonance[0] / c.c * n_imag
	y_bwf = np.exp(-od * 2e-3)

	# [y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_test, outputs=['S0'])
	# y_bwf = rb87.transmission([LME.beam(w=x, P=1e-15, D=5e0, profile='flat')], z=2e-3, doppler=False)
	# print(f'Total time of call: {datetime.now()-t}')
	# [y_bwf] = get_spectra(x, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	y_elecsus = np.abs(np.log(y_elecsus) / 2e-3)
	# y_bwf = np.abs(np.log(y_bwf) / 2e-3)
	# print(f'Fractional difference: {y_elecsus.max()/y_bwf.max()}')
	# y_elecsus /= y_elecsus.max()
	# y_bwf /= y_bwf.max()
	# print(f'Total time of call: {datetime.now()-t}')
	# print(get_spectra(np.array([6000, 6001]), E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0']))

	fig = plt.figure(tight_layout=True)
	# size = fig.get_size_inches()
	# fig.set_size_inches(size[0], 2)
	plt.title('Rb87 D2 RCP B=6000G high T')
	# plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, od, '--', c='r')
	# plt.plot(x, test2, '--', c='C4')
	plt.plot(x, y_elecsus, '--', c='k', label='ElecSus')
	# plt.plot(x, y_bwf-y_elecsus, '--', c='r')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	# plt.ylabel('Residual')
	# plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D2_RCP_B_6000G_high_T_custom_Doppler.png', dpi=200)

def Rb87_D2_RCP_B_6000G_high_T_CG():
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 50., 'Pol': 0, 'Bfield': 6000, 'rb85frac': 0, 'GammaBuf': 0}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 2e-3}#, 'Constrain': False, 'DoppTemp': -273.14999}
	x = np.linspace(4000, 20000, 2000)
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])
	# y_elecsus = -np.log(y_elecsus) / 2e-3
	# [y_elecsus_d] = get_spectra(x, E_in=E_RCP, p_dict={**p_dict, 'GammaBuf':0}, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	# y_bwf = -np.log(y_bwf) / 2e-3
	# y_bwf = np.exp(np.log(y_bwf))
	# print(get_spectra(np.array([6000, 6001]), E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0']))
	plt.figure()
	plt.title('Rb87 D2 RCP B=6000G high T')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	# plt.plot(x, y_elecsus_d, '-', c='C4', label='ElecSus no buffergas')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D2_RCP_B_6000G_high_T_CG.png', dpi=200)

def Rb87_D2_RCP_B_6000G_high_T_sweep_Gamma_t():
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 60., 'Pol': 0, 'Bfield': 6000, 'rb85frac': 0, 'GammaBuf': 600}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 2e-3}
	x = np.linspace(4000, 20000, 2000)
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])
	ys = []
	G = [0e-6, 1e-6, 6e-6, 30e-6, 100e-6]
	for g in G:
		[y_bwf] = get_spectra(x, E_in=E_RCP, p_dict={**p_dict_bwf, 'Gammat': g}, outputs=['S0'])
		ys.append(y_bwf)
	plt.figure()
	plt.title('Rb87 D2 RCP B=6000G high T')
	for i in range(len(ys)):
		plt.plot(x, ys[i], c=f'C{i}', label=f'LME: {G[i]*1e6}')
	plt.plot(x, y_elecsus, '--', c='k', label='ElecSus')
	# plt.plot(x, y_elecsus_d, '-', c='C4', label='ElecSus no buffergas')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D2_RCP_B_6000G_high_T_sweep_Gamma_t.png', dpi=200)

def Rb87_D2_RCP_B_6000G_high_T_single_datapoints():
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 60., 'Pol': 0, 'Bfield': 6000, 'rb85frac': 0}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 2e-3}
	x = np.linspace(4000, 20000, 2000)
	# x_single = np.atleast_1d([6087, 6088])
	# x_single = np.array([17112, 17112.05])
	[y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	print(y_bwf[(5500<x) & (x<6500)].min())
	print(y_bwf[(16500<x) & (x<18000)].min())
	# [y_bwfs] = get_spectra(x_single, E_in=E_RCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 RCP B=6000G high T single datapoints')
	plt.plot(x, y_bwf, c='C1', label='LME')
	# plt.plot(x_single, y_bwfs,'x', c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D2_RCP_B_6000G_high_T_single_datapoints.png', dpi=200)

def Rb87_D2_RCP_B_6000G_Lorentzian_test():
	from datetime import datetime
	# p_test = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 20., 'Pol': 0, 'Bfield': 6000, 'rb85frac': 0,
	# 		'Constrain': False, 'DoppTemp': -273.149999}
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 20., 'Pol': 0, 'Bfield': 6000, 'rb85frac': 0, 'GammaBuf': 0,
			'Constrain': False, 'DoppTemp': -273.14999, 'laserPower': 1e-15, 'laserWaist': 2e-3, 'collisions': 'decay'}
	x = np.linspace(7050, 7400, 2000)
	def Lorentzian(x, x0, a, gamma, b):
		return 1 - a * gamma / ((x-x0)**2 + gamma**2/4) + b
	groundState = LME.state(5, 0, 1/2)     # 5S1/2
	excitedState = LME.state(5, 1, 3/2)    # 5P3/2
	rb87 = LME.atomicSystem('Rb87', [groundState, excitedState], p_dict=p_dict)
	# [y_elecsus] = get_spectra(x, E_in=E_RCP, p_dict=p_test, outputs=['S0'])
	bwf_nat = rb87.optical_depth([LME.beam(w=x, P=1e-15, D=5e0, profile='flat')], doppler=False)
	from scipy.optimize import curve_fit
	popt_nat, _ = curve_fit(Lorentzian, x, bwf_nat, p0=(7200, 0.6, 6, 0))
	print(popt_nat)

	fig = plt.figure(tight_layout=True)
	plt.title('Rb87 D2 RCP B=6000G high T')
	plt.plot(x, bwf_nat, c='C1', label='LME')
	plt.plot(x, Lorentzian(x, *popt_nat), '--', c='C4', label='Lorentzian fit nat')
	# plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	# plt.ylabel('Residual')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D2_RCP_B_6000G_Lorentzian_test.png', dpi=200)


def thesis_plots_lifetime_broadening():
	import matplotlib as mpl
	# p_dict_elecsus = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 20., 'Pol': 0, 'Bfield': 0, 'GammaBuf': 0, 'Constrain': False}
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 20., 'Pol': 0, 'Bfield': 0}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-15, 'laserWaist': 2e-3}
	# p_dict_elecsus = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 60., 'Pol': 0, 'Bfield': 6000, 'rb85frac': 0, 'GammaBuf': 0, 'Constrain': False}
	x = np.linspace(1600, 1800, 2000)
	# x = np.linspace(4000, 20000, 20000)
	size = list(set_size(width, fraction=1))
	size[0] = 0.5 * size[0]
	size[1] = 3
	fig, ax = plt.subplots(tight_layout=True, figsize=size)
	groundState = LME.state(5, 0, 1/2)     # 5S1/2
	excitedState = LME.state(5, 1, 3/2)    # 5P1/2
	rb85 = LME.atomicSystem('Rb85', [groundState, excitedState], p_dict=p_dict_bwf)
	od = rb85.transmission([LME.beam(w=x, P=1e-15, D=5e-3, profile='flat')], z=2e-3, doppler=False)
	plt.plot(x, od, c='C1')
	# [y] = get_spectra(x, E_in=E_RCP, p_dict={**p_dict_elecsus, 'DoppTemp': T-273.15}, outputs=['S0'])
	# plt.plot(x, y, c=cmap(norm(T)))
	# ax.set_title('Doppler broadening')
	ax.set_xlabel('Detuning [MHz]')
	ax.set_ylabel('Transmission')
	# fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='Temperature [K]')
	# plt.savefig(f'{results_folder}/thesis_plots_broadening_doppler.png', dpi=200)
	plt.savefig(f'{results_folder}/thesis_plots_broadening_natural.pdf')

def thesis_plots_doppler_broadening():
	import matplotlib as mpl
	p_dict_elecsus = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 20., 'Pol': 0, 'Bfield': 0, 'GammaBuf': 0, 'Constrain': False}
	# p_dict_elecsus = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 60., 'Pol': 0, 'Bfield': 6000, 'rb85frac': 0, 'GammaBuf': 0, 'Constrain': False}
	x = np.linspace(1200, 2200, 2000)
	# x = np.linspace(4000, 20000, 20000)
	size = list(set_size(width, fraction=1))
	# size[0] = 0.5 * size[0]
	size[1] = 2.5
	fig, ax = plt.subplots(tight_layout=True, figsize=size)
	temps = np.geomspace(0.1, 300, 10, endpoint=True)
	cmap = plt.get_cmap('flare_r')
	norm = mpl.colors.LogNorm(0.1, temps.max())
	for i, T in enumerate(temps):
		[y] = get_spectra(x, E_in=E_RCP, p_dict={**p_dict_elecsus, 'DoppTemp': T-273.15}, outputs=['S0'])
		plt.plot(x, y, c=cmap(norm(T)))
	# ax.set_title('Doppler broadening')
	ax.set_xlabel('Detuning [MHz]')
	ax.set_ylabel('Transmission')
	fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='Temperature [K]')
	plt.savefig(f'{results_folder}/thesis_plots_broadening_doppler.pdf')


def set_size(width, fraction=1):
    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


# plt.rcdefaults()
# plt.style.use("phdthesis.mplstyle")
# width = 418.25


if __name__ == '__main__':

	np.set_printoptions(linewidth=300)
	# Rb85_D1_LCP_B_0G()
	# Rb85_D1_LCP_B_100G()
	# Rb85_D1_LCP_B_1000G()
	# Rb85_D1_RCP_B_0G()
	# Rb85_D1_RCP_B_100G()
	# Rb85_D1_RCP_B_1000G()
	# Rb85_D2_LCP_B_0G()
	# Rb85_D2_LCP_B_100G()
	# Rb85_D2_RCP_B_100G()

	# Rb87_D1_LCP_B_0G()
	# Rb87_D1_LCP_B_100G()
	# Rb87_D1_LCP_B_1000G()
	# Rb87_D1_LCP_B_6000G()
	# Rb87_D1_RCP_B_0G()
	# Rb87_D1_RCP_B_100G()
	# Rb87_D1_RCP_B_1000G()
	# Rb87_D1_RCP_B_6000G()
	# Rb87_D1_LP_B_0G()
	# Rb87_D1_LP_B_100G()
	# Rb87_D2_LCP_B_0G()
	# Rb87_D2_LCP_B_100G()
	# Rb87_D1_LCP_B_100G_power_scan()
	# Rb87_D1_LCP_B_100G_high_T()

	# Rb87_D2_RCP_B_6000G_high_T()
	# Rb87_D2_RCP_B_6000G_high_T_custom_doppler()
	# Rb87_D2_RCP_B_6000G_high_T_custom_transit()
	Rb87_D2_RCP_B_6000G_high_T_custom_transit_check_velocity_classes()

	# Rb87_D2_RCP_B_6000G_high_T_CG()
	# Rb87_D2_RCP_B_6000G_high_T_sweep_Gamma_t()
	# Rb87_D2_RCP_B_6000G_high_T_single_datapoints ()
	# Rb87_D2_RCP_B_6000G_Lorentzian_test()
	# thesis_plots_lifetime_broadening()
	# thesis_plots_doppler_broadening()
	plt.show()

	"""
	Things needed to get fixed
	--------------------------
	- D1 isotope shift of Rb87
	- Linear polarization is wrong
	- Steck notes of section 4.3.1 regarding the dipole moment/linear light/weak fields
	"""
