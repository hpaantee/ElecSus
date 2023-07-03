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
	   'Bfield': 0, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 200)
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
	   'Bfield': 100, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 200)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.figure()
	plt.title('Rb85 D1 LCP B=100G')
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb85_D1_LCP_B_100G.png', dpi=200)

def Rb85_D1_LCP_B_1000G():
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 1000, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 200)
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
	   'Bfield': 0, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 200)
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
	   'Bfield': 100, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 200)
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
	   'Bfield': 1000, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 200)
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
	   'Bfield': 0, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 100)
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
	   'Bfield': 100, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
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
	   'Bfield': 100, 'rb85frac': 100, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(1400, 2100, 100)
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
	   'Bfield': 0, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
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
	   'Bfield': 100, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	# x = np.linspace(3600, 5000, 500)
	x = np.linspace(4500, 4750, 400)
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
	   'Bfield': 1000, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(4250, 6000, 500)
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
	   'Bfield': 6000, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(5500, 9000, 500)
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
	   'Bfield': 0, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(3500, 5000, 500)
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
	   'Bfield': 100, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(3500, 5000, 500)
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
	   'Bfield': 1000, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(3500, 6000, 500)
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
	   'Bfield': 6000, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(8000, 15000, 5000)
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
	   'Bfield': 0, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(3500, 5000, 500)
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
	   'Bfield': 100, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(3500, 5000, 500)
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

def Rb87_D2_LCP_B_100G():
	p_dict = {'Elem':'Rb','Dline':'D2', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 100, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(3800, 4000, 500)
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
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 100, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': -273.149}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(4500, 4550, 500)
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
	p_dict = {'Elem':'Rb','Dline':'D1', 'lcell':2e-3, 'T': 20., 'Pol': 100,
	   'Bfield': 100, 'rb85frac': 0, 'Constrain': False, 'DoppTemp': 20}
	p_dict_bwf = {**p_dict, 'laserPower': 1e-12, 'laserWaist': 5e-3}
	x = np.linspace(3000, 6000, 500)
	[y_elecsus] = get_spectra(x, E_in=E_LCP, p_dict=p_dict, outputs=['S0'])
	plt.figure()
	plt.title('Rb87 D1 LCP B=100G')
	[y_bwf] = get_spectra(x, E_in=E_LCP, p_dict=p_dict_bwf, outputs=['S0'])
	plt.plot(x, y_bwf, c='C1', label='LME')
	plt.plot(x, y_elecsus, '--', c='C4', label='ElecSus')
	plt.xlabel('Detuning [MHz]')
	plt.ylabel('Transmission')
	plt.legend(frameon=False)
	plt.savefig(f'{results_folder}/Rb87_D1_LCP_B_100G_power_scan.png', dpi=200)

if __name__ == '__main__':
	# Rb85_D1_LCP_B_0G()
	# Rb85_D1_LCP_B_100G()
	# # Rb85_D1_LCP_B_1000G()
	# # Rb85_D1_RCP_B_0G()
	# # Rb85_D1_RCP_B_100G()
	# # Rb85_D1_RCP_B_1000G()
	# # Rb85_D2_LCP_B_0G()
	# Rb85_D2_LCP_B_100G()
	# # Rb85_D2_RCP_B_100G()

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
	# Rb87_D2_LCP_B_100G()
	# Rb87_D1_LCP_B_100G_power_scan()
	Rb87_D1_LCP_B_100G_high_T()
	plt.show()