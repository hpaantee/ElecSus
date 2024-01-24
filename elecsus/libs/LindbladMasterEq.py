import arc
import copy
import logging
import numpy as np
import os
import sys
from datetime import datetime
from scipy import constants as c
import scipy as sp
import sympy as sy
import symengine as se
from sympy.physics.wigner import wigner_3j, wigner_6j
import tqdm
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import EigenSystem as ES

from sympy import *
from sympy.printing.latex import latex
init_printing(use_unicode=True) # allow LaTeX printing


log = logging.getLogger('LME')
log.setLevel(logging.DEBUG)

class state:
    def __init__(self, n, l, j, f=None):
        self.n = n
        self.l = l
        self.j = j
        self.f = f

    def __str__(self):
        symbols = ['S', 'P', 'D', 'F']
        return f'{self.n}{symbols[self.l]}{self.j}'

    def __call__(self, precision):
        if precision == 'nlj':
            return (self.n, self.l, self.j)
        elif precision == 'nljf':
            return self.n, self.l, self.j, self.f

    def F(self, f):
        self.f = f
        return self


class beam:
    def __init__(self, **kwargs):
        self.w = kwargs['w']
        self.profile = kwargs['profile']
        if 'sgn' in kwargs:
            self.sgn = kwargs['sgn']
        else:
            self.sgn = 1
        # beam diameter, either as diameter or area
        if 'D' in kwargs:
            self.D = kwargs['D']
            self.A = c.pi * kwargs['D']**2 / 4
        else:
            self.A = kwargs['A']
            self.D = 2 * np.sqrt(kwargs['A'] / c.pi)
        # beam power, either as power or electric field
        if 'E' in kwargs:
            self.setE(kwargs['E'])
        else:
            self.setP(kwargs['P'], kwargs['profile'])

    def setP(self, P, profile):
        self.P = P
        # I = 1/2 * c * epsilon_0 * E0**2
        log.info(f'Beam profile used: {profile}')
        if profile == 'flat':
            I = P / self.A
        elif profile == 'gaussian':
            I = 2 * P / self.A
        else:
            raise KeyError('no beam profile specified')
        self.E = np.sqrt(2 * I / c.c / c.epsilon_0)


    def setE(self, E):
        self.E = E
        self.P = self.A / 4 * c.c * c.epsilon_0 * E**2


    def __iter__(self):
        return iter((self.w, self.P, self.D, self.sgn))


p_dict_defaults = {	'lcell':75e-3,'Bfield':0., 'T':20.,
                    'GammaBuf':0., 'shift':0.,
                    # Polarisation of light
                    'theta0':0., 'Pol':50.,
                    # B-field angle w.r.t. light k-vector
                    'Btheta':0, 'Bphi':0,
                    'Constrain':True, 'DoppTemp':20.,
                    'rb85frac':72.17, 'K40frac':0.01, 'K41frac':6.73,
                    # Beyond weak fields
                    'laserPower': 1e-15, 'laserWaist': 5e-3, 'bwf_precision': 'high',
                    'BoltzmannFactor':True}


class atomicSystem:
    # def __init__(self, element, states, T=20+273.15, beam_diameter=5e-3):
    def __init__(self, element, states, p_dict={}):
        # Insert all default values we did not explicitly specify
        p_dict = {**p_dict_defaults, **p_dict}
        self.isotopeShift = 0
        self.element = element
        if element.lower() in ['li6', 'lithium6']:
            self.atom = arc.Lithium6()
        elif element.lower() in ['li7', 'lithium7']:
            self.atom = arc.Lithium7()
        elif element.lower() in ['na', 'na23', 'sodium', 'sodium23']:
            self.atom = arc.Sodium()
        elif element.lower() in ['k39', 'potassium39']:
            self.atom = arc.Potassium39()
            self.atom.abundance = 1 - (p_dict['K40frac'] + p_dict['K41frac']) / 100
            self.isotopeShift = 15.87
        elif element.lower() in ['k40', 'potassium40']:
            self.atom = arc.Potassium40()
            self.atom.abundance = p_dict['K40frac'] / 100
            self.isotopeShift = -109.773
        elif element.lower() in ['k41', 'potassium41']:
            self.atom = arc.Potassium41()
            self.atom.abundance = p_dict['K41frac'] / 100
            self.isotopeShift = -219.625
        elif element.lower() in ['rb85', 'rubidium85']:
            self.atom = arc.Rubidium85()
            self.atom.abundance = p_dict['rb85frac'] / 100
            self.isotopeShift = 21.734
        elif element.lower() in ['rb87', 'rubidium87']:
            self.atom = arc.Rubidium87()
            self.atom.abundance = 1 - p_dict['rb85frac'] / 100
            self.isotopeShift = -56.361
        elif element.lower() in ['cs', 'cs133', 'caesium', 'caesium133']:
            self.atom = arc.Caesium()
        else:
            raise ValueError


        self.states = copy.deepcopy(states)
        self.max_allowed_states = 3
        self.n_states = len(states)
        if self.n_states > self.max_allowed_states:
            raise RuntimeError('Exceeded allowed number of states!')

        self.T = p_dict['T'] + 273.15
        if p_dict['Constrain'] == True:
            self.DoppT = self.T
        else:
            self.DoppT = p_dict['DoppTemp'] + 273.15
        self.beam_diameter = p_dict['laserWaist']
        self.p_dict = p_dict

        log.debug('Init system properties')
        self.initSystemProperties()
        log.debug('Generate symbols')
        self.generateSymbols()
        log.debug('Genrate matrices')
        self.generateMatrices()
        # Add constrain that total population has to be 1
        self.system_matrix = self.master_equation.as_mutable()
        if 'symbolic_transit' in self.p_dict:
            log.warning('USING symbolic transit time')
            self.system_matrix = self.system_matrix.subs({'tau_t': self.transit_time})
        self.system_matrix[0] = -1 + self.r.trace()
        log.debug('Generate linear system')
        self.A, self.b = self.generate_linear_system()
        # self.atom.conn.close()

    def update_transit(self, mean_speed):
        self.transit_time = self.getTransitTime(mean_speed) * 1e6
        print(f'Updated transit time: {self.transit_time}')
        # self.generateMatrices()
        self.system_matrix = self.master_equation.as_mutable()
        if 'symbolic_transit' in self.p_dict:
            self.system_matrix = self.system_matrix.subs({'tau_t': self.transit_time})
        self.system_matrix[0] = -1 + self.r.trace()
        self.A, self.b = self.generate_linear_system()

    def update(self, p_dict):
        self.p_dict = p_dict
        self.T = p_dict['T'] + 273.15
        if p_dict['Constrain'] == True:
            self.DoppT = self.T
        else:
            self.DoppT = p_dict['DoppTemp'] + 273.15
        self.beam_diameter = p_dict['laserWaist']
        self.transit_time = self.getTransitTime()

    def getTransitTime(self, mean_speed_2d=None):
        # Ref: ARC-Alkali-Rydberg-Calculator Web interface (click 'View code')
        # in s
        if mean_speed_2d is None:
            # mean_speed = self.atom.getAverageSpeed(self.DoppT) # this is 3D!
            mean_speed_2d = np.sqrt(np.pi * c.k * self.DoppT / 2 / self.atom.mass)
        # beam_fwhm = self.beam_diameter / 2 / 0.8493218
        # tau = np.sqrt(c.pi) * beam_fwhm / (2 * mean_speed
        # Sagle
        mean_path = np.pi / 4 * self.beam_diameter
        # G = np.sqrt(8*c.k*self.DoppT/c.pi / self.atom.mass) / self.beam_diameter
        # tau = self.beam_diameter / np.abs(mean_speed)
        tau = mean_path / np.abs(mean_speed_2d)
        # print(1/G)
        # sys.exit()
        # return 1 / G
        # print(G)
        # sys.exit()
        return tau


    def initSystemProperties(self):
        self.f_resonance = self.atom.getTransitionFrequency(
            *self.states[0]('nlj'), *self.states[1]('nlj'))

        self.f = [self.atom.breitRabi(*state('nlj'), np.array([self.p_dict['Bfield']]))[1] for state in self.states]
        self.mf = [self.atom.breitRabi(*state('nlj'), np.array([self.p_dict['Bfield']]))[2] for state in self.states]

        self.n = np.array([len(mf) for mf in self.mf])
        self.total_levels = sum([len(mf) for mf in self.mf])

        self.transit_time = self.getTransitTime() * 1e6
        if 'Gammat' in self.p_dict:
            self.transit_time = self.p_dict['Gammat'] * 1e6

        self.slices = [slice(self.n[0:i].sum(), self.n[0:i+1].sum()) for i in range(self.n_states)]
        DME = np.sqrt((2*self.states[0].j+1) / (2*self.states[1].j+1)) \
            * self.atom.getReducedMatrixElementJ_asymmetric(*self.states[0]('nlj'), *self.states[1]('nlj')) \
            * c.e * c.physical_constants['Bohr radius'][0]
        self.naturalLineWidth = [self.atom.getTransitionRate(
            *self.states[i+1]('nlj'), *self.states[i]('nlj')) / 2 / c.pi * 1e-6
            for i in range(self.n_states-1)]
        # SFF = [self.getSFF(self.states[i].F(self.sublevels[i]),
        #                    self.states[i+1].F(self.sublevels[i+1]))
        #        for i in range(self.n_states-1)]
        # self.dipole_moments = [np.sqrt(SFF[i]/3) * DME[i]
        #                        for i in range(self.n_states-1)]
        H = ES.Hamiltonian(self.element, self.p_dict['Dline'], self.atom.gL, self.p_dict['Bfield'])
        Mg = np.array(H.groundManifold)[:,1:]  # Cut off energy eigenvalues
        Me = np.array(H.excitedManifold)[:,1:]

        self.energySeparation = [None] * self.n_states
        self.energySeparation[0] = H.groundEnergies
        if self.p_dict['Dline'] == 'D1':
            DlineIndexOffset = 0
            self.energySeparation[1] = H.excitedEnergies[0:self.n[0]]
        elif self.p_dict['Dline'] == 'D2':
            DlineIndexOffset = self.n[0]
            self.energySeparation[1] = H.excitedEnergies[self.n[0]:]

        self.eigv = np.diag(np.ones(self.total_levels))

        dme_r = np.matmul(Mg, Me[DlineIndexOffset:DlineIndexOffset+self.n[1], 0*self.n[0]:self.n[0]].T).real
        dme_z = np.matmul(Mg, Me[DlineIndexOffset:DlineIndexOffset+self.n[1], self.n[0]:2*self.n[0]].T).real
        dme_l = np.matmul(Mg, Me[DlineIndexOffset:DlineIndexOffset+self.n[1], 2*self.n[0]:3*self.n[0]].T).real
        dme_squared = np.power(dme_r, 2) + np.power(dme_z, 2) + np.power(dme_l, 2)
        self.Gammas = dme_squared * self.naturalLineWidth[0]

        if self.p_dict['Pol'] == 0: # RCP
            self.dme = dme_r * DME
        elif self.p_dict['Pol'] == 50: #LP
            self.dme = 1 / np.sqrt(2) * (dme_r + dme_l) * DME
        elif self.p_dict['Pol'] == 100: # LCP
            self.dme = dme_l * DME

    def generateSymbols(self):
        #######################################################################
        # Generate symbols and variables
        #######################################################################
        self.wL = sy.symbols(f'w_01')
        self.Efield = sy.symbols(f'Efield_01')
        self.r_individual = sy.symbols(
            f'\\rho_{{(0:{self.total_levels})/(0:{self.total_levels})}}')
        self.r = sy.Matrix(self.total_levels,
                           self.total_levels, self.r_individual)
        if 'symbolic_transit' in self.p_dict:
            self.tau_t = sy.symbols('tau_t')

    def generateMatrices(self):
        #######################################################################
        # Generate matrices
        #######################################################################
        self.H_rabi = sy.zeros(self.total_levels, self.total_levels)
        self.H_rabi[self.slices[0], self.slices[1]] = 0.5e-6 / c.h * self.dme * self.Efield
        self.H_rabi = self.H_rabi + self.H_rabi.transpose()

        detunings = np.concatenate([-self.energySeparation[0], self.wL - self.energySeparation[1]])
        self.H_energylevels = sy.diag(*detunings)
        self.H = self.H_rabi + self.H_energylevels

        # Make Lindblad
        def Lindblad_decay(rates):
            L = sy.zeros(self.total_levels, self.total_levels)
            for i in range(self.total_levels):
                for j in range(self.total_levels):
                    c = sy.Matrix(np.outer(self.eigv[i], self.eigv[j]).T)
                    # L += rates[i,j] * (c@self.r@c.T - 0.5 * (c.T@c@self.r + self.r@c.T@c))
                    # L += (rates[j, i] * self.r[j, j] - rates[i, j] * self.r[i, i]) * sy.Matrix(np.outer(self.eigv[i], self.eigv[i]))
                    L[i,i] += (rates[j, i] * self.r[j, j] - rates[i, j] * self.r[i, i])
                    if (i != j):
                        for k in range(self.total_levels):
                            # L -= 0.5 * (rates[i, k] + rates[j, k]) * self.r[i, j] * sy.Matrix(np.outer(self.eigv[i], self.eigv[j]))
                            L[i,j] -= 0.5 * (rates[i, k] + rates[j, k]) * self.r[i, j]
            return L

        def Lindblad_dephase(rates):
            L = sy.zeros(self.total_levels, self.total_levels)
            for i in range(self.total_levels):
                for j in range(self.total_levels):
                    if (i != j):
                        L[i, j] -= 0.5 * rates[i, j] * self.r[i, j]
            return L

        # defining the correct Lindblad operators for transit-time and collisional broadening
        g_dec = np.zeros((self.total_levels, self.total_levels))
        g_col = np.zeros((self.total_levels, self.total_levels))
        g_transit = np.zeros((self.total_levels, self.total_levels))
        # Putting the decay in this part of the block matrix, we get decay.
        # If we would put it into the other one, we would get spontaneous excitation
        g_dec[self.slices[1], self.slices[0]] = self.Gammas.T

        # Mixing between ground states due to transit time
        # If we set diagonal to zero (no ground_0 to ground_0 state), we need to use self.n[0] - 1
        if 'transit_factor' not in self.p_dict:
            transit_factor = 1
        else:
            transit_factor = self.p_dict['transit_factor']


        g_transit[self.slices[0], self.slices[0]] = 1 / self.transit_time / self.n[0] / transit_factor
        # "Decay" of excited states to ground states due to transit time
        g_transit[self.slices[1], self.slices[0]] = 1 / self.transit_time / self.n[0] / transit_factor
        if 'symbolic_transit' in self.p_dict:
            g_transit = sy.zeros(self.total_levels, self.total_levels)
            g_transit[self.slices[0], self.slices[0]] = 1 / self.tau_t / self.n[0] / transit_factor * sy.ones(self.n[0], self.n[0])
            # "Decay" of excited states to ground states due to transit time
            g_transit[self.slices[1], self.slices[0]] = 1 / self.tau_t / self.n[0] / transit_factor * sy.ones(self.n[1], self.n[0])

        if 'collisions' not in self.p_dict:
            log.warning('Implicitly assume decaying collisions')
            self.p_dict['collisions'] = 'decay'
        if self.p_dict['collisions'] == 'decay':
            log.info('decaying collisions!')
            # g_col[self.slices[0], self.slices[0]] = self.p_dict['GammaBuf'] / self.n[0]
            g_col[self.slices[1], self.slices[0]] = self.p_dict['GammaBuf'] / self.n[0]
            L_dec = Lindblad_decay(g_dec + g_transit + g_col)
            self.master_equation = -sy.I * (self.H*self.r - self.r*self.H) - L_dec
        elif self.p_dict['collisions'] == 'dephase':
            log.info('dephasing collisions!')
            # g_col[self.slices[0], self.slices[0]] = self.p_dict['GammaBuf']
            g_col[self.slices[1], self.slices[0]] = self.p_dict['GammaBuf']
            g_col[self.slices[0], self.slices[1]] = self.p_dict['GammaBuf']
            L_dec = Lindblad_decay(g_dec + g_transit)
            L_deph = Lindblad_dephase(g_col)
            self.master_equation = -sy.I * (self.H*self.r - self.r*self.H) - L_dec - L_deph


    def generate_linear_system(self):
        self.r_list = self.matrix2list(self.r)
        test = np.array(self.r_list)
        # Create list of off-diagonal elements relevant for i->j transition
        self.transition_list = []
        for i in range(self.n_states-1):
            mask = np.full((self.total_levels, self.total_levels), False)
            mask[self.slices[i], self.slices[i+1]] = True
            self.transition_list.append(self.matrix2list(mask))
        self.transition_list2 = []
        for i in range(self.n_states-1):
            mask = np.full((self.total_levels, self.total_levels), False)
            mask[self.slices[i+1], self.slices[i]] = True
            self.transition_list2.append(self.matrix2list(mask))

        # m = np.array(self.transition_list2).squeeze()
        # print(test[m])
        # print(m.shape)
        # sys.exit()
        A, b = sy.linear_eq_to_matrix(self.system_matrix, self.r_list)
        # A = A.simplify(rational=None)
        A = se.Matrix(A)
        A = se.Lambdify([self.wL, self.Efield], A, real=False, cse=True)
        # b is always just an vector with zeros and the first entry being one
        b = np.zeros((self.total_levels**2, 1))
        b[0] = 1
        return A, b

    def v_dist(self, v):
        return np.sqrt(self.atom.mass / (2 * c.pi * c.k * self.DoppT)) \
            * np.exp(-self.atom.mass * v**2 / (2 * c.k * self.DoppT))

    def cdf(self, v):
        o = np.sqrt(c.k * self.DoppT / self.atom.mass) * np.sqrt(2)
        return 0.5 * (1 + sp.special.erf(v/o))

    def cdfinv(self, p):
        o = np.sqrt(c.k * self.DoppT / self.atom.mass) * np.sqrt(2)
        return o * sp.special.erfinv(2 * p - 1)

    def matrix2list(self, mat):
        # Generate list containing all entries of density matrix
        # First diagonal entries: r[0,0], r[1,1], ...
        # Then upper half: r[0,1], r[0,2], ...
        # Then lower half: r[1,0], r[2,0], ...
        l1 = []
        l2 = []
        for i in range(mat.shape[0]):
            for j in range(i+1, mat.shape[1]):
                l1.append(mat[i, j])
                l2.append(mat[j, i])
        return list(mat.diagonal()) + l1 + l2

    def solve(self, beams, v=None, precision='high'):
        if precision == 'high':
            #######################################################################
            # Calculate Rabi Frequencies
            #######################################################################
            f_list, _, _, sgn_list = zip(*beams)
            E_list = [np.atleast_1d(beam.E) for beam in beams]
            w_ge = np.atleast_1d(f_list[0])
            sgn_ge = sgn_list[0]
            wavenumber_ge = self.f_resonance / c.c

            #######################################################################
            # Solve linear system
            #######################################################################
            log.debug('Solve linear system')
            res = np.array([[np.linalg.solve(self.A(w, E), self.b) for E in E_list[0]] for w in w_ge])

            #######################################################################
            # Extract relevant information
            #######################################################################
            # Move density matrix dimension to the front
            res = np.moveaxis(res.squeeze(), -1, 0)
            # If we only want to calculate a single value, res[list] * k
            # would multiply wrong dimensions, e.g. (8,) * (8,1) -> (8,8)
            # So we add a dimension so that (8,1) * (8,1) -> (8,1)
            if res.ndim == 1:
                res = np.expand_dims(res, axis=1)
            k_alt = [np.divide.outer(self.dme.ravel(), E_list[i]) / c.epsilon_0 for i in range(self.n_states-1)]
            # - Return sum of excited states. Indexing order is given by order
            #   of arguments of 'sy.linear_eq_to_matrix' above
            # - Population of excited states is given by diagonal entries
            # - Complex-valued susceptibility is given by off-axis entries
            #   (only one side, they are complex conjugated anyway)
            state_population = np.array([np.sum(res[self.slices[i]], axis=0).real for i in range(self.n_states)])
            chi = np.array([self.atom.abundance * 2 * np.sum(res[self.transition_list[i]] * k_alt[i], axis=0)
                for i in range(self.n_states-1)])
            # chi = np.array([self.atom.abundance * (
            #     np.sum(res[self.transition_list[i]] * k_alt[i], axis=0)
            #     -np.sum(res[self.transition_list2[i]] * k_alt[i], axis=0))
            #     for i in range(self.n_states-1)])
            return state_population[1].squeeze(), chi[0].squeeze()
        else:
            return self.solve_via_LUT(beams[0])

    def solve_w_doppler(self, beams, precision='high'):
        log.debug('enter __solve_w_doppler__')
        beam_ge = beams[0]
        # chi_dopp(∆) = \int p(v,T)chi(∆-kv)dv = (p*chi)(∆)
        # k = 1 / lam2bda = w/c
        w_ge, P_ge, D_ge, sgn_ge = beam_ge
        w_ge = np.atleast_1d(w_ge)
        P_ge = np.atleast_1d(P_ge)
        k_ge = self.f_resonance / c.c / 1e6

        exci_state = np.ones((len(w_ge), len(P_ge)), dtype='float64')
        chi = np.zeros_like(exci_state, dtype='complex128')

        resolution = 2  # MHz
        v = np.arange(w_ge.min()/k_ge - 1000, w_ge.max()/k_ge + 1000, resolution)
        v_distribution = self.v_dist(np.subtract.outer(w_ge / k_ge, v))
        for i, P in enumerate(P_ge):
                # Use symmetry of convolution to calculate population_number
                # once and instead calculate Maxwell Boltzmann distribution
                # more often (which is computationally cheaper)
                E, C = self.solve([beam(w=k_ge * v, P=P, D=D_ge, sgn=1, profile=beam_ge.profile)], precision=precision)
                exci_state[:, i] = np.sum(v_distribution * E, axis=1) * resolution
                chi[:, i] = np.sum(v_distribution * C, axis=1) * resolution
        return exci_state.squeeze(), chi.squeeze()

    def get_dynamic_xaxis(self):
        bounds = [5000, 18000]
        low_resolution = 20  # MHz
        high_resolution = 2  # MHz

        # Initial scan of optical depth to obtain threshold for regions of interest
        x = np.arange(*bounds, high_resolution)
        y = self.optical_depth([beam(w=x, P=1e-15, D=1)], doppler=False)
        limit = 1e-4 * y.min()

        # Find bounds of those regions of interest
        idx_high = np.argwhere(y>limit).flatten()
        idx_high_l = idx_high[np.where(abs(np.diff(idx_high))>1)[0] + 0]
        idx_high_r = idx_high[np.where(abs(np.diff(idx_high))>1)[0] + 1]

        # Recreate the x axis, now with different resolutions depending on the frequency
        x_low = np.arange(*bounds, low_resolution)
        x_high = [np.arange(*x[np.array(i)], high_resolution) for i in zip(idx_high_l, idx_high_r)]
        x_new = np.sort(np.unique(np.concatenate([x_low, *x_high], axis=None)))
        return x_new

    def LUT(self):
        """Generate/use lookup table to speed up subsequent solving of Maxwell-Bloch equations

        Light field is defined by detuning, total power and beam diameter. To reduce the parameter space,
        we calculate the LUT based on the electric field. Later when doing a lookup, we need to "correct" the power,
        so that we use the equivalent field strength as when we calculated the lookup table.
        Note that currently the beam diameter used to define the transit time broadening of the atomic system
        is fixed!

        Parameters
        ----------
        beam_ge : beam
            light field which interacts with our atomic system

        Returns
        -------
        tuple
            tuple of zero (later will be excited state fraction) and complex-valued susceptibility chi
        """
        basedir = os.path.dirname(os.path.realpath(__file__))
        LUT_file = f'{basedir}/LUT/LUT_{self.atom.elementName}_{self.states[1]}.npz'
        if os.path.exists(LUT_file):
            with np.load(LUT_file) as f:
                detunings = f['detunings']
                powers = f['powers']
                beam_diameter = f['beam_diameter']
                chi = f['chi']
        else:
            os.makedirs(f'{basedir}/LUT', exist_ok=True)
            print(f'Generating LUT file {LUT_file}!')
            detunings = self.get_dynamic_xaxis()
            powers = np.geomspace(1e-9, 1e-1, 1000, endpoint=True)
            beam_diameter = 1e-3
            chi = np.empty((detunings.size, powers.size), dtype=np.complex128)
            for i, p in enumerate(tqdm.tqdm(powers)):
                _, chi[:, i] = self.solve([beam(w=detunings, P=p, D=beam_diameter)], precision='high')
            np.savez_compressed(LUT_file, detunings=detunings, powers=powers, beam_diameter=beam_diameter, chi=chi)
        return detunings, powers, beam_diameter, chi

    def transmission(self, beams, z=50e-3, doppler=True, precision='high'):
        log.debug('__enter transmission__')
        alpha = self.optical_depth(beams, doppler, precision)
        return np.exp(alpha * z)

    def absorption(self, beams, z=50e-3, doppler=True, precision='high'):
        return 1 - self.transmission(beams, z, doppler, precision)

    def optical_depth(self, beams, doppler=True, precision='high'):
        log.debug('__enter optical_depth__')
        n = self.atom.getNumberDensity(self.T)
        if doppler:
            _, chi = self.solve_w_doppler(beams, precision=precision)
        else:
            _, chi = self.solve(beams, precision=precision)
        n_imag = np.sqrt(1.0 + chi * n).imag
        return 4 * c.pi * self.f_resonance / c.c * n_imag

    def propagated_transmission(self, beams, z=50e-3, doppler=True, steps=50):
        w, P0, D, _ = beams[0]
        dz = z / steps
        P = np.zeros((steps+1, len(w)))
        T = np.ones((steps+1, len(w)))
        P[0] = P0

        for i in range(1, steps+1):
            T[i] = self.transmission([beam(w=w, P=P[i-1].min(), D=D, profile=beams[0].profile)], z=dz, doppler=doppler)
            P[i] = T[i] * P[i-1]
        T = np.product(T, axis=0)
        return T
        # detunings, powers, beam_diameter, chi = self.LUT()
        # chi_real = RectBivariateSpline(detunings, powers, chi.real, kx=1, ky=1)
        # chi_imag = RectBivariateSpline(detunings, powers, chi.imag, kx=1, ky=1)

        # P0 = P * (beam_diameter / D)#**2
        # n = self.atom.getNumberDensity(self.T)
        # k = self.f_resonance / c.c
        # resolution = 2  # MHz
        # v = np.arange(w.min()/1e6 - 1000, w.max()/1e6 + 1000, resolution)
        # v_distribution = self.v_dist(np.subtract.outer(w / k, v))
        # # Initialize some variables
        # P = np.zeros((steps+1, len(beams[0].w)))
        # T = np.ones((steps+1, len(beams[0].w)))
        # P[0] = P0
        # abs_pref = dz * 4 * c.pi * self.f_resonance / c.c

        # for i in range(1, steps+1):
        #     # RectBivariateSpline wants strictly increasing values for x and y.
        #     # The detuning fulfills this naturally, but power not.
        #     # So we sort it and later "unsort" it again
        #     if doppler:
        #         sequence = np.argsort(P[i-1])
        #         chi_t = chi_real(k*v, P[i-1][sequence], grid=True) + 1j * chi_imag(k*v, P[i-1][sequence], grid=True)
        #         chi = np.sum(v_distribution * chi_t.T[sequence.argsort()], axis=1) * resolution
        #     else:
        #         chi = chi_real(w, P[i-1], grid=False) + 1j * chi_imag(w, P[i-1], grid=False)
        #     T[i] = np.exp(abs_pref * np.sqrt(1. + chi * n).imag)
        #     P[i] = T[i] * P[i-1]
        # T = np.product(T, axis=0)
        # return T


if __name__ == '__main__':
    groundState = state(5, 0, 1/2)     # 5S1/2
    excitedState = state(5, 1, 1/2)    # 5P1/2
#     rydbergState = state(17, 0, 1/2)  # 17S1/2
    p_dict = {'T': 20., 'laserWaist': 5e-3, 'Bfield': 1000, 'rb85frac': 0}
    rb85 = atomicSystem(
        'Rb87',
        [groundState,
        excitedState],
        p_dict)
    x = np.linspace(4000e6, 6000e6, 800)
    od = rb85.transmission([beam(w=x, P=1e-15, D=5e-3)], z=2e-3, doppler=False)
    plt.figure()
    plt.plot(x, od)
    plt.show()
