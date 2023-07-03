import arc
import copy
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
            self.setP(kwargs['P'])

    def setP(self, P):
        self.P = P
        # I = 1/2 * c * epsilon_0 * E0**2
        I = 2 * P / self.A
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

        self.initSystemProperties()
        self.generateSymbols()
        self.generateMatrices()
        # Add constrain that total population has to be 1
        self.system_matrix = self.master_equation.as_mutable()
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
        # self.n_mf1 = 2 * self.sublevels[0][0] + 1
        # self.n_mf2 = 2 * self.sublevels[0][1] + 1
        # self.G_01_val = self.n_mf2 / (self.n_mf1 + self.n_mf2) / self.transit_time
        # self.G_10_val = self.n_mf1 / (self.n_mf1 + self.n_mf2) / self.transit_time

    # def getSFF(self, state1, state2):
    #     # Ref: Steck, Daniel A. "Rubidium 85 D Line Data" (2009)
    #     f1 = np.atleast_1d(state1.f)
    #     f2 = np.atleast_1d(state2.f)
    #     SFF = np.zeros((f1.size, f2.size))
    #     for i, fi in enumerate(f1):
    #         for j, fj in enumerate(f2):
    #             SFF[i, j] = (2 * fj + 1) * (2 * state1.j + 1) \
    #                 * wigner_6j(state1.j, state2.j, 1, fj, fi, self.atom.I)**2
    #     return SFF

    def getF(self, state):
        # Number of Hyperfine states F:
        # |J-I| <= F <= J+I
        return np.arange(np.abs(self.atom.I - state.j),
                         self.atom.I + state.j + 1).astype(np.int32)

    def getBranchingRatio(self, state1, state2):
        # Ref: Wenting, Chen "Two-Photon spectroscopy of rubidium in the
        # vicinity of silicon devices" (2019)

        # State is assumed to be the lower energy state
        if self.atom.getEnergy(*state1('nlj')) > self.atom.getEnergy(*state2('nlj')):  # noqa
            state1, state2 = state2, state1
        f1 = np.atleast_1d(state1.f)
        f2 = np.atleast_1d(state2.f)
        B = np.zeros((f1.size, f2.size))
        for i, fi in enumerate(f1):
            for j, fj in enumerate(f2):
                B[i, j] = (2 * state2.j + 1) * (2 * state1.f + 1) \
                    * wigner_6j(state2.j, fj, self.atom.I, fi, state1.j, 1)**2
        return B

    def getTransitTime(self):
        # Ref: ARC-Alkali-Rydberg-Calculator Web interface (click 'View code')
        # in s
        mean_speed = self.atom.getAverageSpeed(self.DoppT)
        beam_fwhm = self.beam_diameter / 2 / 0.8493218
        tau = np.sqrt(c.pi) * beam_fwhm / (2 * mean_speed)
        return tau

    def initSystemProperties(self):
        self.f_resonance = np.array([self.atom.getTransitionFrequency(
            *self.states[i]('nlj'), *self.states[i+1]('nlj'))
            for i in range(self.n_states-1)])

        self.sublevels = [self.getF(state) for state in self.states]
        self.f = [self.atom.breitRabi(*state('nlj'), np.array([self.p_dict['Bfield']]))[1] for state in self.states]
        self.mf = [self.atom.breitRabi(*state('nlj'), np.array([self.p_dict['Bfield']]))[2] for state in self.states]

        self.n = np.array([len(mf) for mf in self.mf])
        self.total_levels = sum([len(mf) for mf in self.mf])

        self.transit_time = self.getTransitTime() * 1e6

        self.energySeparation = [self.atom.breitRabi(*state('nlj'), np.array([self.p_dict['Bfield']/1e4]))[0][0] * 1e-6
                                 for state in self.states]
        self.energySeparation[0] += self.isotopeShift
        self.slices = [slice(self.n[0:i].sum(), self.n[0:i+1].sum()) for i in range(self.n_states)]
        # dipole matrix moment: |<J||er||J'>|
        # SFF'(F->F') = (2F'+1)(2J+1){J, J', 1, F', F, I}^2
        # d^2 = 1/3 * SFF' * |<J||er||J'>|^2
        # Ω = d * E / hbar = sqrt(SFF/3) * |<J||er||J'>| * E / hbar
        # Ref: Steck, Daniel A. "Rubidium 85 D Line Data" (2009), p. 10
        DME = [self.atom.getReducedMatrixElementJ_asymmetric(
            *self.states[i]('nlj'), *self.states[i+1]('nlj'))
            * c.e * c.physical_constants['Bohr radius'][0]
            for i in range(self.n_states-1)]
        # SFF = [self.getSFF(self.states[i].F(self.sublevels[i]),
        #                    self.states[i+1].F(self.sublevels[i+1]))
        #        for i in range(self.n_states-1)]
        # self.dipole_moments = [np.sqrt(SFF[i]/3) * DME[i]
        #                        for i in range(self.n_states-1)]

        self.dme = [np.zeros((self.n[i], self.n[i+1])) for i in range(self.n_states-1)]
        self.Gammas = [np.zeros((self.n[i], self.n[i+1])) for i in range(self.n_states-1)]
        H = ES.Hamiltonian(self.element, self.p_dict['Dline'], self.atom.gL, self.p_dict['Bfield'])
        Eg = H.groundEnergies * 1e6
        Ee = H.excitedEnergies * 1e6
        Mg = np.array(H.groundManifold)[:,1:]  # Cut off energy eigenvalues
        Me = np.array(H.excitedManifold)[:,1:]
        # self.energySeparation = [Eg, Ee]

        if self.p_dict['Dline'] =='D1':
            interatorList = list(range(self.n[0]))
        elif self.p_dict['Dline']=='D2':
            interatorList = list(range(self.n[0],self.n[1]))

        if self.p_dict['Pol'] == 0:
            bottom = 0
            top = self.n[0]
        elif self.p_dict['Pol'] == 50:
            bottom = self.n[0]
            top = 2*self.n[0]
        elif self.p_dict['Pol'] == 100:
            bottom = 2*self.n[0]
            top = Me.shape[0]

        for i in range(self.n_states-1):
            for m in range(self.n[i]):
                for n in range(self.n[i+1]):
                    if self.p_dict['Dline'] == 'D1':
                        k = n
                    elif self.p_dict['Dline'] == 'D2':
                        k = n + self.n[0]
                    cleb = np.dot(Mg[m], Me[k][bottom:top]).real
                    if np.abs(cleb) > 0.0223:  # see spectra.py: cleb2 > 0.0005
                        self.dme[i][m, n] = cleb * DME[i]

                    b = self.atom.getBranchingRatio(
                        self.states[i].j, self.f[i][m], self.mf[i][m],
                        self.states[i+1].j, self.f[i+1][n], self.mf[i+1][n]
                        )
                    self.Gammas[i][m, n] = b * self.atom.getTransitionRate(
                        *self.states[i+1]('nlj'), *self.states[i]('nlj')) / 2 / c.pi * 1e-6

    def generateSymbols(self):
        #######################################################################
        # Generate symbols and variables
        #######################################################################
        # Symbols for both laser frequencies
        self.wL = np.array([sy.symbols(f'w_{i}{i+1}') for i in range(self.n_states-1)])
        self.Efield = np.array([sy.symbols(f'Efield_{i}{i+1}') for i in range(self.n_states-1)])
        self.r_individual = sy.symbols(
            f'\\rho_{{(0:{self.total_levels})_(0:{self.total_levels})}}')
        self.r = sy.Matrix(self.total_levels,
                           self.total_levels, self.r_individual)

    def generateMatrices(self):
        #######################################################################
        # Generate matrices
        #######################################################################
        self.H_rabi = sy.zeros(self.total_levels, self.total_levels)
        for i in range(self.n_states-1):
            self.H_rabi[self.slices[i], self.slices[i+1]] = 0.5e-6 / c.h * self.dme[i] * self.Efield[i]
        self.H_rabi = self.H_rabi + self.H_rabi.transpose()

        detunings = np.concatenate([-self.energySeparation[i]+self.wL[0:i].sum()
                                    for i in range(self.n_states)], axis=None)
        self.H_energylevels = sy.diag(*detunings)
        self.H = self.H_rabi + self.H_energylevels

        G = np.zeros((self.total_levels, self.total_levels))
        G[self.slices[0], self.slices[0]] = 1 / self.n[0] / self.transit_time
        for i in range(self.n_states-1):
            G[self.slices[i+1], self.slices[i]] = self.Gammas[i].T

        # Ref: Weller, PhD thesis, p. 14, eq. 1.13
        L = sy.zeros(self.total_levels, self.total_levels)
        for i in range(self.total_levels):
            for j in range(self.total_levels):
                L[i, i] += G[j, i] * self.r[j, j] - G[i, j] * self.r[i, i]
                if (i != j):
                    for k in range(self.total_levels):
                        L[i, j] -= 0.5 * (G[i, k] + G[j, k]) * self.r[i, j]

        self.master_equation = -sy.I * (self.H*self.r - self.r*self.H) - L#  - Ldeph

    def generate_linear_system(self):
        self.r_list = self.matrix2list(self.r)
        # Create list of off-diagonal elements relevant for i->j transition
        self.transition_list = []
        for i in range(self.n_states-1):
            mask = np.full((self.total_levels, self.total_levels), False)
            mask[self.slices[i], self.slices[i+1]] = True
            self.transition_list.append(self.matrix2list(mask))
        A, b = sy.linear_eq_to_matrix(self.system_matrix, self.r_list)
        # A = A.simplify(rational=None)
        A = se.Matrix(A)
        A = se.Lambdify([*self.wL, *self.Efield], A, real=False, cse=True)
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
            wavenumber_ge = self.f_resonance[0] / c.c

            if len(beams) != 1:
                w_er = np.atleast_1d(f_list[1])
                sgn_er = sgn_list[1]
                wavenumber_er = self.f_resonance[1] / c.c

            #######################################################################
            # Solve linear system
            #######################################################################
            if self.n_states == 2:
                t0 = datetime.now()
                res = np.array([[np.linalg.solve(self.A(w, E), self.b) for E in E_list[0]] for w in w_ge])
                print(datetime.now() - t0)
            else:
                if v is None:
                    A = [[[[self.A(self.G_01_val, self.G_10_val,
                                    w1, w2, E1, E2)
                            for E2 in E_list[1]]
                            for E1 in E_list[0]]
                            for w2 in w_er]
                            for w1 in w_ge]
                else:
                    w_ge = w_ge[0]
                    w_er = w_er[0]
                    A = [[[self.A(self.G_01_val, self.G_10_val,
                                    w_ge + sgn_ge * wavenumber_ge * vi,
                                    w_er + sgn_er * wavenumber_er * vi,
                                    E1, E2)
                            for E2 in E_list[1]]
                            for E1 in E_list[0]]
                            for vi in v]
            # res = np.linalg.solve(A, b)
            # sys.exit()

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
            k_alt = [np.divide.outer(self.dme[i].ravel(), E_list[i]) / c.epsilon_0 for i in range(self.n_states-1)]
            # - Return sum of excited states. Indexing order is given by order
            #   of arguments of 'sy.linear_eq_to_matrix' above
            # - Population of excited states is given by diagonal entries
            # - Complex-valued susceptibility is given by off-axis entries
            #   (only one side, they are complex conjugated anyway)
            state_population = np.array([np.sum(res[self.slices[i]], axis=0).real for i in range(self.n_states)])
            chi = np.array([self.atom.abundance * 2 * np.sum(res[self.transition_list[i]] * k_alt[i], axis=0)
                for i in range(self.n_states-1)])
            return state_population[1].squeeze(), chi[0].squeeze()
        else:
            return self.solve_via_LUT(beams[0])

    def solve_w_doppler(self, beams, precision='high'):
        beam_ge = beams[0]
        if len(beams) == 1:
            beam_er = None
        else:
            beam_er = beams[1]
        # chi_dopp(∆) = \int p(v,T)chi(∆-kv)dv = (p*chi)(∆)
        # k = 1 / lam2bda = w/c
        w_ge, P_ge, D_ge, sgn_ge = beam_ge
        w_ge = np.atleast_1d(w_ge)
        P_ge = np.atleast_1d(P_ge)
        k_ge = self.f_resonance[0] / c.c

        if beam_er is not None:
            w_er, P_er, D_er, sgn_er = beam_er
            w_er = np.atleast_1d(w_er)
            P_er = np.atleast_1d(P_er)
            k_er = self.f_resonance[1] / c.c

        if beam_er is None:
            exci_state = np.ones((len(w_ge), len(P_ge)), dtype='float64')
        else:
            exci_state = np.zeros((len(w_ge), len(w_er), len(P_ge)), dtype='float64')
        chi = np.zeros_like(exci_state, dtype='complex128')

        resolution = 2  # MHz
        v = np.arange(w_ge.min()/1e6 - 1000, w_ge.max()/1e6 + 1000, resolution)
        v_distribution = self.v_dist(np.subtract.outer(w_ge / k_ge, v))
        for i, P in enumerate(P_ge):
            if beam_er is None:
                # Use symmetry of convolution to calculate population_number
                # once and instead calculate Maxwell Boltzmann distribution
                # more often (which is computationally cheaper)
                E, C = self.solve([beam(w=k_ge * v, P=P, D=D_ge, sgn=1)], precision=precision)
                exci_state[:, i] = np.sum(v_distribution * E, axis=1) * resolution
                chi[:, i] = np.sum(v_distribution * C, axis=1) * resolution
            else:
                # Create nonequidistant velocity distribution
                # with equal bin probabilities. See nonequidistant_sampling.py
                # for more.
                binsize = 10000
                epsilon = 1e-6
                p_bounds = np.linspace(epsilon, 1-epsilon, binsize + 1)
                p = (p_bounds[1:] + p_bounds[:-1]) / 2
                v_bounds = self.cdfinv(p_bounds)
                v = self.cdfinv(p)
                dv = np.diff(v_bounds)
                v_distribution = self.v_dist(v)

                for m, wm in enumerate(tqdm.tqdm(w_er, position=1)):
                    for n, wn in enumerate(tqdm.tqdm(w_ge, position=0, leave=False)):
                        E, C = self.solve([beam(w=wn, P=P, D=D_ge, sgn=sgn_ge),
                                          beam(w=wm, P=P_er, D=D_er, sgn=sgn_er)], v)
                        exci_state[n, m, i] = np.sum(E * dv * v_distribution)
                        chi[n, m, i] = np.sum(C * dv * v_distribution)
        return exci_state.squeeze(), chi.squeeze()

    def get_dynamic_xaxis(self):
        bounds = [-4000, 6000]
        low_resolution = 20  # MHz
        high_resolution = 2  # MHz

        # Initial scan of optical depth to obtain threshold for regions of interest
        x = np.arange(*bounds, high_resolution) * 1e6
        y = self.optical_depth([beam(w=x, P=1e-15, D=1)], doppler=False)
        limit = 1e-4 * y.min()

        # Find bounds of those regions of interest
        idx_high = np.argwhere(y>limit).flatten()
        idx_high_l = idx_high[np.where(abs(np.diff(idx_high))>1)[0] + 0]
        idx_high_r = idx_high[np.where(abs(np.diff(idx_high))>1)[0] + 1]

        # Recreate the x axis, now with different resolutions depending on the frequency
        x_low = np.arange(*bounds, low_resolution) * 1e6
        x_high = [np.arange(*x[np.array(i)], high_resolution*1e6) for i in zip(idx_high_l, idx_high_r)]
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
            powers = np.geomspace(1e-12, 1, 15001, endpoint=True)
            beam_diameter = 1e-3
            chi = np.empty((detunings.size, powers.size), dtype=np.complex128)
            for i, p in enumerate(tqdm.tqdm(powers)):
                _, chi[:, i] = self.solve([beam(w=detunings, P=p, D=beam_diameter)], precision='high')
            np.savez_compressed(LUT_file, detunings=detunings, powers=powers, beam_diameter=beam_diameter, chi=chi)
        return detunings, powers, beam_diameter, chi

    def transmission(self, beams, z=50e-3, doppler=True, precision='high'):
        alpha = self.optical_depth(beams, doppler, precision)
        return np.exp(alpha * z)

    def absorption(self, beams, z=50e-3, doppler=True, precision='high'):
        return 1 - self.transmission(beams, z, doppler, precision)

    def optical_depth(self, beams, doppler=True, precision='high'):
        n = self.atom.getNumberDensity(self.T)
        if doppler:
            _, chi = self.solve_w_doppler(beams, precision=precision)
        else:
            _, chi = self.solve(beams, precision=precision)
        n_imag = np.sqrt(1.0 + chi * n).imag
        return 4 * c.pi * self.f_resonance[0] / c.c * n_imag

    def propagated_transmission(self, beams, z=50e-3, doppler=True, steps=50):
        w, P, D, _ = beams[0]
        dz = z / steps
        detunings, powers, beam_diameter, chi = self.LUT()
        chi_real = RectBivariateSpline(detunings, powers, chi.real, kx=1, ky=1)
        chi_imag = RectBivariateSpline(detunings, powers, chi.imag, kx=1, ky=1)

        P0 = P * (beam_diameter / D)#**2
        n = self.atom.getNumberDensity(self.T)
        k = self.f_resonance[0] / c.c
        resolution = 2  # MHz
        v = np.arange(w.min()/1e6 - 1000, w.max()/1e6 + 1000, resolution)
        v_distribution = self.v_dist(np.subtract.outer(w / k, v))
        # Initialize some variables
        P = np.zeros((steps+1, len(beams[0].w)))
        T = np.ones((steps+1, len(beams[0].w)))
        P[0] = P0
        abs_pref = dz * 4 * c.pi * self.f_resonance[0] / c.c

        for i in range(1, steps+1):
            # RectBivariateSpline wants strictly increasing values for x and y.
            # The detuning fulfills this naturally, but power not.
            # So we sort it and later "unsort" it again
            if doppler:
                sequence = np.argsort(P[i-1])
                chi_t = chi_real(k*v, P[i-1][sequence], grid=True) + 1j * chi_imag(k*v, P[i-1][sequence], grid=True)
                chi = np.sum(v_distribution * chi_t.T[sequence.argsort()], axis=1) * resolution
            else:
                chi = chi_real(w, P[i-1], grid=False) + 1j * chi_imag(w, P[i-1], grid=False)
            T[i] = np.exp(abs_pref * np.sqrt(1. + chi * n).imag)
            P[i] = T[i] * P[i-1]
        T = np.product(T, axis=0)
        return T


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
