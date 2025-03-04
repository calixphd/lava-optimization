# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from numpy import typing as npty

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class AbstractScif(AbstractProcess):
    """Abstract Process for Stochastic Constraint Integrate-and-Fire
    (SCIF) neurons.
    """

    def __init__(
            self,
            *,
            shape: ty.Tuple[int, ...],
            step_size: ty.Optional[int] = 1,
            theta: ty.Optional[int] = 4,
            noise_amplitude: ty.Optional[int] = 0) -> None:
        """
        Stochastic Constraint Integrate and Fire neuron Process.
        Parameters
        ----------
        shape: Tuple
            Number of neurons. Default is (1,).
        step_size: int
            bias current driving the SCIF neuron. Default is 1 (arbitrary).
        theta: int
            threshold above which a SCIF neuron would fire winner-take-all
            spike. Default is 4 (arbitrary).
        """
        super().__init__(shape=shape)

        self.a_in = InPort(shape=shape)
        self.s_sig_out = OutPort(shape=shape)
        self.s_wta_out = OutPort(shape=shape)

        self.state = Var(shape=shape, init=np.zeros(shape=shape).astype(int))
        self.spk_hist = Var(shape=shape, init=np.zeros(shape=shape).astype(int))
        self.noise_ampl = Var(shape=shape, init=noise_amplitude)

        self.step_size = Var(shape=shape, init=int(step_size))
        self.theta = Var(shape=(1,), init=int(theta))

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return self.proc_params['shape']


class CspScif(AbstractScif):
    """Stochastic Constraint Integrate-and-Fire neurons to solve CSPs.
    """

    def __init__(self,
                 *,
                 shape: ty.Tuple[int, ...],
                 step_size: ty.Optional[int] = 1,
                 theta: ty.Optional[int] = 4,
                 neg_tau_ref: ty.Optional[int] = -5,
                 noise_amplitude: ty.Optional[int] = 0):

        super(CspScif, self).__init__(shape=shape,
                                      step_size=step_size,
                                      theta=theta,
                                      noise_amplitude=noise_amplitude)
        self.neg_tau_ref = Var(shape=(1,), init=int(neg_tau_ref))
        self.cnstr_intg = Var(shape=shape, init=np.zeros(shape=shape).astype(
            int))


class QuboScif(AbstractScif):
    """Stochastic Constraint Integrate-and-Fire neurons to solve QUBO
    problems.
    """

    def __init__(self,
                 *,
                 shape: ty.Tuple[int, ...],
                 cost_diag: npty.NDArray,
                 step_size: ty.Optional[int] = 1,
                 theta: ty.Optional[int] = 4,
                 noise_amplitude: ty.Optional[int] = 0,
                 noise_shift: ty.Optional[int] = 8):

        super(QuboScif, self).__init__(shape=shape,
                                       step_size=step_size,
                                       theta=theta,
                                       noise_amplitude=noise_amplitude)
        self.cost_diagonal = Var(shape=shape, init=cost_diag)
        # User provides a desired precision. We convert it to the amount by
        # which unsigned 16-bit noise is right-shifted:
        self.noise_shift = Var(shape=shape, init=noise_shift)


class Boltzmann(AbstractProcess):
    """Stochastic Constraint Integrate-and-Fire neurons to solve QUBO
    problems.
    """

    def __init__(self,
                 *,
                 shape: ty.Tuple[int, ...],
                 temperature: ty.Optional[ty.Union[int, npty.NDArray]] = 1,
                 refract: ty.Optional[ty.Union[int, npty.NDArray]] = 0,
                 init_value=0,
                 init_state=0):
        """
         Stochastic Constraint Integrate and Fire neuron Process.

         Parameters
         ----------
         shape: Tuple
             Number of neurons. Default is (1,).
         temperature: ArrayLike
             Temperature of the system, defining the level of noise.
         """
        super().__init__(shape=shape)

        self.a_in = InPort(shape=shape)
        self.s_sig_out = OutPort(shape=shape)
        self.s_wta_out = OutPort(shape=shape)

        self.spk_hist = Var(
            shape=shape, init=(np.zeros(shape=shape) + init_value).astype(int)
        )

        self.temperature = Var(shape=shape, init=int(temperature))

        self.refract = Var(shape=shape, init=refract)

        self.debug = Var(shape=shape, init=0)

        # Initial state determined in DiscreteVariables
        self.state = Var(shape=shape, init=init_state.astype(int))

        @property
        def shape(self) -> ty.Tuple[int, ...]:
            return self.proc_params['shape']
