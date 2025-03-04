# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.lib.optimization.problems.constraints import (
    Constraints,
    DiscreteConstraints,
)
from lava.lib.optimization.problems.cost import Cost
from lava.lib.optimization.problems.problems import (CSP, OptimizationProblem,
                                                     QUBO)
from lava.lib.optimization.problems.variables import (DiscreteVariables,
                                                      Variables)


class CompliantInterfaceInheritance(OptimizationProblem):
    @property
    def cost(self):
        return None

    @property
    def constraints(self):
        return None

    @property
    def variables(self):
        return None


class NotCompliantInterfaceInheritance(OptimizationProblem):
    def constraints(self):
        return None

    def variables(self):
        return None


class TestOptimizationProblem(unittest.TestCase):
    def setUp(self) -> None:
        self.compliant_instantiation = CompliantInterfaceInheritance()

    def test_cannot_create_instance(self):
        with self.assertRaises(TypeError):
            OptimizationProblem()

    def test_compliant_sublcass(self):
        self.assertIsInstance(self.compliant_instantiation, OptimizationProblem)

    def test_not_compliant_sublcass(self):
        with self.assertRaises(TypeError):
            NotCompliantInterfaceInheritance()

    def test_variables_attribute(self):
        self.assertIsInstance(
            self.compliant_instantiation._variables, Variables
        )

    def test_constraints_attribute(self):
        self.assertIsInstance(
            self.compliant_instantiation._constraints, Constraints
        )


class TestQUBO(unittest.TestCase):
    def setUp(self) -> None:
        self.qubo = QUBO(np.eye(10, dtype=int))

    def test_create_obj(self):
        self.assertIsInstance(self.qubo, QUBO)

    def test_q_is_square_matrix(self):
        n, m = self.qubo.cost.coefficients[2].shape
        self.assertEqual(n, m)

    def test_cost_class(self):
        self.assertIsInstance(self.qubo.cost, Cost)

    def test_only_quadratic_term_in_cost(self):
        self.assertEqual(list(self.qubo.cost.coefficients.keys()), [2])

    def test_variables_class(self):
        self.assertIsInstance(self.qubo.variables, DiscreteVariables)

    def test_variables_are_binary(self):
        for n, size in enumerate(self.qubo.variables.domain_sizes):
            with self.subTest(msg=f"Var ID {n}"):
                self.assertEqual(size, 2)

    def test_number_of_variables(self):
        self.assertEqual(self.qubo.variables.num_variables, 10)

    def test_constraints_is_none(self):
        self.assertIsNone(self.qubo.constraints)

    def test_cannot_set_constraints(self):
        with self.assertRaises(AttributeError):
            self.qubo.constraints = np.eye(10)

    def test_set_cost(self):
        new_cost = np.eye(4, dtype=int)
        self.qubo.cost = new_cost
        self.assertIs(self.qubo.cost.get_coefficient(2), new_cost)

    def test_variables_update_after_setting_cost(self):
        new_cost = np.eye(4, dtype=int)
        self.qubo.cost = new_cost
        self.assertEqual(self.qubo.variables.num_variables, 4)

    def test_class_of_setted_cost(self):
        new_cost = np.eye(10, dtype=int)
        self.qubo.cost = new_cost
        self.assertIsInstance(self.qubo.cost, Cost)

    def test_cannot_set_nonquadratic_cost(self):
        with self.assertRaises(ValueError):
            self.qubo.cost = np.eye(10, dtype=int).reshape(5, 20)

    def test_assertion_raised_if_q_is_not_square(self):
        with self.assertRaises(ValueError):
            QUBO(np.eye(10, dtype=int).reshape(5, 20))

    def test_validate_input_method_fails_assertion(self):
        with self.assertRaises(ValueError):
            self.qubo.validate_input(np.eye(10).reshape(5, 20))

    def test_validate_input_method_does_not_fail_assertion(self):
        try:
            self.qubo.validate_input(np.eye(10, dtype=int))
        except AssertionError:
            self.fail("Assertion failed with correct input!")


class TestCSP(unittest.TestCase):
    def setUp(self) -> None:
        self.domain_sizes = [5, 5, 4]
        self.constraints = [
            (0, 1, np.logical_not(np.eye(5))),
            (1, 2, np.eye(5, 4)),
        ]
        self.csp = CSP(domains=[5, 5, 4], constraints=self.constraints)

    def test_create_obj(self):
        self.assertIsInstance(self.csp, CSP)

    def test_cost_class(self):
        self.assertIsInstance(self.csp.cost, Cost)

    def test_cost_is_constant(self):
        self.assertEqual(self.csp.cost.max_degree, 0)

    def test_cannot_set_cost(self):
        with self.assertRaises(AttributeError):
            self.csp.cost = np.eye(10)

    def test_variables_class(self):
        self.assertIsInstance(self.csp.variables, DiscreteVariables)

    def test_variables_domain_sizes(self):
        for n, size in enumerate(self.csp.variables.domain_sizes):
            with self.subTest(msg=f"Var ID {n}"):
                self.assertEqual(size, self.domain_sizes[n])

    def test_constraints_class(self):
        self.assertIsInstance(self.csp.constraints, DiscreteConstraints)

    def test_constraints_var_subsets(self):
        for n, (v1, v2, rel) in enumerate(self.constraints):
            with self.subTest(msg=f"Constraint {n}."):
                self.assertEqual(self.csp.constraints.var_subsets[n], (v1, v2))

    def test_constraints_relations(self):
        for n, (v1, v2, rel) in enumerate(self.constraints):
            with self.subTest(msg=f"Constraint {n}."):
                self.assertTrue(
                    (self.csp.constraints.relations[n] == rel).all()
                )

    def test_set_constraints(self):
        new_constraints = [(0, 1, np.eye(5))]
        self.csp.constraints = new_constraints
        self.assertIs(self.csp.constraints._constraints, new_constraints)

    @unittest.skip("WIP")
    def test_validate_input_method_fails_assertion(self):
        with self.assertRaises(AssertionError):
            self.csp.validate_input(np.eye(10).reshape(5, 20))

    @unittest.skip("WIP")
    def test_validate_input_method_does_not_fail_assertion(self):
        try:
            self.csp.validate_input(np.eye(10))
        except AssertionError:
            self.fail("Assertion failed with correct input!")


if __name__ == "__main__":
    unittest.main()
