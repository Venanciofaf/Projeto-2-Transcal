import unittest
import numpy as np
from math import isclose

# Importa as funções e classes do seu código (suponha que estão em `truss_solver.py`)
from projeto import TrussElement, assemble_global_stiffness_matrix, apply_boundary_conditions, gauss_seidel

class TestTrussSolver(unittest.TestCase):

    def setUp(self):
        self.elements = [
            TrussElement(0, 1, area=0.0001, elasticity=2e11, theta_deg=0),
            TrussElement(1, 2, area=0.0001, elasticity=2e11, theta_deg=60),
            TrussElement(0, 2, area=0.0001, elasticity=2e11, theta_deg=90)
        ]
        self.num_nodes = 3
        self.K_global = assemble_global_stiffness_matrix(self.elements, self.num_nodes)
        self.F = np.zeros(self.num_nodes * 2)
        self.F[3] = -150  # Força no nó 1 em y
        self.boundary_conditions = [0, 1, 4, 5]
        self.K_bc, self.F_bc = apply_boundary_conditions(self.K_global.copy(), self.F.copy(), self.boundary_conditions)

    def test_matrix_shape(self):
        """Testa se a matriz de rigidez global tem a forma correta."""
        self.assertEqual(self.K_global.shape, (6, 6))

    def test_boundary_conditions_applied(self):
        """Verifica se as condições de contorno foram aplicadas corretamente."""
        for idx in self.boundary_conditions:
            row = self.K_bc[idx]
            col = self.K_bc[:, idx]
            self.assertTrue(np.allclose(row, np.eye(6)[idx]))
            self.assertTrue(np.allclose(col, np.eye(6)[:, idx]))
            self.assertEqual(self.F_bc[idx], 0)

    def test_gauss_seidel_convergence(self):
        """Testa se o método de Gauss-Seidel converge e retorna valores plausíveis."""
        disp = gauss_seidel(self.K_bc, self.F_bc)
        self.assertEqual(len(disp), 6)
        self.assertTrue(isclose(disp[3], -1.24e-5, rel_tol=1e-1))  # deslocamento no nó central em y

    def test_displacement_zero_on_fixed_nodes(self):
        """Verifica se nós fixos têm deslocamento zero."""
        disp = gauss_seidel(self.K_bc, self.F_bc)
        for idx in self.boundary_conditions:
            self.assertAlmostEqual(disp[idx], 0.0, places=8)

if __name__ == "__main__":
    unittest.main()
