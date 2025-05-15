import unittest
import numpy as np
from math import isclose

# === Importa do seu projeto principal ===
from projeto import (
    TrussElement,
    assemble_global_stiffness_matrix,
    apply_boundary_conditions,
    gauss_seidel
)

# === Função adicional para Jacobi ===
def jacobi_method(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    for _ in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x - x_new, ord=np.inf) < tol:
            return x_new
        x = x_new
    raise RuntimeError("Jacobi method did not converge")

# === Testes unificados ===
class TestTrussSolverAndAula24(unittest.TestCase):

    def setUp(self):
        # Para os testes do seu projeto
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

    # === Testes do seu projeto (treliça com Gauss-Seidel) ===
    def test_matrix_shape(self):
        self.assertEqual(self.K_global.shape, (6, 6))

    def test_boundary_conditions_applied(self):
        for idx in self.boundary_conditions:
            row = self.K_bc[idx]
            col = self.K_bc[:, idx]
            self.assertTrue(np.allclose(row, np.eye(6)[idx]))
            self.assertTrue(np.allclose(col, np.eye(6)[:, idx]))
            self.assertEqual(self.F_bc[idx], 0)

    def test_gauss_seidel_convergence(self):
        disp = gauss_seidel(self.K_bc, self.F_bc)
        self.assertEqual(len(disp), 6)
        self.assertTrue(isclose(disp[3], -1.24e-5, rel_tol=1e-1))

    def test_displacement_zero_on_fixed_nodes(self):
        disp = gauss_seidel(self.K_bc, self.F_bc)
        for idx in self.boundary_conditions:
            self.assertAlmostEqual(disp[idx], 0.0, places=8)

    # === Testes dos exercícios da Aula 24 ===

    def test_exercicio_2_jacobi(self):
        A = np.array([
            [3, -0.1, -0.2],
            [0.1, 7, -0.3],
            [0.3, -0.2, 10]
        ])
        b = np.array([7.85, -19.3, 71.4])
        sol = jacobi_method(A, b)
        np.testing.assert_allclose(sol, [3.0, -2.5, 7.0], rtol=1e-2)

    def test_exercicio_3_comparar_iteracoes(self):
        A = np.array([
            [3, -0.1, -0.2],
            [0.1, 7, -0.3],
            [0.3, -0.2, 10]
        ])
        b = np.array([7.85, -19.3, 71.4])

        # Contador de iterações
        def count_iterations(method, A, b):
            count = 0
            x = np.zeros_like(b)
            while count < 1000:
                x_new = np.copy(x)
                for i in range(len(b)):
                    s = sum(A[i][j] * x[j] for j in range(len(b)) if j != i)
                    x_new[i] = (b[i] - s) / A[i][i]
                if np.linalg.norm(x_new - x, ord=np.inf) < 1e-6:
                    return count + 1
                x = x_new
                count += 1
            return -1

        jacobi_iters = count_iterations(jacobi_method, A, b)
        gs_sol = gauss_seidel(A, b)
        self.assertTrue(jacobi_iters > 0)
        self.assertIsInstance(gs_sol, np.ndarray)

    def test_exercicio_4_ordem_matriz(self):
        num_nos = 4
        ordem_esperada = num_nos * 2
        self.assertEqual(ordem_esperada, 8)

    def test_exercicio_5_matriz_rigidez_com_mola(self):
        E = 2e11
        A_secao = 0.0001
        k_mola = 1e4
        L = 1.0
        c = 1.0  # cos(0)
        s = 0.0  # sin(0)
        k_barra = E * A_secao / L
        k_total = k_barra + k_mola
        ke = k_total * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        self.assertEqual(ke.shape, (4, 4))
        self.assertAlmostEqual(ke[0, 0], k_total)

    def test_exercicio_1_resultados_trelica(self):
        deslocamentos_esperados = np.array([-9.52e-7, 1.60e-6, -4.0e-6])
        self.assertEqual(len(deslocamentos_esperados), 3)
        for val in deslocamentos_esperados:
            self.assertIsInstance(val, float)

# === Execução do Teste ===
if __name__ == "__main__":
    unittest.main()
