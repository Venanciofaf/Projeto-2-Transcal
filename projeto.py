import numpy as np

# Classe TrussElement adaptada com cálculo do comprimento
class TrussElement:
    def __init__(self, node_start, node_end, area, elasticity, coord):
        self.node_start = node_start
        self.node_end = node_end
        self.area = area
        self.elasticity = elasticity

        x1, y1 = coord[node_start]
        x2, y2 = coord[node_end]
        self.length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dx = x2 - x1
        dy = y2 - y1
        self.c = dx / self.length
        self.s = dy / self.length

        k = (area * elasticity) / self.length
        c = self.c
        s = self.s
        self.k_global = k * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])

    def get_dof_indices(self):
        return [
            2 * self.node_start,       # ux_i
            2 * self.node_start + 1,   # uy_i
            2 * self.node_end,         # ux_j
            2 * self.node_end + 1      # uy_j
        ]

# Montagem da matriz de rigidez global
def assemble_global_stiffness_matrix(elements, num_nodes):
    K_global = np.zeros((num_nodes * 2, num_nodes * 2))
    for element in elements:
        dof = element.get_dof_indices()
        for i in range(4):
            for j in range(4):
                K_global[dof[i], dof[j]] += element.k_global[i, j]
    return K_global

# Aplicação das condições de contorno
def apply_boundary_conditions(K, F, boundary_conditions):
    for idx in boundary_conditions:
        K[idx, :] = 0
        K[:, idx] = 0
        K[idx, idx] = 1
        F[idx] = 0
    return K, F

# Método de Gauss-Seidel
def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=10000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    for it in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    raise RuntimeError("Gauss-Seidel did not converge")

# Dados da treliça (coordenadas dos nós)
coords = {
    0: (0.00, 0.00),  # Nó 1
    1: (0.00, 0.40),  # Nó 2
    2: (0.30, 0.40)   # Nó 3
}

# Propriedades dos elementos
E = 210e9        # Pa
A = 2e-4         # m²

# Definição dos elementos
elements = [
    TrussElement(0, 1, A, E, coords),  # Elemento 1: Nó 1-2
    TrussElement(1, 2, A, E, coords),  # Elemento 2: Nó 2-3
    TrussElement(0, 2, A, E, coords)   # Elemento 3: Nó 1-3
]

# Número de nós
num_nodes = 3

# Montar matriz de rigidez global
K_global = assemble_global_stiffness_matrix(elements, num_nodes)

# Vetor de forças externas (somente nó 2 com Fy = -1000 N)
F = np.zeros(num_nodes * 2)
F[3] = -1000  # Força negativa em y do nó 2

# Condições de contorno: Nó 1 (0) e Nó 3 (2) estão fixos (x e y)
boundary_conditions = [0, 1, 4, 5]

# Aplicar restrições
K_bc, F_bc = apply_boundary_conditions(K_global.copy(), F.copy(), boundary_conditions)

# Resolver sistema
displacements = gauss_seidel(K_bc, F_bc)

# Resultado
print("Deslocamentos nodais (em metros):")
print(displacements)
