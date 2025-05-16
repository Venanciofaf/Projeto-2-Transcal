import numpy as np

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
def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=20_000):
    """
    Resolve A·x = b  e devolve  (x, n_iter).
    """
    A = A.astype(float)
    b = b.astype(float)

    n = len(b)
    x = np.zeros(n) if x0 is None else x0.astype(float).copy()

    diag = np.diag(A)
    if np.any(np.isclose(diag, 0)):
        raise ZeroDivisionError("Zero na diagonal de A.")

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            sigma  = np.dot(A[i, :i],     x[:i])
            sigma += np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i]   = (b[i] - sigma) / diag[i]

        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x, k

    raise RuntimeError("Gauss-Seidel não convergiu.")

def calc_strain_stress(elements, u_vec):
    """
    Parameters
    ----------
    elements : list[TrussElement]
    u_vec    : ndarray  (n_gdl,)   vetor completo de deslocamentos

    Returns
    -------
    eps : ndarray (n_elem,)  deformações
    sig : ndarray (n_elem,)  tensões (Pa)
    """
    eps, sig = [], []
    for el in elements:
        dof = el.get_dof_indices()
        ux_i, uy_i, ux_j, uy_j = u_vec[dof]
        delta = el.c*(ux_j - ux_i) + el.s*(uy_j - uy_i)
        strain = delta / el.length
        stress = el.elasticity * strain
        eps.append(strain)
        sig.append(stress)
    return np.array(eps), np.array(sig)











# Dados da treliça (coordenadas dos nós)
coords = {
    0: (0.00, 0.00),
    1: (0.00, 0.40),
    2: (0.30, 0.40)
}

# Propriedades dos elementos
E = 210e9        # Pa
A = 2e-4         # m²

# Definição dos elementos
elements = [
    TrussElement(0, 1, A, E, coords),
    TrussElement(1, 2, A, E, coords),
    TrussElement(0, 2, A, E, coords)
]

# Número de nós
num_nodes = 3

# Montar matriz de rigidez global
K_global = assemble_global_stiffness_matrix(elements, num_nodes)

# Vetor de forças externas (somente nó 2 com Fy = -1000 N)
F = np.zeros(num_nodes * 2)

F[4] = 150
F[5] = -100

# Condições de contorno: Nó 1 (0) e Nó 3 (2) estão fixos (x e y)
boundary_conditions = [0, 2, 3]

# Aplicar restrições
K_bc, F_bc = apply_boundary_conditions(K_global.copy(), F.copy(), boundary_conditions)
u, n_iter = gauss_seidel(K_bc, F_bc)

free_dofs = [i for i in range(num_nodes*2) if i not in boundary_conditions]
# índices livres: 1-based → 2, 5, 6   (ou 0-based → 1, 4, 5)
rowcol_lbls = [d + 1 for d in free_dofs]

K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
F_reduced = F[free_dofs]
u_red, n_iter_red = gauss_seidel(K_reduced, F_reduced)

# ----------------------------------------
#  TABELA "INFORMAÇÕES DOS NÓS"
# ----------------------------------------

title   = "Informações dos nós"
headers = ("Número do nó", "x (m)", "y (m)")
rows = [[k+1, *coords[k]] for k in sorted(coords)]

row_fmt = "{:^13}|{:^7}|{:^7}"

line_len = 31
print("=" * line_len)
print(f"{title:^{line_len}}")
print("=" * line_len)
print(row_fmt.format(*headers))
print("-" * line_len)

for n, x, y in rows:
    print(row_fmt.format(n, f"{x:.2f}", f"{y:.2f}"))

print("=" * line_len)

# ----------------------------------------
#  TABELA "PROPRIEDADES DOS ELEMENTOS"
# ----------------------------------------

def build_element_rows(elements):
    rows = []
    for idx, el in enumerate(elements, start=1):
        incidence = f"{el.node_start+1}-{el.node_end+1}"
        dofs      = [d+1 for d in el.get_dof_indices()]
        rows.append([
            idx, incidence,
            f"{el.area:.0e}", f"{el.elasticity:.0e}",
            f"{el.c:+.1f}", f"{el.s:+.1f}",
            f"{el.length:.2f}", " ".join(map(str, dofs))
        ])
    return rows

headers_el = ["Nº", "Incidência", "Área (m²)", "E (Pa)", "c", "s", "L (m)", "GDL"]
rows_el    = build_element_rows(elements)

# ---- larguras de coluna ----
col_w   = [max(len(str(r[i])) for r in rows_el + [headers_el])
           for i in range(len(headers_el))]
row_fmt = " | ".join(f"{{:^{w}}}" for w in col_w)
tot_w   = sum(col_w) + 3*len(col_w) - 1

title = "Propriedades dos elementos"
print("=" * tot_w)
print(f"{title:^{tot_w}}")
print("=" * tot_w)
print(row_fmt.format(*headers_el))
print("-" * tot_w)
for row in rows_el:
    print(row_fmt.format(*row))
print("=" * tot_w)

# --------------------------------------------------
#  VISUALIZAÇÃO DA MATRIZ COM OS GRAUS DE LIBERDADE
# --------------------------------------------------

def print_matrix_with_labels(K, row_labels=None, col_labels=None, fmt="{:>11.2e}"):
    """
    Imprime a matriz K com cabeçalhos de linhas/colunas e colchetes laterais.

    Parameters
    ----------
    K : 2-D array-like (n x n)
    row_labels, col_labels : list/tuple de str|int, length = n
        Se None, usa índices iniciando em 1.
    fmt : str
        Formato de cada valor (ex.: "{:>10.3f}", "{:>11.2e}", …)
    """
    K = np.asarray(K)
    n  = K.shape[0]
    if row_labels is None:
        row_labels = list(range(1, n+1))
    if col_labels is None:
        col_labels = list(range(1, n+1))

    # largura de cada célula
    cell_width = max(len(fmt.format(np.max(np.abs(K)))), 8)
    fval = "{:>"+str(cell_width)+".2e}"
    flab = "{:^"+str(cell_width)+"}"

    # ---- cabeçalho das colunas ----
    header = " " * (cell_width + 3)
    header += "".join(flab.format(l) for l in col_labels)
    print(header)

    # ---- matriz linha a linha ----
    for i, rlab in enumerate(row_labels):

        line = flab.format(rlab) + " ["
        line += "".join(fval.format(v) for v in K[i])
        line += " ]"
        print(line)

print("\nMatriz K global:")
print_matrix_with_labels(K_global)

# ----------------------------------------
# IMPRESSÃO DO VETOR GLOBAL DE FORÇAS  [F]
# ----------------------------------------

def print_force_vector_matrix_style(F_vec, bc_0_based, col_name="F(N)"):
    """
    Mostra o vetor global de forças no mesmo estilo de print_matrix_with_labels
    (colchetes, cabeçalhos e largura de 11 caracteres).

    ▸ GDL restrito → "R_<nó><x/y>"
    ▸ GDL livre    → valor numérico 11.2e
    """
    import numpy as np
    F_vec  = np.asarray(F_vec, dtype=float)
    fixed  = {i + 1 for i in bc_0_based}
    n      = F_vec.size
    labels = list(range(1, n + 1))

    cell_w = 11
    fval   = "{:>"+str(cell_w)+".2e}"
    flab   = "{:^"+str(cell_w)+"}"

    # ---------- cabeçalho ----------
    header = " " * (cell_w + 3) + flab.format(col_name)
    print(header)

    # ---------- linhas ----------
    for gdl, val in zip(labels, F_vec):

        if gdl in fixed:
            node = (gdl + 1) // 2
            dir_ = 'x' if gdl % 2 == 1 else 'y'
            cell = flab.format(f"R_{node}{dir_}")
        else:
            cell = fval.format(val)

        print(flab.format(gdl) + " [" + cell + " ]")
    print()

# ------------  CHAMADA  ------------
print("\nVetor Global de Forças [F] (incógnitas nas reações):")
print_force_vector_matrix_style(F, boundary_conditions)

# ================================================
#  MATRIZ DE RIGIDEZ REDUZIDA  (após B.C.)
# ================================================

print("\nMatriz K reduzida:")
print_matrix_with_labels(
        K_reduced,
        row_labels=rowcol_lbls,
        col_labels=rowcol_lbls,
        fmt="{:>11.2e}"
)

# ===========================
#   VETOR DE FORÇAS REDUZIDO 
# ===========================

print("\nMatriz F reduzido:")
print_matrix_with_labels(
        F_reduced.reshape(-1, 1),
        row_labels=rowcol_lbls,
        col_labels=["F(N)"],
        fmt="{:>11.2e}"
)

# =========================================================
#  DESLOCAMENTOS REDUZIDOS  (Gauss-Seidel em K_red · u = F)
# =========================================================

print(f"\nMatriz u reduzida (convergiu em {n_iter_red} iterações):")
print_matrix_with_labels(
        u_red.reshape(-1, 1),
        row_labels=rowcol_lbls,
        col_labels=["u(m)"],
        fmt="{:>11.2e}"
)

# ==============================================
#  DESLOCAMENTOS (Gauss-Seidel em K · u = F)
# ==============================================

print(f"\nMatriz u estendida (convergiu em {n_iter} iterações):")
print_matrix_with_labels(
        u.reshape(-1, 1),
        col_labels=["u(m)"],
        fmt="{:>11.2e}"
)
    
# =========================================================
#  VETOR GLOBAL DE FORÇAS  F  (forças + reações)
# =========================================================

reac = K_global @ u - F
F_N = F.copy()
F_N[boundary_conditions] = reac[boundary_conditions]

print("\nReações nos apoios (N):")
for idx in boundary_conditions:
    node = (idx // 2) + 1
    dir_ = 'x' if idx % 2 == 0 else 'y'
    label = f"R_{node}{dir_:s}"
    print(f"  {label:<4}= {reac[idx]: .2e}")

print("\nMatriz F(N) completa (forças externas + reações):")
print_matrix_with_labels(
        F_N.reshape(-1, 1),
        row_labels=list(range(1, num_nodes*2 + 1)),
        col_labels=["F(N)"],
        fmt="{:>11.2e}"
)
print("\n")

# =================================================================
#   TABELA DE DEFORMAÇÃO E DE TENSÃO EM CADA ELEMENTO DA TRELIÇA
# =================================================================

# ------------------ cálculo ε / σ ------------------
eps, sig = calc_strain_stress(elements, u)

def print_strain_stress_table(eps, sig):
    """
    eps, sig : 1-D arrays (ou listas) do mesmo tamanho
    """
    header = ["Elemento", "Deformação", "Tensão (Pa)", "Tipo"]

    # ---------- linhas ----------
    rows = []
    for idx, (e, s) in enumerate(zip(eps, sig), start=1):
        tipo = "Tração" if s >= 0 else "Compressão"
        rows.append([str(idx), f"{e:.3e}", f"{s:.3e}", tipo])

    # ---------- larguras ----------
    ncol  = len(header)
    width = [0]*ncol
    for j in range(ncol):
        width[j] = max(len(header[j]), *(len(r[j]) for r in rows))

    # ---------- helpers ----------
    def fmt(text, j, align="center"):
        if align == "right":
            return text.rjust(width[j])
        if align == "left":
            return text.ljust(width[j])
        # center
        pad = width[j] - len(text)
        return " "*(pad//2) + text + " "*(pad - pad//2)

    # ---------- imprime cabeçalho ----------
    head_line = "| " + " | ".join(fmt(header[j], j) for j in range(ncol)) + " |"
    dash_line = "|-" + "-|-".join("-"*width[j] for j in range(ncol)) + "-|"
    print(head_line)
    print(dash_line)

    # ---------- imprime linhas ----------
    for r in rows:
        print("| "
              + fmt(r[0], 0, "right")  + " | "
              + fmt(r[1], 1)            + " | "
              + fmt(r[2], 2)            + " | "
              + fmt(r[3], 3, "left")    + " |")
    print()


# -------------- chamada ---------------
print("Tabela de deformações e tensões:")
print()
print_strain_stress_table(eps, sig)