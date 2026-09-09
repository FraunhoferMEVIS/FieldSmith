# Dipole and cuboid field models

FieldSmith evaluates the magnetic flux density produced by an assembly as the
sum of the contributions from its individual magnets. Positions and dimensions
are expressed in metres, remanence and the resulting field in tesla.

## Dipole approximation

`calc_b_at_points_fast` represents magnet $i$, centred at $\mathbf r_i$, by
a point dipole. For volume $V_i$, remanence magnitude $B_{r,i}$, and unit
magnetization direction $\hat{\mathbf m}_i$,

$
\mathbf M_i = \frac{B_{r,i}}{\mu_0}\hat{\mathbf m}_i,
\qquad
\mathbf m_i = V_i\mathbf M_i
            = \frac{V_i}{\mu_0}\mathbf B_{r,i}.
$

At an evaluation point $\mathbf r$, define
$\mathbf R_i=\mathbf r-\mathbf r_i$. The implementation computes

$
\mathbf B(\mathbf r)=\frac{\mu_0}{4\pi}
\sum_i\left[
\frac{3\mathbf R_i(\mathbf m_i\cdot\mathbf R_i)}{\lVert\mathbf R_i\rVert^5}
-\frac{\mathbf m_i}{\lVert\mathbf R_i\rVert^3}
\right].
$

The inputs `dipole_br`, `dipole_pos`, and `points` have shapes `(M, 3)`,
`(M, 3)`, and `(P, 3)`. `vol` is scalar or has shape `(M,)`. Evaluation may be
chunked over points and magnets without changing the sum. Distances are clamped
to `eps` to avoid division by zero. The approximation neglects finite magnet
dimensions and is therefore least accurate near a magnet.

## Finite-size cuboid model

`calc_b_at_points_cuboid` evaluates the closed-form exterior field of uniformly
magnetized cuboids. Consider one cuboid in its local frame, with half-lengths
$a,b,c$ and point coordinates $(x,y,z)$ relative to its centre. For
$p,q,s\in\{-1,1\}$, define

$
x_p=x-pa,\qquad y_q=y-qb,\qquad z_s=z-sc,
$

$
\rho_{pqs}=\sqrt{x_p^2+y_q^2+z_s^2}.
$

For remanence $B_{r,z}$ along the local $z$-axis, the code evaluates

$
\mathbf B^{(z)}(x,y,z)=\frac{B_{r,z}}{4\pi}
\sum_{p,q,s}(-pqs)
\begin{bmatrix}
\ln(y_q+\rho_{pqs})\\
\ln(x_p+\rho_{pqs})\\
-\operatorname{atan2}(x_py_q,z_s\rho_{pqs})
\end{bmatrix}.
$

The local $x$- and $y$-magnetized contributions are obtained by cyclically
permuting the coordinates, dimensions, and remanence components. Their sum gives
the field for an arbitrary remanence vector.

Cuboids are world-axis-aligned when `axes` is omitted. Otherwise, each matrix
$Q_i$ stores the local body axes as rows. The code transforms the displacement
and remanence into the body frame, evaluates the local field, and transforms it
back:

$
\mathbf r'_i=Q_i(\mathbf r-\mathbf r_i),\qquad
\mathbf B'_{r,i}=Q_i\mathbf B_{r,i},\qquad
\mathbf B_i(\mathbf r)=Q_i^{\mathsf T}\mathbf B'_i(\mathbf r'_i).
$

`magnet_br`, `magnet_size`, and `magnet_pos` have shape `(M, 3)`;
`magnet_size` contains full side lengths. `axes`, when present, has shape
`(M, 3, 3)`. The implementation adds `eps` to removable logarithm and arctangent
singularities and processes points in chunks. It returns the exterior field; an
interior-field calculation would additionally require the material contribution
$\mathbf B=\mu_0(\mathbf H+\mathbf M)$.

## Rectangular-prism scalar potential

[`rectangular_prism.py`](../src/fieldsmith/field_simulations/rectangular_prism.py)
implements the calculation from *The Magnetic Scalar Potential for a Rectangular
Prism* by James B. et al. The exterior field is obtained from the scalar potential
using automatic differentiation.
