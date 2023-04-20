# Copyright (C) 2023 Ivan Fumagalli
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from numpy import isclose
from dolfin import *
from multiphenics import *

import os

"""
Steady Stokes problem.
uS: Stokes velocity in H^1(OmS)
pS: Stokes pressure in L^2(OmS)
"""

parameters["ghost_mode"] = "shared_facet"  # required by dS

# ********* I/O parameters  ******* #

outputPath = "output"
outputFileBasename = "stokes"

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# ********* Model constants  ******* #

G_ = 1.
G = Constant(G_)
l_ = 1.
l = Constant(l_)
k = 1.
Kval = Constant(k)
KvalCorr = Constant(max(k, 1.))
mu_ = 1.
mu = Constant(mu_)
beta_ = 1.
beta = Constant(beta_)

alpha_ = 1.+2*G_+2*l_
alpha = Constant(alpha_)

# ********* Numerical method constants  ******* #

deg = 1
degP = 1
etaU = 10
eta = 10
tau = 1

# ******* Exact solution and sources ****** #

exactdeg = 7

A = pi*pi*k/G_
dP_ex = Expression(("-cos(pi*x[0])*cos(pi*x[1])",\
                    "sin(pi*x[0])*sin(pi*x[1])"), degree=exactdeg)
pP_ex = Expression("pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", degree=exactdeg)
uS_ex = Expression(("A*(cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]))",\
                    "-A*(cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]))"), degree=exactdeg, A=A)
pS_ex = Expression("(1+2*mu_*A)*pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", degree=exactdeg, A=A, mu_=mu_)

fP = Expression(("pi*pi*( (alpha_-4*G_-2*l_)*cos(pi*x[0])*cos(pi*x[1])-alpha_*sin(pi*x[0])*sin(pi*x[1]) )", \
                 "pi*pi*( alpha_*cos(pi*x[0])*cos(pi*x[1])-(alpha_-4*G_-2*l_)*sin(pi*x[0])*sin(pi*x[1]) )"), degree=exactdeg, alpha_=alpha_, G_=G_, l_=l_)
gP = Expression("(2*A+beta_)*pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", degree=exactdeg, A=A, beta_=beta_)
fS = Expression(("(1+4*mu_*A)*pi*pi*( cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]) )", \
                 "pi*pi*( cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]) )"), degree=exactdeg,  A=A, mu_=mu_)
gS = Constant(0.)
gNeuS = Expression(("0.0", \
                    "-pi*sin(pi*x[0])*(1)"), degree=exactdeg)
gNeuSTop = Expression(("0.0", \
                       "pi*sin(pi*x[0])*(-1)"), degree=exactdeg)
gNeuP = Expression(("0.0", \
                    "pi*(alpha_-2*G_-2*l_)*sin(pi*x[0])"), degree=exactdeg, alpha_=alpha_, G_=G_, l_=l_)

symgrad_dP_ex = Expression((("pi*(sin(pi*x[0])*cos(pi*x[1]))","pi*(cos(pi*x[0])*sin(pi*x[1]))"),\
                            ("pi*(cos(pi*x[0])*sin(pi*x[1]))","pi*(sin(pi*x[0])*cos(pi*x[1]))")), degree=exactdeg, A=A)

grad_pP_ex = Expression(("pi*pi*(cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]))",\
                         "pi*pi*(-sin(pi*x[0])*sin(pi*x[1])+cos(pi*x[0])*cos(pi*x[1]))"), degree=exactdeg, A=A)

symgrad_uS_ex = Expression((("-A*pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", "0.0"),\
                            ("0.0", "A*pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))")), degree=exactdeg, A=A)

# ******* Construct mesh and define normal, tangent ****** #

poroel = 10
stokes = 13
dirP = 14
dirS = 15
interf = 16
neuSTop = 20


# ******* Set subdomains, boundaries, and interface ****** #

mesh = RectangleMesh(Point(0.0, 1.0), Point(1.0, 2.0), 50, 50)
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 2.0) and on_boundary)

class SRight(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 1.0) and between(x[1], (1.0, 2.0)) and on_boundary)

class SLeft(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 0.0) and between(x[1], (1.0, 2.0)) and on_boundary)

class MStokes(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 1.

class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

def tensor_jump(u, n):
    return (outer(u('+'), n('+')) + outer(n('+'), u('+')))/2 + (outer(n('-'), u('-')) + outer(u('-'), n('-')))/2
def tensor_jump_b(u, n):
    return  outer(u, n)/2 + outer(n, u)/2

MStokes().mark(subdomains, stokes)
Interface().mark(boundaries, interf)

Top().mark(boundaries, neuSTop)
SRight().mark(boundaries, dirS)
SLeft().mark(boundaries, dirS)


n = FacetNormal(mesh)
t = as_vector((-n[1], n[0]))

# ******* Set subdomains, boundaries, and interface ****** #

OmS = MeshRestriction(mesh, MStokes())
Sig = MeshRestriction(mesh, Interface())

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS = Measure("dS", domain=mesh, subdomain_data=boundaries)

# verifying the domain size
areaS = assemble(1.*dx(stokes))
print("area(Omega_S) = ", areaS)

# ***** Global FE spaces and their restrictions ****** #

P2v = VectorFunctionSpace(mesh, "DG", deg)
P1 = FunctionSpace(mesh, "DG", degP)

#                         uS, pS
Hh = BlockFunctionSpace([P2v, P1],
                        restrict=[OmS, OmS])

trial = BlockTrialFunction(Hh)
uS, pS = block_split(trial)
test = BlockTestFunction(Hh)
vS, qS = block_split(test)

print("DoFs = ", Hh.dim(), " -- DoFs with unified Taylor-Hood = ", P2v.dim() + P1.dim())


# ******** Other parameters and BCs ************* #

# no BCs: imposed by DG

# ********  Define weak forms ********** #

h = CellDiameter(mesh)
h_avg = (2*h('+')*h('-'))/(h('+')+h('-'))
n = FacetNormal(mesh)
h_avg_S = (h('+')+h('-')-abs(h('+')-h('-')))/2

AS = 2.0 * mu * inner(sym(grad(uS)), sym(grad(vS))) * dx(stokes) \
     + (mu*etaU*deg*deg/h_avg*inner(tensor_jump(uS,n),tensor_jump(vS,n))*dS(0)) \
     - (2*mu*inner(avg(sym(grad(uS))), tensor_jump(vS,n))*dS(0)) - (2*mu*inner(avg(sym(grad(vS))), tensor_jump(uS,n))*dS(0)) \
     + (mu*etaU*deg*deg/h*inner(tensor_jump_b(uS,n), tensor_jump_b(vS,n))*ds(dirS)) \
     - (2*mu*inner(sym(grad(uS)), tensor_jump_b(vS,n))*ds(dirS)) \
     - (2*mu*inner(sym(grad(vS)), tensor_jump_b(uS,n))*ds(dirS))

B1St = - pS * div(vS) * dx(stokes) \
       + jump(vS,n) * avg(pS) * dS(0) \
       + inner(vS,n) * pS * ds(dirS)

B1S = qS * div(uS) * dx(stokes) \
       - jump(uS,n) * avg(qS) * dS(0) \
       - inner(uS,n) * qS * ds(dirS)

SS = eta*h_avg_S/degP * inner(jump(pS,n), jump(qS,n)) * dS(0)

FuS = dot(fS, vS) * dx(stokes) \
      + (mu*etaU*deg*deg/h*inner(tensor_jump_b(uS_ex,n),tensor_jump_b(vS,n))*ds(dirS)) \
      - (2*mu*inner(sym(grad(vS)), tensor_jump_b(uS_ex,n))*ds(dirS)) \
      + inner(gNeuS, vS) * ds(interf) \
      + inner(gNeuSTop, vS) * ds(neuSTop)

GqS = gS*qS * dx(stokes) \
      - inner(uS_ex,n) * qS * ds(dirS)

# ****** Assembly and solution of linear system ******** #

rhs = [FuS, GqS]

# this can be ordered arbitrarily. I've chosen
#        uS   pS
lhs = [[ AS, B1St],
       [B1S,   SS]]

AA = block_assemble(lhs)
FF = block_assemble(rhs)
# bcs.apply(AA)
# bcs.apply(FF)

sol = BlockFunction(Hh)
block_solve(AA, sol.block_vector(), FF, "mumps")
uS_h, pS_h = block_split(sol)
print(uS_h.vector().norm("l2"), [1709.466 if deg==2 else 1209.399])
print(pS_h.vector().norm("l2"), [5648.087 if deg==2 else 5651.009])
assert isclose(uS_h.vector().norm("l2"), [1709.466 if deg==2 else 1209.399])
assert isclose(pS_h.vector().norm("l2"), [5648.087 if deg==2 else 5651.009])

# ****** Saving data ******** #

uS_h.rename("uS", "uS")
pS_h.rename("pS", "pS")

output = XDMFFile(outputPath+'/'+outputFileBasename+".xdmf")
output.parameters["rewrite_function_mesh"] = False
output.parameters["functions_share_mesh"] = True
output.write(uS_h, 0.0)
output.write(pS_h, 0.0)
