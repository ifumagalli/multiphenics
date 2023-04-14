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

"""
Stokes-Poroelasticity equations
Coupled mixed formulation using Lagrange multiplier,
from Layton, Schieweck, Yotov, Coupling fluid flow with
     porous media flow. SINUM 2003, DOI:10.1137/S0036142901392766

defined on the interface

uS: Stokes velocity in H^1(OmS)
uP: Poroelastic displacement in H^1(OmP)
pS: Stokes pressure in L^2(OmS)
pP: Poroelastic pressure in L^2(OmP)

transmission conditions:

uS.nS + uD.nD = 0
-(2*mu*eps(uS)-pS*I)*nS.nS = pD
-(2*mu*eps(uS)-pS*I)*tS.nS = alpha*mu*k^-0.5*uS.tS
"""

parameters["ghost_mode"] = "shared_facet"  # required by dS

# ********* Model constants  ******* #

G = Constant(1.)
l = Constant(1.)
k = 1.
Kval = Constant(k)
KvalCorr = Constant(max(k, 1.))
mu = Constant(1.)
alpha = Constant(1.)
fS = Constant((0., 0.))
fP = fS
gS = Constant(0.)
gP = gS

# ********* Numerical method constants  ******* #

deg = 2
degP = 1
etaU = 10
eta = 10
tau = 1

# ******* Construct mesh and define normal, tangent ****** #
poroel = 10
stokes = 13
outlet = 14
inlet = 15
interf = 16
wallS = 17
wallP = 18

# ******* Set subdomains, boundaries, and interface ****** #

mesh = RectangleMesh(Point(-1.0, -2.0), Point(1.0, 2.0), 50, 100)
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 2.0) and on_boundary)

class SRight(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 1.0) and between(x[1], (0.0, 2.0)) and on_boundary)

class SLeft(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], -1.0) and between(x[1], (0.0, 2.0)) and on_boundary)

class PRight(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 1.0) and between(x[1], (-2.0, 0.0)) and on_boundary)

class PLeft(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], -1.0) and between(x[1], (-2.0, 0.0)) and on_boundary)

class Bot(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], -2.0) and on_boundary)

class MStokes(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 0.

class MPoroel(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= 0.

class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

def tensor_jump(u, n):
    return (outer(u('+'), n('+')) + outer(n('+'), u('+')))/2 + (outer(n('-'), u('-')) + outer(u('-'), n('-')))/2
def tensor_jump_b(u, n):
    return  outer(u, n)/2 + outer(n, u)/2

MPoroel().mark(subdomains, poroel)
MStokes().mark(subdomains, stokes)
Interface().mark(boundaries, interf)

Top().mark(boundaries, inlet)
SRight().mark(boundaries, wallS)
SLeft().mark(boundaries, wallS)
PRight().mark(boundaries, wallP)
PLeft().mark(boundaries, wallP)
Bot().mark(boundaries, outlet)


n = FacetNormal(mesh)
t = as_vector((-n[1], n[0]))

# ******* Set subdomains, boundaries, and interface ****** #

OmS = MeshRestriction(mesh, MStokes())
OmP = MeshRestriction(mesh, MPoroel())
Sig = MeshRestriction(mesh, Interface())

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS = Measure("dS", domain=mesh, subdomain_data=boundaries)

# verifying the domain size
areaP = assemble(1.*dx(poroel))
areaS = assemble(1.*dx(stokes))
lengthI = assemble(1.*dS(interf))
print("area(Omega_P) = ", areaP)
print("area(Omega_S) = ", areaS)
print("length(Sigma) = ", lengthI)

# ***** Global FE spaces and their restrictions ****** #

P2v = VectorFunctionSpace(mesh, "DG", deg)
P1 = FunctionSpace(mesh, "DG", degP)
BDM1 = VectorFunctionSpace(mesh, "DG", deg)
P0 = FunctionSpace(mesh, "DG", degP)
Pt = FunctionSpace(mesh, "DGT", degP)

# the space for uD can be RT or BDM
# the space for lambda should be DGT, but then it cannot be
#     projected and saved to output file. If we want to see lambda
#     we need to use e.g. P1 instead (it cannot be DG), but this makes
#     that the interface extends to the neighbouring element in OmegaD


#                         uS,   uP, pS, pP, la
Hh = BlockFunctionSpace([P2v, BDM1, P1, P0, Pt],
                        restrict=[OmS, OmP, OmS, OmP, Sig])

trial = BlockTrialFunction(Hh)
uS, uP, pS, pP, la = block_split(trial)
test = BlockTestFunction(Hh)
vS, vP, qS, qP, xi = block_split(test)

print("DoFs = ", Hh.dim(), " -- DoFs with unified Taylor-Hood = ", P2v.dim() + P1.dim())


# ******** Other parameters and BCs ************* #

inflow = Expression(("0.0", "pow(x[0], 2)-1.0"), degree=2)
noSlip = Constant((0., 0.))
fakeSlip = Constant((0., 0.))
pOut = Constant(0.0)

# no BCs: imposed by DG

# ********  Define weak forms ********** #

h = CellDiameter(mesh)
h_avg = (2*h('+')*h('-'))/(h('+')+h('-'))
n = FacetNormal(mesh)
h_avg_S = (h('+')+h('-')-abs(h('+')-h('-')))/2

AS = 2.0 * mu * inner(sym(grad(uS)), sym(grad(vS))) * dx(stokes) \
     + (mu*etaU*deg*deg/h_avg*inner(tensor_jump(uS,n),tensor_jump(vS,n))*dS(0)) \
     - (2*mu*inner(avg(sym(grad(uS))), tensor_jump(vS,n))*dS(0)) - (2*mu*inner(avg(sym(grad(vS))), tensor_jump(uS,n))*dS(0)) \
     + (mu*etaU*deg*deg/h*inner(tensor_jump_b(uS,n), tensor_jump_b(vS,n))*ds(inlet)) \
     - (2*mu*inner(sym(grad(uS)), tensor_jump_b(vS,n))*ds(inlet)) \
     - (2*mu*inner(sym(grad(vS)), tensor_jump_b(uS,n))*ds(inlet)) \
     + (mu*etaU*deg*deg/h*inner(tensor_jump_b(uS,n), tensor_jump_b(vS,n))*ds(wallS)) \
     - (2*mu*inner(sym(grad(uS)), tensor_jump_b(vS,n))*ds(wallS)) \
     - (2*mu*inner(sym(grad(vS)), tensor_jump_b(uS,n))*ds(wallS))

B1St = - pS * div(vS) * dx(stokes) \
       + jump(vS,n) * avg(pS) * dS(0) \
       + inner(vS,n) * pS * ds(inlet) \
       + inner(vS,n) * pS * ds(wallS)

B1S = qS * div(uS) * dx(stokes) \
       - jump(uS,n) * avg(qS) * dS(0) \
       - inner(uS,n) * qS * ds(inlet) \
       - inner(uS,n) * qS * ds(wallS)

SS = eta*h_avg_S/degP * inner(jump(pS,n), jump(qS,n)) * dS(0)

AP = (2*G*inner(sym(grad(uP)),sym(grad(vP)))*dx(poroel)) + (l*div(uP)*div(vP)*dx(poroel)) \
     + ((2*l+5*G)*etaU*deg*deg/h_avg*inner(tensor_jump(uP,n),tensor_jump(vP,n))*dS(0)) \
     - (2*G*inner(avg(sym(grad(uP))), tensor_jump(vP,n))*dS(0)) - (2*G*inner(avg(sym(grad(vP))), tensor_jump(uP,n))*dS(0)) \
     - (l*avg(div(uP))*jump(vP,n)*dS(0)) - (l*avg(div(vP))*jump(uP,n)*dS(0)) \
     + ((2*l+5*G)*etaU*deg*deg/h*inner(tensor_jump_b(uP,n), tensor_jump_b(vP,n))*ds(wallP)) \
     - (2*G*inner(sym(grad(uP)), tensor_jump_b(vP,n)) * ds(wallP)) \
     - (2*G*inner(sym(grad(vP)), tensor_jump_b(uP,n)) * ds(wallP)) \
     - (l*div(uP)*inner(vP,n)*ds(wallP)) - (l*div(vP)*inner(uP,n)*ds(wallP))

SP = (Kval/G*inner(grad(pP),grad(qP))*dx(poroel)) \
     + (KvalCorr*eta*degP*degP/h_avg_S*inner(jump(pP,n),jump(qP,n))*dS(0)) \
     - (Kval/G*inner(avg(grad(pP)),jump(qP,n))*dS(0)) - (Kval/G*inner(avg(grad(qP)),jump(pP,n))*dS(0)) \
     - (Kval/G*inner(grad(pP),n)*qP*ds(outlet)) - (Kval/G*inner(grad(qP),n)*pP*ds(outlet)) \
     + (KvalCorr*eta*degP*degP/h*pP*qP*ds(outlet))

JSt = - avg(qP) * jump(uS, n) * dS(interf)
JS = avg(pP) * jump(vS, n) * dS(interf)
JP = avg(qP) * jump(uP, n) * dS(interf)
JPt = avg(pP) * jump(vP, n) * dS(interf)

B1P = qP * div(uP) * dx(poroel) \
      - alpha * jump(uP,n) * avg(qP) * dS(0) \
      - alpha * inner(uP,n) * qP * ds(wallP) \
      + JP

B1Pt = - alpha * pP * div(vP) * dx(poroel) \
       + alpha * jump(vP,n) * avg(pP) * dS(0) \
       + JPt

FuS = dot(fS, vS) * dx(stokes) \
      + (mu*etaU*deg*deg/h*inner(tensor_jump_b(inflow,n),tensor_jump_b(vS,n))*ds(inlet)) \
      - (2*mu*inner(sym(grad(vS)), tensor_jump_b(inflow,n))*ds(inlet)) \
      + (mu*etaU*deg*deg/h*inner(tensor_jump_b(noSlip,n),tensor_jump_b(vS,n))*ds(wallS)) \
      - (2*mu*inner(sym(grad(vS)), tensor_jump_b(noSlip,n))*ds(wallS))

FuP = dot(fP, vP) * dx(poroel) \
      + (2*l+5*G)*etaU*deg*deg/h*inner(tensor_jump_b(fakeSlip,n),tensor_jump_b(vP,n))*ds(wallP) \
      - 2*G*inner(sym(grad(vP)), tensor_jump_b(fakeSlip,n))*ds(wallP) \
      - l*div(vP)*dot(fakeSlip,n)*ds(wallP)

GqS = gS*qS * dx(stokes) \
      - inner(inflow,n) * qS * ds(inlet) \
      - inner(noSlip,n) * qS * ds(wallS)

GqP = - gP*qP * dx(poroel) \
      - alpha * inner(fakeSlip,n) * qP * ds(wallP) \
      + (KvalCorr*eta*degP*degP/h*pOut*qP*ds(outlet))

# ****** Assembly and solution of linear system ******** #

rhs = [FuS, FuP, GqS, GqP, 0]

# this can be ordered arbitrarily. I've chosen
#        uS   uP   pS   pP  la
lhs = [[ AS,   0, B1St,  JSt,    0],
       [  0,  AP,    0, B1Pt,    0],
       [B1S,   0,   SS,    0,    0],
       [ JS, B1P,    0,   SP,    0],
       [  0,   0,    0,    0,   SS]]

AA = block_assemble(lhs)
FF = block_assemble(rhs)
# bcs.apply(AA)
# bcs.apply(FF)

sol = BlockFunction(Hh)
block_solve(AA, sol.block_vector(), FF)#, "mumps")
uS_h, uP_h, pS_h, pP_h, la_h = block_split(sol)
# assert isclose(uS_h.vector().norm("l2"), 73.54915)
# assert isclose(uP_h.vector().norm("l2"), 2.713143)
# assert isclose(pS_h.vector().norm("l2"), 175.4097)
# assert isclose(pP_h.vector().norm("l2"), 54.45552)
print(uS_h.vector().norm("l2"), 73.54915)
print(uP_h.vector().norm("l2"), 2.713143)
print(pS_h.vector().norm("l2"), 175.4097)
print(pP_h.vector().norm("l2"), 54.45552)

# ****** Saving data ******** #
uS_h.rename("uS", "uS")
pS_h.rename("pS", "pS")
uP_h.rename("uP", "uP")
pP_h.rename("pP", "pP")

output = XDMFFile("stokes_poroelasticity.xdmf")
output.parameters["rewrite_function_mesh"] = False
output.parameters["functions_share_mesh"] = True
output.write(uS_h, 0.0)
output.write(pS_h, 0.0)
output.write(uP_h, 0.0)
output.write(pP_h, 0.0)
