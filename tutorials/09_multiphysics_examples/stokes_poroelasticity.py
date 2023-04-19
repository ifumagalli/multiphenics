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

uS.nS = (K/G)grad(pP).nP
-(2*mu*eps(uS)-pS*I)*nS.nS = pP
-(2*mu*eps(uS)-pS*I)*tS.nS = 0
-(2*G*eps(uP)+l*div(uP)*I + alpha*p*I)*nP -(2*mu*eps(uS)-pS*I)*nS = 0
"""

parameters["ghost_mode"] = "shared_facet"  # required by dS

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

mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 2.0), 50, 100)
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

class PRight(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 1.0) and between(x[1], (0.0, 1.0)) and on_boundary)

class PLeft(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 0.0) and between(x[1], (0.0, 1.0)) and on_boundary)

class Bot(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) and on_boundary)

class MStokes(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 1.

class MPoroel(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= 1.

class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

def tensor_jump(u, n):
    return (outer(u('+'), n('+')) + outer(n('+'), u('+')))/2 + (outer(n('-'), u('-')) + outer(u('-'), n('-')))/2
def tensor_jump_b(u, n):
    return  outer(u, n)/2 + outer(n, u)/2

MPoroel().mark(subdomains, poroel)
MStokes().mark(subdomains, stokes)
Interface().mark(boundaries, interf)

Top().mark(boundaries, neuSTop)
SRight().mark(boundaries, dirS)
SLeft().mark(boundaries, dirS)
PRight().mark(boundaries, dirP)
PLeft().mark(boundaries, dirP)
Bot().mark(boundaries, dirP)

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

#                         uS,   uP, pS, pP
Hh = BlockFunctionSpace([P2v, BDM1, P1, P0],
                        restrict=[OmS, OmP, OmS, OmP])

trial = BlockTrialFunction(Hh)
uS, uP, pS, pP = block_split(trial)
test = BlockTestFunction(Hh)
vS, vP, qS, qP = block_split(test)

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

AP = (2*G*inner(sym(grad(uP)),sym(grad(vP)))*dx(poroel)) + (l*div(uP)*div(vP)*dx(poroel)) \
     + ((2*l+5*G)*etaU*deg*deg/h_avg*inner(tensor_jump(uP,n),tensor_jump(vP,n))*dS(0)) \
     - (2*G*inner(avg(sym(grad(uP))), tensor_jump(vP,n))*dS(0)) - (2*G*inner(avg(sym(grad(vP))), tensor_jump(uP,n))*dS(0)) \
     - (l*avg(div(uP))*jump(vP,n)*dS(0)) - (l*avg(div(vP))*jump(uP,n)*dS(0)) \
     + ((2*l+5*G)*etaU*deg*deg/h*inner(tensor_jump_b(uP,n), tensor_jump_b(vP,n))*ds(dirP)) \
     - (2*G*inner(sym(grad(uP)), tensor_jump_b(vP,n)) * ds(dirP)) \
     - (2*G*inner(sym(grad(vP)), tensor_jump_b(uP,n)) * ds(dirP)) \
     - (l*div(uP)*inner(vP,n)*ds(dirP)) - (l*div(vP)*inner(uP,n)*ds(dirP))

SP = (Kval/G*inner(grad(pP),grad(qP))*dx(poroel)) \
     + beta*pP*qP*dx(poroel) \
     + (KvalCorr/G*eta*degP*degP/h_avg_S*inner(jump(pP,n),jump(qP,n))*dS(0)) \
     - (Kval/G*inner(avg(grad(pP)),jump(qP,n))*dS(0)) - (Kval/G*inner(avg(grad(qP)),jump(pP,n))*dS(0)) \
     - (Kval/G*inner(grad(pP),n)*qP*ds(dirP)) - (Kval/G*inner(grad(qP),n)*pP*ds(dirP)) \
     + (KvalCorr/G*eta*degP*degP/h*pP*qP*ds(dirP)) \
     - (Kval/G*inner(grad(pP('+')),n('+'))*qP('+')*dS(interf)) - (Kval/G*inner(grad(qP('+')),n('+'))*pP('+')*dS(interf))

JSt = pP('+') * dot(vS('-'), n('-')) * dS(interf)
JS = qP('+') * dot(uS('-'), n('+')) * dS(interf)
#NO IN STEADY# JP = - qP('-') * dot(uP('-'), n('-')) * dS(interf)
JPt = pP('-') * dot(vP('-'), n('-')) * dS(interf)

B1Pt = - alpha * pP * div(vP) * dx(poroel) \
       + alpha * jump(vP,n) * avg(pP) * dS(0) \
       + alpha * inner(vP,n) * pP * ds(dirP) \
       + JPt

# NB In the STEADY case, the coupling pP->uP is one-directional
B1P = 0#NO IN STEADY# JP #\
      #NO IN STEADY# + alpha * qP * div(uP) * dx(poroel) \
      #NO IN STEADY# - alpha * jump(uP,n) * avg(qP) * dS(0) \
      #NO IN STEADY# - alpha * inner(uP,n) * qP * ds(dirP)

FuS = dot(fS, vS) * dx(stokes) \
      + (mu*etaU*deg*deg/h*inner(tensor_jump_b(uS_ex,n),tensor_jump_b(vS,n))*ds(dirS)) \
      - (2*mu*inner(sym(grad(vS)), tensor_jump_b(uS_ex,n))*ds(dirS)) \
      + inner(gNeuSTop, vS) * ds(neuSTop) # \
      #THIS SEEMS TO REPLACE JSt (and quite precisely, especially in terms of uS,pS,uP)#  + inner(gNeuS, vS('+')) * dS(interf)

FuP = dot(fP, vP) * dx(poroel) \
      + (2*l+5*G)*etaU*deg*deg/h*inner(tensor_jump_b(dP_ex,n),tensor_jump_b(vP,n))*ds(dirP) \
      - 2*G*inner(sym(grad(vP)), tensor_jump_b(dP_ex,n))*ds(dirP) \
      - l*div(vP)*inner(dP_ex,n)*ds(dirP)

GqS = gS*qS * dx(stokes) \
      - inner(uS_ex,n) * qS * ds(dirS)

GqP = gP*qP * dx(poroel) \
      + (KvalCorr/G*eta*degP*degP/h*pP_ex*qP*ds(dirP)) \
      - (Kval/G*pP_ex*inner(grad(qP),n)*ds(dirP)) \
      - (Kval/G*pP_ex*inner(grad(qP('+')),n('+'))*dS(interf)) #\
      #NO IN STEADY# - alpha * inner(dP_ex,n) * qP * ds(dirP)

# ****** Assembly and solution of linear system ******** #

rhs = [FuS, FuP, GqS, GqP]

# this can be ordered arbitrarily. I've chosen
#        uS   uP    pS    pP
lhs = [[ AS,   0, B1St,  JSt],
       [  0,  AP,    0, B1Pt],
       [B1S,   0,   SS,    0],
       [ JS, B1P,    0,   SP]]

AA = block_assemble(lhs)
FF = block_assemble(rhs)
# bcs.apply(AA)
# bcs.apply(FF)

sol = BlockFunction(Hh)
block_solve(AA, sol.block_vector(), FF, "mumps")
uS_h, uP_h, pS_h, pP_h = block_split(sol)
# assert isclose(uS_h.vector().norm("l2"), 73.54915)
# assert isclose(uP_h.vector().norm("l2"), 2.713143)
# assert isclose(pS_h.vector().norm("l2"), 175.4097)
# assert isclose(pP_h.vector().norm("l2"), 54.45552)
print(uS_h.vector().norm("l2"), [1709.466 if deg==2 else 1209.399])
print(pS_h.vector().norm("l2"), [5648.087 if deg==2 else 5651.009])
print(uP_h.vector().norm("l2"), [86.59258])
print(pP_h.vector().norm("l2"), [272.1546])

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
