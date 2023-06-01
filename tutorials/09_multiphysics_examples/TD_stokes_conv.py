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
from math import ceil
from dolfin import *
from multiphenics import *

import csv
import os

"""
Time-dependent Stokes problem.
uS: Stokes velocity in H^1(OmS)
pS: Stokes pressure in L^2(OmS)
"""

def postprocess(sol, conv_idx, time_idx, outputPath, outputFileBasename, err_uS_L2, err_uS_H10, err_pS_L2, sol_diff = None):

    uS_h, pS_h = block_split(sol)
    print(uS_h.vector().norm("l2"), [1709.466 if deg==2 else 1209.399])
    print(pS_h.vector().norm("l2"), [5648.087 if deg==2 else 5651.009])

    # ****** Computing error ******** #

    form_err_uS_L2 = inner(uS_h - uS_ex, uS_h - uS_ex) * dx(stokes)
    form_err_uS_H10 = inner(sym(grad(uS_h)) - symgrad_uS_ex, sym(grad(uS_h)) - symgrad_uS_ex) * dx(stokes)
    form_err_pS_L2 = (pS_h - pS_ex)*(pS_h - pS_ex) * dx(stokes)
    err_uS_L2 = max(err_uS_L2, sqrt(assemble(form_err_uS_L2)))
    err_uS_H10 = max(err_uS_H10, sqrt(assemble(form_err_uS_H10)))
    err_pS_L2 = max(err_pS_L2, sqrt(assemble(form_err_pS_L2)))
    print("Errors: uL2: ", err_uS_L2, "  uH10: ", err_uS_H10, "  pL2: ", err_pS_L2)

    # ****** Saving data ******** #

    with open(outputPath+'/'+outputFileBasename+'.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([1.0/N, time, err_uS_L2, err_uS_H10, err_pS_L2])

    output = XDMFFile(outputPath+'/'+outputFileBasename+"_"+str(conv_idx)+"."+str(time_idx)+".xdmf")
    output.parameters["rewrite_function_mesh"] = False
    output.parameters["functions_share_mesh"] = True

    uS_h.rename("uS", "uS")
    pS_h.rename("pS", "pS")
    output.write(uS_h, time)
    output.write(pS_h, time)

    if (sol_diff is not None):
        uS_diff, pS_diff = block_split(sol_diff)
        uS_diff.rename("uS_diff", "uS_diff")
        pS_diff.rename("pS_diff", "pS_diff")
        output.write(uS_diff, time)
        output.write(pS_diff, time)

    output.close()

# ********************************* #
# ************** MAIN ************* #
# ********************************* #

parameters["ghost_mode"] = "shared_facet"  # required by dS

# ********* I/O parameters  ******* #

outputPath = "output"
outputFileBasename = "TD_stokes_conv"

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

rhoP = 1
cP = 1
rhoS = 1

t0 = 0
tmax = 1e-4

# ********* Numerical method constants  ******* #

deg = 1
degP = 1
etaU = Constant(10)
eta = Constant(10)

dt = 1e-5
newmarkBeta_ = 0.25
newmarkBeta = Constant(newmarkBeta_)
newmarkGamma_ = 0.5
newmarkGamma = Constant(newmarkGamma_)
theta = Constant(1.0)

# ******* Exact solution, initial condition, and sources ****** #

exactdeg = 7

A = pi*pi*k/G_
dP_ex = Expression(("-cos(pi*x[0])*cos(pi*x[1])",\
                    "sin(pi*x[0])*sin(pi*x[1])"), degree=exactdeg)
dP_ex_OLD = dP_ex
pP_ex = Expression("pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", degree=exactdeg)
pP_ex_OLD = pP_ex
uS_ex = Expression(("A*(cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]))",\
                    "-A*(cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]))"), degree=exactdeg, A=A)
uS_ex_OLD = uS_ex
pS_ex = Expression("(1+2*mu_*A)*pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", degree=exactdeg, A=A, mu_=mu_)
pS_ex_OLD = pS_ex

initialCondition = Expression(( \
    "A*(cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]))", \
    "A*(cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]))", \
    "(1+2*mu_*A)*pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", \
    ), degree=exactdeg, A=A, mu_=mu_)

fP = Expression(("pi*pi*( (alpha_-4*G_-2*l_)*cos(pi*x[0])*cos(pi*x[1])-alpha_*sin(pi*x[0])*sin(pi*x[1]) )", \
                 "pi*pi*( alpha_*cos(pi*x[0])*cos(pi*x[1])-(alpha_-4*G_-2*l_)*sin(pi*x[0])*sin(pi*x[1]) )"), degree=exactdeg, alpha_=alpha_, G_=G_, l_=l_)
fP_OLD = fP
gP = Expression("(2*A+beta_)*pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", degree=exactdeg, A=A, beta_=beta_)
gP_OLD = gP
fS = Expression(("(1+4*mu_*A)*pi*pi*( cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]) )", \
                 "pi*pi*( cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]) )"), degree=exactdeg,  A=A, mu_=mu_)
fS_OLD = fS
gS = Constant(0.)
gS_OLD = gS
gNeuS = Expression(("0.0", \
                    "-pi*sin(pi*x[0])*(1)"), degree=exactdeg)
gNeuS_OLD = gNeuS
gNeuSTop = Expression(("0.0", \
                       "pi*sin(pi*x[0])*(-1)"), degree=exactdeg)
gNeuSTop_OLD = gNeuSTop
gNeuP = Expression(("0.0", \
                    "pi*(alpha_-2*G_-2*l_)*sin(pi*x[0])"), degree=exactdeg, alpha_=alpha_, G_=G_, l_=l_)
gNeuP_OLD = gNeuP

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

# ******* Loop for h-convergence ****** #

with open(outputPath+'/'+outputFileBasename+'.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                           quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(["h", "time", "err_u_L2", "err_u_H10", "err_p_L2"])

for ii in range(1,6):

    N = 5*(2**ii)
    print("\n##### CONV ITER ", ii, ": h = ", 1.0/N,"\n")

    # ******* Set subdomains, boundaries, and interface ****** #

    mesh = RectangleMesh(Point(0.0, 1.0), Point(1.0, 2.0), N, N)
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
    sol_OLD = BlockFunction(Hh)
    uS_OLD, pS_OLD = block_split(sol_OLD)

    print("DoFs = ", Hh.dim(), " -- DoFs with unified Taylor-Hood = ", P2v.dim() + P1.dim())

    # ******** Other parameters and BCs ************* #

    h = CellDiameter(mesh)
    h_avg = (2*h('+')*h('-'))/(h('+')+h('-'))
    n = FacetNormal(mesh)
    h_avg_S = (h('+')+h('-')-abs(h('+')-h('-')))/2

    # no BCs: imposed by DG

    # ****** Initial conditions ******** #

    uS_OLD = interpolate(uS_ex, Hh.sub(0))
    pS_OLD = interpolate(pS_ex, Hh.sub(1))
    sol_OLD = BlockFunction(Hh, [uS_OLD, pS_OLD])

    # ****** Loop for time advancement ******** #

    err_uS_L2 = 0
    err_uS_H10 = 0
    err_pS_L2 = 0

    time_N = ceil((tmax-t0)/dt)
    time = 0

    postprocess(sol_OLD, ii, 0, outputPath, outputFileBasename+"_allTimes", err_uS_L2, err_uS_H10, err_pS_L2)

    sol_diff = BlockFunction(Hh)

    for time_idx in range(1, time_N):

        time = time + dt
        print("Time t = ", time)

        # ****** (Re-)define weak forms ******** #

        AS = rhoS/dt * inner(uS,vS) * dx(stokes) \
             + theta*(\
             2.0 * mu * inner(sym(grad(uS)), sym(grad(vS))) * dx(stokes) \
             + (mu*etaU*deg*deg/h_avg*inner(tensor_jump(uS,n),tensor_jump(vS,n))*dS(0)) \
             - (2*mu*inner(avg(sym(grad(uS))), tensor_jump(vS,n))*dS(0)) - (2*mu*inner(avg(sym(grad(vS))), tensor_jump(uS,n))*dS(0)) \
             + (mu*etaU*deg*deg/h*inner(tensor_jump_b(uS,n), tensor_jump_b(vS,n))*ds(dirS)) \
             - (2*mu*inner(sym(grad(uS)), tensor_jump_b(vS,n))*ds(dirS)) \
             - (2*mu*inner(sym(grad(vS)), tensor_jump_b(uS,n))*ds(dirS)) \
             )

        B1St = theta*(\
               - pS * div(vS) * dx(stokes) \
               + jump(vS,n) * avg(pS) * dS(0) \
               + inner(vS,n) * pS * ds(dirS) \
               )

        B1S = theta*(\
               qS * div(uS) * dx(stokes) \
               - jump(uS,n) * avg(qS) * dS(0) \
               - inner(uS,n) * qS * ds(dirS) \
               )

        SS = eta*h_avg_S/degP * inner(jump(pS,n), jump(qS,n)) * dS(0)

        FuS = rhoS/dt * inner(uS_OLD,vS) * dx(stokes) \
              - (1-theta)*( \
              2.0 * mu * inner(sym(grad(uS_OLD)), sym(grad(vS))) * dx(stokes) \
              + (mu*etaU*deg*deg/h_avg*inner(tensor_jump(uS_OLD,n),tensor_jump(vS,n))*dS(0)) \
              - (2*mu*inner(avg(sym(grad(uS_OLD))), tensor_jump(vS,n))*dS(0)) - (2*mu*inner(avg(sym(grad(vS))), tensor_jump(uS_OLD,n))*dS(0)) \
              + (mu*etaU*deg*deg/h*inner(tensor_jump_b(uS_OLD,n), tensor_jump_b(vS,n))*ds(dirS)) \
              - (2*mu*inner(sym(grad(uS_OLD)), tensor_jump_b(vS,n))*ds(dirS)) \
              - (2*mu*inner(sym(grad(vS)), tensor_jump_b(uS_OLD,n))*ds(dirS)) \
              ) \
              - (1-theta)*( \
              -pS_OLD * div(vS) * dx(stokes) \
              + jump(vS,n) * avg(pS_OLD) * dS(0) \
              + inner(vS,n) * pS_OLD * ds(dirS) \
              ) \
              + theta * ( \
              dot(fS, vS) * dx(stokes) \
              + (mu*etaU*deg*deg/h*inner(tensor_jump_b(uS_ex,n),tensor_jump_b(vS,n))*ds(dirS)) \
              - (2*mu*inner(sym(grad(vS)), tensor_jump_b(uS_ex,n))*ds(dirS)) \
              + inner(gNeuS, vS) * ds(interf) \
              + inner(gNeuSTop, vS) * ds(neuSTop) \
              ) \
              + (1-theta) * ( \
              + dot(fS_OLD, vS) * dx(stokes) \
              + (mu*etaU*deg*deg/h*inner(tensor_jump_b(uS_ex_OLD,n),tensor_jump_b(vS,n))*ds(dirS)) \
              - (2*mu*inner(sym(grad(vS)), tensor_jump_b(uS_ex_OLD,n))*ds(dirS)) \
              + inner(gNeuS_OLD, vS) * ds(interf) \
              + inner(gNeuSTop_OLD, vS) * ds(neuSTop) \
              )

        GqS = -(1-theta)*(
              qS * div(uS_OLD) * dx(stokes) \
              - jump(uS_OLD,n) * avg(qS) * dS(0) \
              - inner(uS_ex_OLD,n) * qS * ds(dirS) \
              + eta*h_avg_S/degP * inner(jump(pS_OLD,n),jump(qS,n)) * dS(0) \
              ) \
              + theta*( \
              gS*qS * dx(stokes) \
              - inner(uS_ex,n) * qS * ds(dirS) \
              ) \
              + (1-theta)*( \
              gS_OLD*qS * dx(stokes) \
              - inner(uS_ex_OLD,n) * qS * ds(dirS) \
              )

        rhs = [FuS, GqS]

        # this can be ordered arbitrarily. I've chosen
        #        uS   pS
        lhs = [[ AS, B1St],
               [B1S,   SS]]

        # ****** Assembly and solution of linear system ******** #

        AA = block_assemble(lhs)
        FF = block_assemble(rhs)
        # bcs.apply(AA)
        # bcs.apply(FF)

        sol = BlockFunction(Hh)
        block_solve(AA, sol.block_vector(), FF, "mumps")
        uS_h, pS_h = block_split(sol)
        print(uS_h.vector().norm("l2"), [122.4312])
        print(pS_h.vector().norm("l2"), [272.1546])

        # ****** Update OLD variables ******** #

        sol_diff = BlockFunction(Hh)
        sol_diff = (sol-sol_OLD).copy(deepcopy=True)

        block_assign(sol_OLD, sol)
        uS_OLD, pS_OLD = (subf.copy(deepcopy=True) for subf in block_split(sol_OLD))

        # ****** postprocess ******** #

        postprocess(sol_OLD, ii, time_idx, outputPath, outputFileBasename+"_allTimes", err_uS_L2, err_uS_H10, err_pS_L2, sol_diff)

    # Re-exporting only last time in different convergence file.
    postprocess(sol_OLD, ii, time_idx, outputPath, outputFileBasename, err_uS_L2, err_uS_H10, err_pS_L2)
