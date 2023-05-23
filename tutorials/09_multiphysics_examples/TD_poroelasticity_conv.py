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
Time-dependent poroelasticity problem with 1 fluid compartment.
uP: Poroelastic displacement in H^1(OmP)
pP: Poroelastic pressure in L^2(OmP)
"""

def postprocess(sol, conv_idx, time_idx, outputPath, outputFileBasename, err_uP_L2, err_uP_H10, err_pP_L2, err_pP_H10):

    uP_h, pP_h = block_split(sol)
    # assert isclose(uP_h.vector().norm("l2"), 2.713143)
    # assert isclose(pP_h.vector().norm("l2"), 54.45552)
    print(uP_h.vector().norm("l2"), [86.59258])
    print(pP_h.vector().norm("l2"), [272.1546])

    # ****** Computing error ******** #

    form_err_uP_L2 = inner(uP_h - dP_ex, uP_h - dP_ex) * dx(poroel)
    form_err_uP_H10 = inner(sym(grad(uP_h)) - symgrad_dP_ex, sym(grad(uP_h)) - symgrad_dP_ex) * dx(poroel)
    form_err_pP_L2 = (pP_h - pP_ex)*(pP_h - pP_ex) * dx(poroel)
    form_err_pP_H10 = inner(grad(pP_h) - grad_pP_ex, grad(pP_h) - grad_pP_ex) * dx(poroel)
    err_uP_L2 = max(err_uP_L2, sqrt(assemble(form_err_uP_L2)))
    err_uP_H10 = max(err_uP_H10, sqrt(assemble(form_err_uP_H10)))
    err_pP_L2 = max(err_pP_L2, sqrt(assemble(form_err_pP_L2)))
    err_pP_H10 = max(err_pP_H10, sqrt(assemble(form_err_pP_H10)))
    print("Errors: uL2: ", err_uP_L2, "  uH10: ", err_uP_H10, "  pL2: ", err_pP_L2, "  pH10: ", err_pP_H10)

    # ****** Saving data ******** #

    uP_h.rename("uP", "uP")
    pP_h.rename("pP", "pP")

    output = XDMFFile(outputPath+'/'+outputFileBasename+"_"+str(conv_idx)+"."+str(time_idx)+".xdmf")
    output.parameters["rewrite_function_mesh"] = False
    output.parameters["functions_share_mesh"] = True
    output.write(uP_h, time)
    output.write(pP_h, time)
    output.close()

    with open(outputPath+'/'+outputFileBasename+'.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([1.0/N, time, err_uP_L2, err_uP_H10, err_pP_L2, err_pP_H10])

parameters["ghost_mode"] = "shared_facet"  # required by dS

# ********* I/O parameters  ******* #

outputPath = "output"
outputFileBasename = "TD_poroelasticity_conv"

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

rhoP = 0
cP = 0
rhoS = 0

t0 = 0
tmax = 1e-4

# ********* Numerical method constants  ******* #

deg = 1
degP = 1
etaU = Constant(10)
eta = Constant(10)

dt = 1e-5
newmarkBeta = Constant(0.25)
newmarkGamma = Constant(0.5)
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
    "-cos(pi*x[0])*cos(pi*x[1])", \
    "sin(pi*x[0])*sin(pi*x[1])", \
    "pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))" \
    ), degree=exactdeg)

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
    csvwriter.writerow(["h", "time", "err_u_L2", "err_u_H10", "err_p_L2", "err_p_H10"])

for ii in range(1,5):

    N = 5*(2**ii)
    print("\n##### CONV ITER ", ii, ": h = ", 1.0/N,"\n")

    # ******* Set subdomains, boundaries, and interface ****** #

    mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), N, N)
    subdomains = MeshFunction("size_t", mesh, 2)
    subdomains.set_all(0)
    boundaries = MeshFunction("size_t", mesh, 1)
    boundaries.set_all(0)

    class Bot(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[1], 0.0) and on_boundary)

    class PRight(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0], 1.0) and between(x[1], (0.0, 1.0)) and on_boundary)

    class PLeft(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0], 0.0) and between(x[1], (0.0, 1.0)) and on_boundary)
    
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
    Interface().mark(boundaries, interf)

    Bot().mark(boundaries, dirP)
    PRight().mark(boundaries, dirP)
    PLeft().mark(boundaries, dirP)

    n = FacetNormal(mesh)
    t = as_vector((-n[1], n[0]))

    # ******* Set subdomains, boundaries, and interface ****** #

    OmP = MeshRestriction(mesh, MPoroel())
    Sig = MeshRestriction(mesh, Interface())

    dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dS = Measure("dS", domain=mesh, subdomain_data=boundaries)

    # verifying the domain size
    areaP = assemble(1.*dx(poroel))
    print("area(Omega_P) = ", areaP)

    # ***** Global FE spaces and their restrictions ****** #

    BDM1 = VectorFunctionSpace(mesh, "DG", deg)
    P0 = FunctionSpace(mesh, "DG", degP)

    #                          uP, pP
    Hh = BlockFunctionSpace([BDM1, P0],
                            restrict=[OmP, OmP])

    trial = BlockTrialFunction(Hh)
    uP, pP = block_split(trial)
    test = BlockTestFunction(Hh)
    vP, qP = block_split(test)
    sol_OLD = BlockFunction(Hh)
    uP_OLD, pP_OLD = block_split(sol_OLD)
    to_zP_OLD = BlockFunction(Hh)
    zP_OLD, tmp = block_split(to_zP_OLD)
    to_aP_OLD = BlockFunction(Hh)
    aP_OLD, tmp = block_split(to_aP_OLD)
    to_aP_OLD2 = BlockFunction(Hh)
    aP_OLD2, tmp = block_split(to_aP_OLD2)

    print("DoFs = ", Hh.dim(), " -- DoFs with unified Taylor-Hood = ", BDM1.dim() + P0.dim())


    # ******** Other parameters and BCs ************* #

    # no BCs: imposed by DG

    # ********  Define weak forms ********** #

    h = CellDiameter(mesh)
    h_avg = (2*h('+')*h('-'))/(h('+')+h('-'))
    n = FacetNormal(mesh)
    h_avg_S = (h('+')+h('-')-abs(h('+')-h('-')))/2


    AP = rhoP/(newmarkBeta*dt*dt) * inner(uP,vP) * dx(poroel) \
         + (2*G*inner(sym(grad(uP)),sym(grad(vP)))*dx(poroel)) + (l*div(uP)*div(vP)*dx(poroel)) \
         + ((2*l+5*G)*etaU*deg*deg/h_avg*inner(tensor_jump(uP,n),tensor_jump(vP,n))*dS(0)) \
         - (2*G*inner(avg(sym(grad(uP))), tensor_jump(vP,n))*dS(0)) - (2*G*inner(avg(sym(grad(vP))), tensor_jump(uP,n))*dS(0)) \
         - (l*avg(div(uP))*jump(vP,n)*dS(0)) - (l*avg(div(vP))*jump(uP,n)*dS(0)) \
         + ((2*l+5*G)*etaU*deg*deg/h*inner(tensor_jump_b(uP,n), tensor_jump_b(vP,n))*ds(dirP)) \
         - (2*G*inner(sym(grad(uP)), tensor_jump_b(vP,n)) * ds(dirP)) \
         - (2*G*inner(sym(grad(vP)), tensor_jump_b(uP,n)) * ds(dirP)) \
         - (l*div(uP)*inner(vP,n)*ds(dirP)) - (l*div(vP)*inner(uP,n)*ds(dirP))

    SP = cP/dt * pP * qP * dx(poroel) \
         + theta*( \
         (Kval/G*inner(grad(pP),grad(qP))*dx(poroel)) \
         + beta*pP*qP*dx(poroel) \
         + (KvalCorr/G*eta*degP*degP/h_avg_S*inner(jump(pP,n),jump(qP,n))*dS(0)) \
         - (Kval/G*inner(avg(grad(pP)),jump(qP,n))*dS(0)) - (Kval/G*inner(avg(grad(qP)),jump(pP,n))*dS(0)) \
         + (KvalCorr/G*eta*degP*degP/h*pP*qP*ds(dirP)) \
         - (Kval/G*inner(grad(pP),n)*qP*ds(dirP)) - (Kval/G*inner(grad(qP),n)*pP*ds(dirP)) \
         + (KvalCorr/G*eta*degP*degP/h*pP*qP*ds(interf)) \
         - (Kval/G*inner(grad(pP),n)*qP*ds(interf)) - (Kval/G*inner(grad(qP),n)*pP*ds(interf)) \
         )

    B1Pt = - alpha * pP * div(vP) * dx(poroel) \
           + alpha * jump(vP,n) * avg(pP) * dS(0) \
           + alpha * inner(vP,n) * pP * ds(dirP)
    
    # NB In the STEADY case, the coupling pP->uP is one-directional
    B1P = 0
    # B1P = theta*newmarkGamma/(newmarkBeta*dt) * ( \
    #       alpha * qP * div(uP) * dx(poroel) \
    #       - alpha * jump(uP,n) * avg(qP) * dS(0) \
    #       - alpha * inner(uP,n) * qP * ds(dirP) \
    #       )

    FuP = rhoP/(newmarkBeta*dt*dt) * inner(uP_OLD,vP) * dx(poroel) \
          + rhoP/(newmarkBeta*dt) * inner(zP_OLD,vP) * dx(poroel) \
          + rhoP*(1.0-2*newmarkBeta)/(2*newmarkBeta) * inner(aP_OLD,vP) * dx(poroel) \
          + dot(fP, vP) * dx(poroel) \
          + (2*l+5*G)*etaU*deg*deg/h*inner(tensor_jump_b(dP_ex,n),tensor_jump_b(vP,n))*ds(dirP) \
          - 2*G*inner(sym(grad(vP)), tensor_jump_b(dP_ex,n))*ds(dirP) \
          - l*div(vP)*inner(dP_ex,n)*ds(dirP) \
          + inner(gNeuP, vP) * ds(interf)
    
    GqP = cP/dt * pP_OLD * qP * dx(poroel) \
          - (1-theta)*( \
          (Kval/G*inner(grad(pP_OLD),grad(qP))*dx(poroel)) \
          + beta*pP_OLD*qP*dx(poroel) \
          + (KvalCorr/G*eta*degP*degP/h_avg_S*inner(jump(pP_OLD,n),jump(qP,n))*dS(0)) \
          - (Kval/G*inner(avg(grad(pP_OLD)),jump(qP,n))*dS(0)) - (Kval/G*inner(avg(grad(qP)),jump(pP_OLD,n))*dS(0)) \
          + (KvalCorr/G*eta*degP*degP/h*pP_OLD*qP*ds(dirP)) \
          - (Kval/G*inner(grad(pP_OLD),n)*qP*ds(dirP)) - (Kval/G*inner(grad(qP),n)*pP_OLD*ds(dirP)) \
          + (KvalCorr/G*eta*degP*degP/h*pP_OLD*qP*ds(interf)) \
          - (Kval/G*inner(grad(pP_OLD),n)*qP*ds(interf)) - (Kval/G*inner(grad(qP),n)*pP_OLD*ds(interf)) \
          ) \
          + theta*( \
          gP*qP * dx(poroel) \
          + (KvalCorr/G*eta*degP*degP/h*pP_ex*qP*ds(dirP)) \
          - (Kval/G*pP_ex*inner(grad(qP),n)*ds(dirP)) \
          + (KvalCorr/G*eta*degP*degP/h*pP_ex*qP*ds(interf)) \
          - (Kval/G*pP_ex*inner(grad(qP),n)*ds(interf)) \
          ) \
          + (1-theta)*( \
          gP_OLD*qP * dx(poroel) \
          + (KvalCorr/G*eta*degP*degP/h*pP_ex_OLD*qP*ds(dirP)) \
          + (KvalCorr/G*eta*degP*degP/h*pP_ex_OLD*qP*ds(interf)) \
          - (Kval/G*pP_ex_OLD*inner(grad(qP),n)*ds(dirP)) \
          - (Kval/G*pP_ex_OLD*inner(grad(qP),n)*ds(interf)) \
          ) \
          - theta*newmarkGamma/(newmarkBeta*dt) * ( \
          alpha * qP * div(uP_OLD) * dx(poroel) \
          + alpha * jump(uP_OLD,n) * avg(qP) * dS(0) \
          + alpha * inner(uP_OLD,n) * qP * ds(dirP) \
          + alpha * inner(uP_OLD,n) * qP * ds(interf) \
          ) \
          - (theta*newmarkGamma/newmarkBeta - 1) * ( \
          alpha * qP * div(zP_OLD) * dx(poroel) \
          + alpha * jump(zP_OLD,n) * avg(qP) * dS(0) \
          + alpha * inner(zP_OLD,n) * qP * ds(dirP) \
          + alpha * inner(zP_OLD,n) * qP * ds(interf) \
          ) \
          - theta*(newmarkGamma/(2*newmarkBeta) - 1)*dt * ( \
          alpha * qP * div(aP_OLD) * dx(poroel) \
          + alpha * jump(aP_OLD,n) * avg(qP) * dS(0) \
          + alpha * inner(aP_OLD,n) * qP * ds(dirP) \
          + alpha * inner(aP_OLD,n) * qP * ds(interf) \
          )
    
    # ****** Initial conditions ******** #

    uP_OLD = interpolate(dP_ex, Hh.sub(0))
    pP_OLD = interpolate(pP_ex, Hh.sub(1))
    sol_OLD = BlockFunction(Hh, [uP_OLD, pP_OLD])
    zeros = Expression(('0.0','0.0'), degree=1)
    aP_OLD2 = interpolate(zeros, Hh.sub(0))
    aP_OLD = interpolate(zeros, Hh.sub(0))
    zP_OLD = interpolate(zeros, Hh.sub(0))

    # ****** Loop for time advancement ******** #

    err_uP_L2 = 0
    err_uP_H10 = 0
    err_pP_L2 = 0
    err_pP_H10 = 0

    time_N = ceil((tmax-t0)/dt)
    time = 0

    postprocess(sol_OLD, ii, 0, outputPath, outputFileBasename+"_allTimes", err_uP_L2, err_uP_H10, err_pP_L2, err_pP_H10)

    for time_idx in range(1, time_N):

        time = time + dt
        print("Time t = ", time)

        # ****** Assembly and solution of linear system ******** #

        rhs = [FuP, GqP]

        # this can be ordered arbitrarily. I've chosen
        #        uP   pP
        lhs = [[ AP, B1Pt],
               [B1P,   SP]]

        AA = block_assemble(lhs)
        FF = block_assemble(rhs)
        # bcs.apply(AA)
        # bcs.apply(FF)

        sol = BlockFunction(Hh)
        block_solve(AA, sol.block_vector(), FF, "mumps")
        uP_h, pP_h = block_split(sol)
        print(uP_h.vector().norm("l2"), [122.4312])
        print(pP_h.vector().norm("l2"), [272.1546])

        # ****** Update OLD variables ******** #

        aP_OLD2 = aP_OLD
        aP_OLD = (uP-uP_OLD)/(newmarkBeta*dt*dt) \
                 - zP_OLD/(newmarkBeta*dt) \
                 + (2*newmarkBeta-1)/(2*newmarkBeta) * aP_OLD2
        zP_OLD = zP_OLD + dt*(newmarkGamma*aP_OLD + (1-newmarkGamma)*aP_OLD2)
        block_assign(sol_OLD, sol)
        uP_OLD, pP_OLD = block_split(sol_OLD)

        # ****** postprocess ******** #

        postprocess(sol_OLD, ii, time_idx, outputPath, outputFileBasename+"_allTimes", err_uP_L2, err_uP_H10, err_pP_L2, err_pP_H10)

    # Re-exporting only last time in different convergence file.
    postprocess(sol_OLD, ii, time_idx, outputPath, outputFileBasename, err_uP_L2, err_uP_H10, err_pP_L2, err_pP_H10)
