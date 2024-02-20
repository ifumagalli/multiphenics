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
from numpy import asarray as numpy_asarray
from numpy import int32 as numpy_int32
from math import ceil
from dolfin import *
from multiphenics import *

from dolfin.cpp.mesh import MeshFunctionBool

import csv
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + '/utilities')

"""
Time-dependent Stokes-Poroelasticity problem.

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

def MeshImportFromFile(filename):

	if not (filename[len(filename)-3:len(filename)] == ".h5"):

		# Error if the extension of the file is not possible
		print("ERROR! The extension of the file cannot be used!")
		sys.exit(0)

	# Mesh initialization
	mesh = Mesh()

	# Mesh .h5 reading
	file = HDF5File(MPI.comm_world, filename, "r")
	file.read(mesh, "/mesh", False)

	file.close()

	if (MPI.comm_world.Get_rank() == 0):
		print("Mesh read from file!")

	return mesh

def ImportMeshFunctionsFromFile(filename, mesh):

    file = HDF5File(MPI.comm_world, filename, "r")

    # Detect mesh dimension
    D = mesh.topology().dim()

    # Definition the facet functions
    BoundaryID = MeshFunction("size_t", mesh, D-1)
    SubdomainID = MeshFunction("size_t", mesh, D)

    # Read the functions from the file
    name = "/boundaries"
    file.read(BoundaryID, name)
    name = "/subdomains"
    file.read(SubdomainID, name)

    file.close()

    # Create list of MeshFunctionBool identifying the restrictions
    # === NB === we assume that the values are specifically 1 and 2. Lines affected by these are marked by #@12#
    print("create restrictions ... ", flush=True)
    # restrictions = list()
    # for val in [1, 2]:   #@12#
    #     mfb = MeshFunctionBool(mesh, D)
    #     print(mfb.where_equal(True), flush=True)
    #     mfb.set_all(0)
    #     print("...",val,"...", flush=True)
    #     restrictions.append(list(mfb))
    tmp_restrictions = [MeshFunctionBool(mesh, D), MeshFunctionBool(mesh, D)] #@12#
    for r in tmp_restrictions:   #@12#
        print(r.where_equal(True), flush=True)
    print("... initialized ...", flush=True)
    print(SubdomainID.array(), tmp_restrictions, flush=True)
    # for cell in cells(mesh):
    #     restrictions[SubdomainID[cell]-1][0][cell] = True      #@12#
    tmp_restrictions[0].set_values([v == 1 for v in SubdomainID.array()])  #@12#
    tmp_restrictions[1].set_values([v == 2 for v in SubdomainID.array()])  #@12#
    restrictions = [tmp_restrictions] * 4
    print("... done", restrictions, flush=True)

    return BoundaryID, SubdomainID, restrictions

def postprocessXDMF(sol, conv_idx, time_idx, outputPath, outputFileBasename, subdomFEfun):
    
    uS_h, uP_h, pS_h, pP_h = block_split(sol)
    
    uS_h.rename("uS", "uS")
    pS_h.rename("pS", "pS")
    uP_h.rename("uP", "uP")
    pP_h.rename("pP", "pP")

    output = XDMFFile(outputPath+'/'+outputFileBasename+"_"+str(conv_idx)+"."+str(time_idx)+".xdmf")
    output.parameters["rewrite_function_mesh"] = False
    output.parameters["functions_share_mesh"] = True
    output.write(uS_h, time)
    output.write(pS_h, time)
    output.write(uP_h, time)
    output.write(pP_h, time)

    output.write(subdomFEfun, time) # time is necessary to avoid "Error: Unable to write Function to XDMF. Reason: Not writing a time series."
    output.close()

def postprocessCSV(sol, conv_idx, time_idx, outputPath, outputFileBasename, err_uS_L2, err_uS_H10, err_pS_L2, err_uP_L2, err_uP_H10, err_pP_L2, err_pP_H10, err_energy_2, h_avg, h_avg_S):

    uS_h, uP_h, pS_h, pP_h = block_split(sol)
    # assert isclose(uS_h.vector().norm("l2"), 73.54915)
    # assert isclose(uP_h.vector().norm("l2"), 2.713143)
    # assert isclose(pS_h.vector().norm("l2"), 175.4097)
    # assert isclose(pP_h.vector().norm("l2"), 54.45552)
    print(uS_h.vector().norm("l2"), [1709.466 if deg==2 else 1209.399])
    print(pS_h.vector().norm("l2"), [5648.087 if deg==2 else 5651.009])
    print(uP_h.vector().norm("l2"), [86.59258])
    print(pP_h.vector().norm("l2"), [272.1546])

    # ****** Computing error ******** #

    form_err_uS_L2 = inner(uS_h - uS_ex, uS_h - uS_ex) * dx(stokes)
    form_err_uS_H10 = inner(sym(grad(uS_h)) - symgrad_uS_ex, sym(grad(uS_h)) - symgrad_uS_ex) * dx(stokes)
    form_err_pS_L2 = (pS_h - pS_ex)*(pS_h - pS_ex) * dx(stokes)
    form_err_uP_L2 = inner(uP_h - dP_ex, uP_h - dP_ex) * dx(poroel)
    form_err_uP_H10 = inner(sym(grad(uP_h)) - symgrad_dP_ex, sym(grad(uP_h)) - symgrad_dP_ex) * dx(poroel) + (div(uP_h) - div_dP_ex)*(div(uP_h) - div_dP_ex) * dx(poroel)
    form_err_pP_L2 = (pP_h - pP_ex)*(pP_h - pP_ex) * dx(poroel)
    form_err_pP_H10 = inner(grad(pP_h) - grad_pP_ex, grad(pP_h) - grad_pP_ex) * dx(poroel)
    form_err_jumps = (
        KvalCorr/G*eta*degP*degP/h_avg * inner(jump(pP_h - pP_ex, n), jump(pP_h - pP_ex, n)) * dS(0)
        + (2*l+5*G)*etaU*deg*deg/h_avg * inner(tensor_jump(uP_h - dP_ex, n), tensor_jump(uP_h - dP_ex, n)) * dS(0)
        + mu*etaU*deg*deg/h_avg * inner(tensor_jump(uS_h - uS_ex, n), tensor_jump(uS_h - uS_ex, n)) * dS(0)
        + eta*h_avg_S/degP * inner(jump(pS_h - pS_ex, n), jump(pS_h - pS_ex, n)) * dS(0)
        )

    # temporary assembly at current time
    ass_2_err_uS_L2 = assemble(form_err_uS_L2)
    ass_2_err_uS_H10 = assemble(form_err_uS_H10)
    ass_2_err_pS_L2 = assemble(form_err_pS_L2)
    ass_2_err_uP_L2 = assemble(form_err_uP_L2)
    ass_2_err_uP_H10 = assemble(form_err_uP_H10)
    ass_2_err_pP_L2 = assemble(form_err_pP_L2)
    ass_2_err_pP_H10 = assemble(form_err_pP_H10)

    err_energy_2 = err_energy_2 + ass_2_err_pP_H10 + ass_2_err_pP_L2 + ass_2_err_uS_H10 + ass_2_err_pS_L2 + assemble(form_err_jumps)

    # update of the L-inf norm
    err_uS_L2 = max(err_uS_L2, sqrt(ass_2_err_uS_L2))
    err_uS_H10 = max(err_uS_H10, sqrt(ass_2_err_uS_H10))
    err_pS_L2 = max(err_pS_L2, sqrt(ass_2_err_pS_L2))
    err_uP_L2 = max(err_uP_L2, sqrt(ass_2_err_uP_L2))
    err_uP_H10 = max(err_uP_H10, sqrt(ass_2_err_uP_H10))
    err_pP_L2 = max(err_pP_L2, sqrt(ass_2_err_pP_L2))
    err_pP_H10 = max(err_pP_H10, sqrt(ass_2_err_pP_H10))

    print("Errors: Stokes : uL2: ", err_uS_L2, "  uH10: ", err_uS_H10, "  pL2: ", err_pS_L2)
    print("        poroel : uL2: ", err_uP_L2, "  uH10: ", err_uP_H10, "  pL2: ", err_pP_L2, "  pH10: ", err_pP_H10)

    # ****** Saving data ******** #

    with open(outputPath+'/'+outputFileBasename+'.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([1.0/N, time, err_uS_L2, err_uS_H10, err_pS_L2, err_uP_L2, err_uP_H10, err_pP_L2, err_pP_H10, sqrt(err_energy_2)])

# ********************************* #
# ************** MAIN ************* #
# ********************************* #

parameters["ghost_mode"] = "shared_facet"  # required by dS

# ********* I/O parameters  ******* #

meshFilename = "dom_pt3p4_Hmatt15_mesh16_ALL.h5"

outputPath = "outputBrain"
outputFileBasename = "TD_stokes_poroelasticity_brain"

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
tmax = 3e-5

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
theta = Constant(0.5)

# ******* Exact solution, initial condition, and sources ****** #

exactdeg = max(deg, degP) + 1

A = pi*pi*k/G_
dP_ex = Expression(("cos(pi*x[0])*cos(pi*x[1])",\
                    "sin(pi*x[0])*sin(pi*x[1])",\
                    "x[2]"), degree=exactdeg)
dP_ex_OLD = dP_ex
pP_ex = Expression("pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", degree=exactdeg)
pP_ex_OLD = pP_ex
uS_ex = Expression(("A*(cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]))",\
                    "-A*(cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]))",\
                    "x[2]*x[2]"), degree=exactdeg, A=A)
uS_ex_OLD = uS_ex
pS_ex = Expression("(1+2*mu_*A)*pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", degree=exactdeg, A=A, mu_=mu_)
pS_ex_OLD = pS_ex

initialCondition = Expression(( \
    "A*(cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]))", \
    "-A*(cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]))", \
    "x[2]*x[2]",\
    "-cos(pi*x[0])*cos(pi*x[1])", \
    "sin(pi*x[0])*sin(pi*x[1])", \
    "x[2]",\
    "(1+2*mu_*A)*pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", \
    "pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))" \
    ), degree=exactdeg, A=A, mu_=mu_)

fP = Expression(("pi*pi*( (alpha_-4*G_-2*l_)*cos(pi*x[0])*cos(pi*x[1])-alpha_*sin(pi*x[0])*sin(pi*x[1]) )", \
                 "pi*pi*( alpha_*cos(pi*x[0])*cos(pi*x[1])-(alpha_-4*G_-2*l_)*sin(pi*x[0])*sin(pi*x[1]) )", \
                 "cos(x[2])"), degree=exactdeg, alpha_=alpha_, G_=G_, l_=l_)
fP_OLD = fP
gP = Expression("(2*A+beta_)*pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", degree=exactdeg, A=A, beta_=beta_)
gP_OLD = gP
fS = Expression(("(1+4*mu_*A)*pi*pi*( cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]) )", \
                 "pi*pi*( cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]) )", \
                 "sin(x[2])"), degree=exactdeg,  A=A, mu_=mu_)
fS_OLD = fS
gS = Constant(0.)
gS_OLD = gS
gNeuS = Expression(("0.0", \
                    "-pi*sin(pi*x[0])*(1)",\
                    "0.0"), degree=exactdeg)
gNeuS_OLD = gNeuS
gNeuSTop = Expression(("0.0", \
                       "pi*sin(pi*x[0])*(-1)",
                       "0.0"), degree=exactdeg)
gNeuSTop_OLD = gNeuSTop
gNeuP = Expression(("0.0", \
                    "pi*(alpha_-2*G_-2*l_)*sin(pi*x[0])",\
                    "0.0"), degree=exactdeg, alpha_=alpha_, G_=G_, l_=l_)
gNeuP_OLD = gNeuP

symgrad_dP_ex = Expression((("pi*(sin(pi*x[0])*cos(pi*x[1]))","pi*(cos(pi*x[0])*sin(pi*x[1]))",
                             "0.0"),\
                            ("pi*(cos(pi*x[0])*sin(pi*x[1]))","pi*(sin(pi*x[0])*cos(pi*x[1]))","0.0"),\
                            ("x[0]","x[1]","x[2]")), degree=exactdeg, A=A)

div_dP_ex = Expression("2*pi*sin(pi*x[0])*cos(pi*x[1])", degree=exactdeg)

grad_pP_ex = Expression(("pi*pi*(cos(pi*x[0])*cos(pi*x[1])-sin(pi*x[0])*sin(pi*x[1]))",\
                         "pi*pi*(-sin(pi*x[0])*sin(pi*x[1])+cos(pi*x[0])*cos(pi*x[1]))",\
                         "0.0"), degree=exactdeg, A=A)

symgrad_uS_ex = Expression((("-A*pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))", "0.0",
                             "0.0"),\
                            ("0.0", "A*pi*(sin(pi*x[0])*cos(pi*x[1])+cos(pi*x[0])*sin(pi*x[1]))",
                            "0.0"),\
                            ("x[0]","x[1]","x[2]")), degree=exactdeg, A=A)

# ******* Construct mesh and define normal, tangent ****** #

stokes = 2
poroel = 1
dirP = 4
dirS = 5
interf = 6
neuSTop = 8

# ******* Loop for h-convergence ****** #

with open(outputPath+'/'+outputFileBasename+'.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                           quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(["h", "time", "err_uS_L2", "err_uS_H10", "err_pS_L2", "err_uP_L2", "err_uP_H10", "err_pP_L2", "err_pP_H10", "err_energy"])

for ii in range(1,2):

    # ******* Set subdomains, boundaries, and interface ****** #

    mesh = MeshImportFromFile(meshFilename)
    boundaries, subdomains, restrictions = ImportMeshFunctionsFromFile(meshFilename, mesh)
	# ds_vent = Measure('ds', domain = mesh, subdomain_data = BoundaryID, subdomain_id = stokes))
	# ds_skull = Measure('ds', domain = mesh, subdomain_data = BoundaryID, subdomain_id = int(param["Domain Definition"]["ID for Skull"]))

    class MStokes(SubDomain):
        def inside(self, x, on_boundary):
            return (x[0]-0.001)**2 + (x[1]-0.05)**2 + (x[2]-0.02)**2 < 0.003**2

    class MPoroel(SubDomain):
        def inside(self, x, on_boundary):
            return True

    # subdomains = MeshFunction("size_t", mesh, 3)
    # subdomains.set_all(0)
    # MPoroel().mark(subdomains, poroel)
    # MStokes().mark(subdomains, stokes)

    # useful for output
    SubDomFESpace = FunctionSpace(mesh, 'DG', 0)
    subdomFEfun = Function(SubDomFESpace)
    dm = SubDomFESpace.dofmap()
    tagArray = numpy_asarray(subdomains.array(), dtype=numpy_int32)
    for cell in cells(mesh):
        tagArray[dm.cell_dofs(cell.index())] = subdomains[cell]
    # print(subdomains.array().size, tagArray.size, subdomFEfun.vector().size(), flush=True)
    subdomFEfun.vector()[:] = tagArray
    subdomFEfun.rename('tag','tag')

    n = FacetNormal(mesh)
    t = as_vector((-n[1], n[0]))

    def tensor_jump(u, n):
        return (outer(u('+'), n('+')) + outer(n('+'), u('+')))/2 + (outer(n('-'), u('-')) + outer(u('-'), n('-')))/2
    def tensor_jump_b(u, n):
        return  outer(u, n)/2 + outer(n, u)/2

    # ******* Set subdomains, boundaries, and interface ****** #

    OmS = MeshRestriction(mesh, MStokes())
    OmP = MeshRestriction(mesh, MPoroel())

    dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dS = Measure("dS", domain=mesh, subdomain_data=boundaries)

    # verifying the domain size
    areaP = assemble(1.*dx(poroel))
    areaS = assemble(1.*dx(stokes))
    lengthI = assemble(1.*dS(interf))
    print("area(Omega_P) = ", areaP)
    print("area(Omega_S) = ", areaS)

    # ***** Global FE spaces and their restrictions ****** #

    P2v = VectorFunctionSpace(mesh, "DG", deg)
    P1 = FunctionSpace(mesh, "DG", degP)
    BDM1 = VectorFunctionSpace(mesh, "DG", deg)
    P0 = FunctionSpace(mesh, "DG", degP)

    print(restrictions, flush=True)
    at_least_one_mesh_function = any([(
                isinstance(subdomain, (list, tuple))
                    and
                all([isinstance(mesh_function, MeshFunction) for mesh_function in subdomain])
            ) for subdomain in restrictions])
    print(at_least_one_mesh_function, flush=True)

    #                         uS,   uP, pS, pP
    Hh = BlockFunctionSpace([P2v, BDM1, P1, P0],
                            restrict=restrictions)
    #                        restrict=[OmS, OmP, OmS, OmP])

    trial = BlockTrialFunction(Hh)
    uS, uP, pS, pP = block_split(trial)
    test = BlockTestFunction(Hh)
    vS, vP, qS, qP = block_split(test)
    sol_OLD = BlockFunction(Hh)
    uS_OLD, uP_OLD, pS_OLD, pP_OLD = block_split(sol_OLD)
    to_zP_OLD = BlockFunction(Hh)
    tmp1, zP_OLD, tmp2, tmp3 = block_split(to_zP_OLD)
    to_aP_OLD = BlockFunction(Hh)
    tmp1, aP_OLD, tmp2, tmp3 = block_split(to_aP_OLD)
    to_aP_OLD2 = BlockFunction(Hh)
    tmp1, aP_OLD2, tmp2, tmp3 = block_split(to_aP_OLD2)

    print("DoFs = ", Hh.dim(), " -- DoFs with unified Taylor-Hood = ", P2v.dim() + P1.dim())


    # comm= MPI.comm_world
    # rank = comm.Get_rank()
    # dofmap = P1.dofmap()
    # with open('pardofsN3_'+str(rank)+'.txt', 'w') as fff:
    #     fff.write(str(dofmap.dofs()) + '\n' + str(mesh.num_cells()) \
    #         + ' , ' + str(dofmap.global_dimension()) \
    #         + ' , ' + str(dofmap.ownership_range()) \
    #         + ' \n ' + str(dofmap.off_process_owner()) \
    #         + ' \n ' + str(dofmap.shared_nodes()) \
    #         + ' \n ' + str(dofmap.neighbours()) \
    #         + '\n')
    #     for cell_no in range(mesh.num_cells()):
    #       fff.write("rank = " + str(rank) + ", local cell = " + str(cell_no) + ", " + \
    #             str(dofmap.cell_dofs(cell_no)) + " vs " + str(dofmap.dofs()[cell_no]) + "\n")
    #       # first: print the rank of the process
    #       # second: print the local cell index cell_no
    #       # third: print the global index of the dof associated to the cell of local index cell_no
    #       # forth: print the global index of the dof associated to the dof of local index cell_no


    # ******** Other parameters and BCs ************* #

    h = CellDiameter(mesh)
    h_avg = (2*h('+')*h('-'))/(h('+')+h('-'))
    n = FacetNormal(mesh)
    h_avg_S = (h('+')+h('-')-abs(h('+')-h('-')))/2

    # no BCs: imposed by DG

    # ****** Initial conditions ******** #

    print("Initial conditions ...", flush=True)
    
    uS_OLD = interpolate(uS_ex, Hh.sub(0))
    uP_OLD = interpolate(dP_ex, Hh.sub(1))
    pS_OLD = interpolate(pS_ex, Hh.sub(2))
    pP_OLD = interpolate(pP_ex, Hh.sub(3))
    sol_OLD = BlockFunction(Hh, [uS_OLD, uP_OLD, pS_OLD, pP_OLD])
    zeros = Expression(('0.0','0.0','0.0'), degree=1)
    aP_OLD2 = interpolate(zeros, Hh.sub(1))
    aP_OLD = interpolate(zeros, Hh.sub(1))
    zP_OLD = interpolate(zeros, Hh.sub(1))
    print("Initial conditions - done", flush=True)

    # ****** Loop for time advancement ******** #

    err_uS_L2 = 0
    err_uS_H10 = 0
    err_pS_L2 = 0
    err_uP_L2 = 0
    err_uP_H10 = 0
    err_pP_L2 = 0
    err_pP_H10 = 0
    err_energy_2 = 0

    time_N = ceil((tmax-t0)/dt)
    time = 0

    # postprocessCSV(sol_OLD, ii, 0, outputPath, outputFileBasename+"_allTimes", err_uS_L2, err_uS_H10, err_pS_L2, err_uP_L2, err_uP_H10, err_pP_L2, err_pP_H10, err_energy_2, h_avg, h_avg_S)
    postprocessXDMF(sol_OLD, ii, 0, outputPath, outputFileBasename, subdomFEfun)

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
             + (KvalCorr/G*eta*degP*degP/h_avg*inner(jump(pP,n),jump(qP,n))*dS(0)) \
             - (Kval/G*inner(avg(grad(pP)),jump(qP,n))*dS(0)) - (Kval/G*inner(avg(grad(qP)),jump(pP,n))*dS(0)) \
             + (KvalCorr/G*eta*degP*degP/h*pP*qP*ds(dirP)) \
             - (Kval/G*inner(grad(pP),n)*qP*ds(dirP)) - (Kval/G*inner(grad(qP),n)*pP*ds(dirP)) \
             )
             # + (KvalCorr/G*eta*degP*degP/h('+')*pP('+')*qP('+')*dS(interf)) \
             # - (Kval/G*inner(grad(pP('+')),n('+'))*qP('+')*dS(interf)) - (Kval/G*inner(grad(qP('+')),n('+'))*pP('+')*dS(interf)) \

        JSt = pP('+') * dot(vS('-'), n('-')) * dS(interf)
        JS = qP('+') * dot(uS('-'), n('+')) * dS(interf)
        JPt = -pP('-') * dot(vP('-'), n('-')) * dS(interf)
        # JP: use Newmark extrap to approx \dot{d} in the eq for pP
        JP = theta*newmarkGamma/(newmarkBeta*dt) * qP('+') * dot(uP('+'), n('+')) * dS(interf)

        B1Pt = - alpha * pP * div(vP) * dx(poroel) \
               + alpha * jump(vP,n) * avg(pP) * dS(0) \
               + alpha * inner(vP,n) * pP * ds(dirP) \
               + JPt

        # NB In the STEADY case, the coupling pP->uP is one-directional
        # B1P = 0
        B1P = theta*newmarkGamma/(newmarkBeta*dt) * ( \
              alpha * qP * div(uP) * dx(poroel) \
              - alpha * jump(uP,n) * avg(qP) * dS(0) \
              - alpha * inner(uP,n) * qP * ds(dirP) \
              ) \
              + JP

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
              + inner(gNeuSTop, vS) * ds(neuSTop) \
              ) \
              + (1-theta) * ( \
              + dot(fS_OLD, vS) * dx(stokes) \
              + (mu*etaU*deg*deg/h*inner(tensor_jump_b(uS_ex_OLD,n),tensor_jump_b(vS,n))*ds(dirS)) \
              - (2*mu*inner(sym(grad(vS)), tensor_jump_b(uS_ex_OLD,n))*ds(dirS)) \
              + inner(gNeuSTop_OLD, vS) * ds(neuSTop) \
              )
              #THIS SEEMS TO REPLACE JSt (and quite precisely, especially in terms of uS,pS,uP)#  + inner(gNeuS, vS('+')) * dS(interf)

        FuP = rhoP/(newmarkBeta*dt*dt) * inner(uP_OLD,vP) * dx(poroel) \
              + rhoP/(newmarkBeta*dt) * inner(zP_OLD,vP) * dx(poroel) \
              + rhoP*(1.0-2*newmarkBeta)/(2*newmarkBeta) * inner(aP_OLD,vP) * dx(poroel) \
              + dot(fP, vP) * dx(poroel) \
              + (2*l+5*G)*etaU*deg*deg/h*inner(tensor_jump_b(dP_ex,n),tensor_jump_b(vP,n))*ds(dirP) \
              - 2*G*inner(sym(grad(vP)), tensor_jump_b(dP_ex,n))*ds(dirP) \
              - l*div(vP)*inner(dP_ex,n)*ds(dirP)

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

        GqP = cP/dt * pP_OLD * qP * dx(poroel) \
              - (1-theta)*( \
              (Kval/G*inner(grad(pP_OLD),grad(qP))*dx(poroel)) \
              + beta*pP_OLD*qP*dx(poroel) \
              + (KvalCorr/G*eta*degP*degP/h_avg*inner(jump(pP_OLD,n),jump(qP,n))*dS(0)) \
              - (Kval/G*inner(avg(grad(pP_OLD)),jump(qP,n))*dS(0)) - (Kval/G*inner(avg(grad(qP)),jump(pP_OLD,n))*dS(0)) \
              + (KvalCorr/G*eta*degP*degP/h*pP_OLD*qP*ds(dirP)) \
              - (Kval/G*inner(grad(pP_OLD),n)*qP*ds(dirP)) - (Kval/G*inner(grad(qP),n)*pP_OLD*ds(dirP)) \
              ) \
              + theta*( \
              gP*qP * dx(poroel) \
              + (KvalCorr/G*eta*degP*degP/h*pP_ex*qP*ds(dirP)) \
              - (Kval/G*pP_ex*inner(grad(qP),n)*ds(dirP)) \
              ) \
              + (1-theta)*( \
              gP_OLD*qP * dx(poroel) \
              + (KvalCorr/G*eta*degP*degP/h*pP_ex_OLD*qP*ds(dirP)) \
              - (Kval/G*pP_ex_OLD*inner(grad(qP),n)*ds(dirP)) \
              ) \
              + theta*newmarkGamma/(newmarkBeta*dt) * ( \
              alpha * qP * div(uP_OLD) * dx(poroel) \
              - alpha * jump(uP_OLD,n) * avg(qP) * dS(0) \
              - alpha * inner(uP_OLD,n) * qP * ds(dirP) \
              ) \
              + (theta*newmarkGamma/newmarkBeta - 1) * ( \
              alpha * qP * div(zP_OLD) * dx(poroel) \
              - alpha * jump(zP_OLD,n) * avg(qP) * dS(0) \
              - alpha * inner(zP_OLD,n) * qP * ds(dirP) \
              ) \
              + theta*(newmarkGamma/(2*newmarkBeta) - 1)*dt * ( \
              alpha * qP * div(aP_OLD) * dx(poroel) \
              - alpha * jump(aP_OLD,n) * avg(qP) * dS(0) \
              - alpha * inner(aP_OLD,n) * qP * ds(dirP) \
              ) \
              - theta*newmarkGamma/(newmarkBeta*dt) * ( \
              + alpha * inner(dP_ex,n) * qP * ds(dirP) \
              - alpha * inner(dP_ex_OLD,n) * qP * ds(dirP) \
              ) #\
              # - (1-theta)*( \
              # + (KvalCorr/G*eta*degP*degP/h('+')*pP_OLD('+')*qP('+')*dS(interf)) \
              # - (Kval/G*inner(grad(pP_OLD('+')),n('+'))*qP('+')*dS(interf)) - (Kval/G*inner(grad(qP('+')),n('+'))*pP_OLD('+')*dS(interf)) \
              # ) \
              # + theta*( \
              # - (Kval/G*pP_ex*inner(grad(qP('+')),n('+'))*dS(interf)) \
              # ) \
              # + (1-theta)*( \
              # - (Kval/G*pP_ex_OLD*inner(grad(qP('+')),n('+'))*dS(interf)) \
              # ) \

        # ****** Assembly and solution of linear system ******** #

        rhs = [FuS, FuP, GqS, GqP]

        # this can be ordered arbitrarily. I've chosen
        #        uS   uP    pS    pP
        lhs = [[ AS,   0, B1St,  JSt],
               [  0,  AP,    0, B1Pt],
               [B1S,   0,   SS,    0],
               [ JS, B1P,    0,   SP]]

        print("Assembling LHS ...", flush=True)
        AA = block_assemble(lhs)
        print("Assembling LHS - done", flush=True)
        print("Assembling RHS ...", flush=True)
        FF = block_assemble(rhs)
        print("Assembling RHS - done", flush=True)
        # bcs.apply(AA)
        # bcs.apply(FF)

        sol = BlockFunction(Hh)
        print("Solving ...", flush=True)
        block_solve(AA, sol.block_vector(), FF, "mumps")
        print("Solving - done", flush=True)

        # ****** Update OLD variables ******** #
        print("Updating OLD variables ...", flush=True)

        block_assign(to_aP_OLD2, to_aP_OLD)
        tmp1, aP_OLD2, tmp2, tmp3 = block_split(to_aP_OLD2)

        sol_diff = BlockFunction(Hh)
        sol_diff = (sol-sol_OLD).copy(deepcopy=True)
        block_assign(to_aP_OLD, sol_diff/(newmarkBeta_*dt*dt) \
                 - to_zP_OLD/(newmarkBeta_*dt) \
                 + (2*newmarkBeta_-1)/(2*newmarkBeta_) * to_aP_OLD2)
        tmp1, aP_OLD, tmp2, tmp3 = block_split(to_aP_OLD)

        block_assign(to_zP_OLD, to_zP_OLD + dt*(newmarkGamma_*to_aP_OLD + (1-newmarkGamma_)*to_aP_OLD2))
        tmp1, zP_OLD, tmp2, tmp3 = block_split(to_zP_OLD)

        block_assign(sol_OLD, sol)
        uS_OLD, uP_OLD, pS_OLD, pP_OLD = block_split(sol_OLD)

        print("Updating OLD variables - done", flush=True)
        # ****** postprocess ******** #

        # postprocessCSV(sol_OLD, ii, time_idx, outputPath, outputFileBasename+"_allTimes", err_uS_L2, err_uS_H10, err_pS_L2, err_uP_L2, err_uP_H10, err_pP_L2, err_pP_H10, err_energy_2, h_avg, h_avg_S)
        postprocessXDMF(sol_OLD, ii, time_idx, outputPath, outputFileBasename, subdomFEfun)

    # # Re-exporting only last time in different convergence file.
    # postprocessCSV(sol_OLD, ii, time_idx, outputPath, outputFileBasename, err_uS_L2, err_uS_H10, err_pS_L2, err_uP_L2, err_uP_H10, err_pP_L2, err_pP_H10, err_energy_2, h_avg, h_avg_S)
