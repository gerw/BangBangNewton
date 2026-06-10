module BangBangSolver

using LinearAlgebra
using FEMBundle
using SparseArrays
using StaticArrays
using UnPack
using Printf
using Parameters

include("bang_bang_problem.jl")
include("assembly.jl")
include("cg.jl")
include("newton_solver.jl")

export BangBangProblem, PDEInformation, LinearPDEInformation, NonlinearPDEInformation, CubicPDEInformation, TanhPDEInformation
export eval_a, eval_da, eval_dda
export assemble_u, assemble_u!
export NewtonSolver, eval_dual_obj!, init!, iter!, solve!, eval_obj!, eval_deriv!
export DualSolverCG, PrimalSolverTR

end
