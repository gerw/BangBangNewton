using BangBangSolver
using FEMBundle

using LinearAlgebra
using Printf
using StaticArrays
using WriteVTK

function square_mesh(refs)
	# Setup the mesh
	vertices = [-1. 1.; 1. 1.; 1. -1.; -1. -1.]
	p = Vector{SVector{2,Float64}}(undef, 4)
	t = Vector{SVector{3,Int}}(undef, 2)

	p[1] = @SVector[-1.,  1.]
	p[2] = @SVector[ 1.,  1.]
	p[3] = @SVector[ 1., -1.]
	p[4] = @SVector[-1., -1.]

	t[1] = @SVector[1, 2, 3]
	t[2] = @SVector[1, 3, 4]

	be = [1 2; 2 3; 3 4; 4 1]'
	segments = ones(4)

	mesh = Mesh(vertices, p, be, t, segments)

	# Initial refinements
	for i=1:refs
		mesh = refine_all_cells(mesh)
	end

	return mesh
end

function solve_example(refs = 7; stype::SType = DualSolverCG(), pdeinfo = LinearPDEInformation(), bounds = 57.) where SType
	mesh = square_mesh(3)

	yd_fun = x -> (2*sin(2*pi*(x[1]+1)/2).*cos(2*pi*x[2]) + .8*(x[2]+x[1].^2) - .5)
	ua = -bounds
	ub = bounds

	println()
	@printf "%6s: %s\n" "Solver" SType
	@printf "%6s: %s\n" "PDE" typeof(pdeinfo)
	@printf "%6s: %s\n" "bounds" bounds
	println()

	if pdeinfo isa LinearPDEInformation
		@printf "%6s & %4s & %6s & %8s & %8s \\\\\n" "nodes" "iter" "solves" "time" "obj"
	else
		@printf "%6s & %4s & %4s & %6s & %8s & %8s \\\\\n" "nodes" "iter" "fact" "solves" "time" "obj"
	end

	for i=1:refs
		if i > 1
			mesh = refine_all_cells(mesh)
		end

		t = @elapsed newt = begin
			problem = BangBangProblem(mesh, yd_fun, ua, ub, pdeinfo)
			newt = NewtonSolver(problem, stype)
			solve!( newt, false )
			newt
		end

		if problem.pdeinfo isa LinearPDEInformation
			@printf "%6d & %4d & %6d & %8.4f & %8.6f \\\\\n" mesh.np newt.iter problem.n_system_solves t newt.obj
		else
			@printf "%6d & %4d & %4d & %6d & %8.4f & %8.6f \\\\\n" mesh.np newt.iter problem.n_factorizations problem.n_system_solves t newt.obj
		end

		if i == refs && (typeof(pdeinfo) == CubicPDEInformation || typeof(stype) == DualSolverCG)
			# Store the solution
			mkpath("solutions_paper")
			name = @sprintf "solutions_paper/sol_%02d_%02d_%s.vtu" i bounds typeof(pdeinfo)
			write_vtk(name, newt.problem.mesh, ["p" => newt.p, "xi" => newt.xi, "y" => newt.s.y, "yd" => problem.yd])
		end
	end

end

function solve_all_examples(refs=7)
	for infos in [
								(DualSolverCG(),   LinearPDEInformation(), 50.),
								(PrimalSolverTR(), LinearPDEInformation(), 50.),
								(DualSolverCG(),   LinearPDEInformation(), 57.),
								(PrimalSolverTR(), LinearPDEInformation(), 57.),
								(PrimalSolverTR(), CubicPDEInformation(),  50.),
								(PrimalSolverTR(), CubicPDEInformation(),  60.),
								]
		solver, pdeinfo, bounds = infos
		solve_example( refs; stype = solver, pdeinfo = pdeinfo, bounds = bounds )
	end
end

function solve_fixed_point(refs=5, bounds=2.7, maxiter=50)
	mesh = square_mesh(2+refs)

	yd_fun = x -> (2*sin(2*pi*(x[1]+1)/2).*cos(2*pi*x[2]) + .8*(x[2]+x[1].^2) - .5)
	ua = -bounds
	ub = bounds

	println("Nodes: ", mesh.np)

	problem = BangBangProblem(mesh, yd_fun, ua, ub, LinearPDEInformation())
	newt = NewtonSolver(problem, DualSolverCG())
	init!(newt, true)

	for i = 1:maxiter
		newt.xi .= problem.yd - newt.y
		newt.dual_obj,~ = eval_dual_obj!(newt)

		gap = newt.s.obj + newt.dual_obj
		nres = sqrt(newt.res'*mul!(newt.tmp,problem.M,newt.res))

		@printf "%3d % e %e  % e  %e  %e % 2d\n" i newt.dual_obj newt.s.obj gap nres 0 0
	end
end
