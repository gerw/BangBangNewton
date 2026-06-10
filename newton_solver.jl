import LinearAlgebra.mul!

abstract type NewtonSolverType end

abstract type DualSolver <: NewtonSolverType end

struct DualSolverCG <: DualSolver end

abstract type PrimalSolver <: NewtonSolverType end

@with_kw mutable struct PrimalSolverTR <: PrimalSolver
	Delta_old::Float64 = 0.0
	Delta::Float64 = 1.0
	eta1::Float64 = .75
	eta2::Float64 = .25
	gamma1::Float64 = .5
	gamma2::Float64 = 2.
	rho::Float64 = 0.
end

"""
A struct holding one iterate of the Newton solver
"""
mutable struct Iterate
	xi::Vector{Float64}
	F::Vector{Float64}
	s::State
	L::SparseMatrixCSC{Float64, Int64}
	function Iterate(problem::AbstractBangProblem)
		xi = Vector{Float64}(undef, problem.mesh.np)
		F = Vector{Float64}(undef, problem.mesh.np)
		s = State(problem)
		L = sparse([],[],[],problem.mesh.np,problem.mesh.np)
		return new(xi, F, s, L)
	end
end

mutable struct NewtonSolver{P <: AbstractBangProblem, SType <: NewtonSolverType }
	problem::P

	# Current iterate
	iterate::Iterate

	# Trial iterate, currently only used in PrimalSolverTR
	iterate_trial::Iterate

	# Current iterates
	p::Vector{Float64}
	res::Vector{Float64}

	# Step
	z::Vector{Float64}

	# Iteration stuff
	tau::Float64
	ncg::Int64
	iter::Int64
	output::Bool
	dual_obj::Float64

	# Temporary vector
	tmp::Vector{Float64}
	tmp2::Vector{Float64}
	cgmem::CGMemory{Float64}

	# For construction of sparse matrix
	I::Vector{Int64}
	J::Vector{Int64}
	S::Vector{Float64}


	# Scaling of interface mass matrix
	scale::Vector{Float64}

	stype::SType

	function NewtonSolver( problem::P, stype::SType = DualSolverCG() ) where {P <: AbstractBangProblem, SType <: NewtonSolverType }

		if P <: NonlinearBangProblem && SType <: DualSolver
			throw(error("Nonlinear problems cannot be solved with dual solvers."))
		end

		iterate = Iterate(problem)
		iterate_trial = Iterate(problem)

		p = Vector{Float64}(undef, problem.mesh.np)
		res = Vector{Float64}(undef, problem.mesh.np)

		z = Vector{Float64}(undef, problem.mesh.np)

		tmp = Vector{Float64}(undef, problem.mesh.np)
		tmp2 = Vector{Float64}(undef, problem.mesh.np)
		cgmem = CGMemory{Float64}(problem.mesh.np)

		I = Vector{Int64}(undef, problem.mesh.nt)
		J = Vector{Int64}(undef, problem.mesh.nt)
		S = Vector{Float64}(undef, problem.mesh.nt)

		scale = Vector{Float64}()

		new{P, SType}(problem, iterate, iterate_trial, p, res, z, 0.0, 0, 0, true, 0.0, tmp, tmp2, cgmem, I, J, S, scale, stype)
	end
end

# Easy access to fields of current iterate
function Base.getproperty(obj::NewtonSolver, sym::Symbol)
	if sym === :xi || sym === :s || sym === :L || sym === :F
		return getproperty(getfield(obj, :iterate), sym)
	elseif sym === :obj || sym === :y
		return getproperty(getfield(obj, :iterate).s, sym)
	else # fallback to getfield
		return getfield(obj, sym)
	end
end
function Base.getproperty(obj::Iterate, sym::Symbol)
	if sym === :obj || sym === :y
		return getproperty(getfield(obj, :s), sym)
	else # fallback to getfield
		return getfield(obj, sym)
	end
end

function eval_dual_obj!( newt::NewtonSolver{P, DualSolverCG} ) where {P}
	@unpack problem, xi, y, p, s, F, res, tmp, tmp2, I, J, S = newt
	@unpack M, yd = problem

	# Compute adjoint
	mul!(tmp, M, xi)
	solve_lin_state!( problem, nothing, p, tmp )

	scale = nothing

	L, obj = assemble_u!(problem, p, F, I, J, S, nothing, scale)
	newt.iterate.L = L

	solve_lin_state!( problem, nothing, y, F )

	# dual_obj = .5*(xi-yd)'*M*(xi-yd) + (F'*p - obj) - .5*yd'*M*yd
	@. tmp = xi - yd
	dual_obj = .5*(tmp'*mul!(tmp2,M,tmp)) + (F'*p - obj) - .5*(yd'*mul!(tmp2,M,yd))
	@. tmp = y - yd
	s.obj = .5*(tmp'*mul!(tmp2,M,tmp)) + obj
	res .= xi - yd + y

	return dual_obj, s.obj
end

function eval_obj!( newt::NewtonSolver{P, SType}, iterate::Iterate = newt.iterate ) where {P, SType}
	@unpack problem, tmp, tmp2, I, J, S = newt
	@unpack M, yd = problem
	@unpack xi, y, s, F = iterate

	# Compute forcing term
	iterate.L, obj = assemble_u!(problem, xi, F, I, J, S, nothing, nothing)

	# Solve state equation
	solve_state!( problem, s, F )

	# Evaluate objective
	@. tmp = yd - y
	s.obj = .5*(tmp'*mul!(tmp2,M,tmp)) + obj
end

function eval_deriv!( newt::NewtonSolver{P, SType} ) where {P, SType}
	@unpack problem, xi, y, p, s, res, tmp, tmp2 = newt
	@unpack M, yd = problem

	# Compute adjoint
	@. tmp = yd - y
	mul!(tmp2, M, tmp)
	solve_lin_state!( problem, s, p, tmp2 )

	# Evaluate residual
	res .= xi - p

	nothing
end

function eval_nres( newt::NewtonSolver )
	@unpack problem, res, tmp, L = newt
	@unpack K, M = problem

	# TODO: In which norm should we measure the residual?
	# nres = sqrt(res'*mul!(tmp,M,res))
	nres = sqrt(res'*mul!(tmp,M,res) + res'*mul!(tmp,K,res))
	nres_interface = sqrt(res'*mul!(tmp,L,res))

	return nres, nres_interface
end

function solve!( newt::NewtonSolver, output::Bool = true, maxiter = 2000)
	init!( newt, output )

	while iter!( newt ) == false && newt.iter < maxiter
	end

	if newt.iter >= maxiter
		println("Solver did not converge in ", maxiter, " iterations.")
	end
end

function init!( newt::NewtonSolver{P, SType}, output::Bool = true ) where {P, SType}
	if SType <: DualSolver
		# Initialize dual variable
		newt.xi .= newt.problem.yd
	else
		# Initialize auxiliary variable
		solve_lin_state!(newt.problem, nothing, newt.xi, newt.problem.yd)
		# Initialize state variable
		newt.s.y .= 0.
	end
	newt.iter = 0
	newt.ncg = 0
	newt.tau = 0
	newt.output = output

	if newt.output
		if SType <: DualSolver
			println("it  dual obj      primal obj     gap           norm_res      tau           cg iter")
		elseif SType <: PrimalSolverTR
			println("it  primal obj     norm_res      norm_res_int  Delta         rho           cg iter")
		end
	end

	if SType <: DualSolver
		newt.dual_obj, ~ = eval_dual_obj!( newt )
	else
		eval_obj!( newt )
		eval_deriv!( newt )
	end

	nothing
end

function iter!( newt::NewtonSolver{P, DualSolverCG} ) where {P}
	@unpack problem, xi, y, p, s, res, z, tmp, L, dual_obj, tau, ncg = newt
	@unpack K, M, yd = problem

	newt.iter += 1

	nres = sqrt(res'*mul!(tmp,M,res))

	gap = s.obj + dual_obj

	if newt.output
		@printf "%3d % e %e  % e  %e  %e % 2d\n" newt.iter dual_obj s.obj gap nres tau ncg
	end

	if nres < 1e-10
		return true
	end

	# Compute search direction
	~, newt.ncg = cg!(NewtonMatrix(newt), res, z, newt.cgmem, nothing, min(nres^(3/2), .05*nres), M)

	dz = res'*M*z

	tau = 1.0
	xi_old = copy(xi)

	dual_obj_new = 0.
	primal_obj_new = 0.

	while true
		@. xi = xi_old - tau * z

		dual_obj_new, primal_obj_new = eval_dual_obj!( newt )

		nres_new = sqrt(res'*mul!(tmp,M,res))

		if (gap < 1e-8 && nres_new < .9*nres) || dual_obj_new < dual_obj - .125*tau*dz
			break
		end

		tau = tau / 2
		if tau < 1e-12
			println("WARNING: Line search failed")
			return true
		end
	end

	newt.dual_obj = dual_obj_new
	# newt.primal_obj = primal_obj_new
	newt.tau = tau

	return false
end

function iter!( newt::NewtonSolver{P, PrimalSolverTR} ) where {P}
	@unpack iterate, iterate_trial, obj, res, z, tmp, tmp2, L, ncg, stype = newt
	@unpack Delta_old, Delta, eta1, eta2, gamma1, gamma2, rho = stype

	newt.iter += 1

	# Evaluate norms of the residual
	nres, nres_interface = eval_nres(newt)

	if newt.output
		@printf "%3d %e  %e  %e  %e % e % 2d\n" newt.iter obj nres nres_interface Delta_old rho ncg
	end

	# Termination criterion
	if nres < 1e-10 && nres_interface < 1e-10
		return true
	end

	# Compute search direction
	tol = min(nres_interface^(3/2), .05*nres_interface)
	~, newt.ncg, Delta_act = steihaug_cg!(NewtonMatrix(newt), res, z, newt.cgmem, Delta, tol, L)

	# Predicted reduction
	pred = -( res'*mul!(tmp, L, z) + .5*(z'*mul!(tmp, L, mul!(tmp2, NewtonMatrix(newt), z)) ) )
	# pred = -( z'*mul!(tmp2, L, tmp .= res .+ .5 .* mul!(tmp2, NewtonMatrix(newt), z)) )

	@assert pred >= 0

	# Try the step
	@. iterate_trial.xi = iterate.xi + z
	eval_obj!( newt, iterate_trial )

	# Test the step
	ared = iterate.s.obj - iterate_trial.s.obj

	# Remember the old trust-region radius
	Delta_old = Delta
	rho = ared/pred

	## Accept all steps if we are near the solution
	if ( nres < 1e-8 || abs(ared) < 1e-12 ) && rho < eta2
		rho = 1.234
	end

	if rho >= eta2
		# Accept step
		newt.iterate, newt.iterate_trial = newt.iterate_trial, newt.iterate
		eval_deriv!( newt )

		if rho >= eta1
			# Enlarge trust-region radius
			Delta = max(Delta, Delta_act * gamma2)
		end
	else
		# Discard step and decrease trust-region radius
		Delta = Delta_act * gamma1
	end

	@pack! stype = Delta_old, Delta, rho

	return false
end

struct NewtonMatrix{P, SType} <: AbstractMatrix{Float64}
	newton::NewtonSolver{P, SType}
end

function mul!(x, A::NewtonMatrix{P, DualSolverCG}, b) where {P}
	@unpack newton = A
	@unpack problem, L, tmp, tmp2 = newton
	@unpack M = problem

	mul!(tmp, M, b)
	solve_lin_state!(problem, nothing, tmp2, tmp)

	mul!(tmp, L, tmp2)

	solve_lin_state!(problem, nothing, tmp2, tmp)
	x .= b + tmp2
end

function mul!(x, A::NewtonMatrix{P, PrimalSolverTR}, b) where {P}
	@unpack newton = A
	@unpack problem, L, tmp, tmp2, y, s, p = newton
	@unpack M, M_lump, pdeinfo = problem

	mul!(tmp, L, b)
	solve_lin_state!(problem, s, tmp2, tmp)

	# Apply hessian
	mul!(tmp, M, tmp2)
	@. tmp += M_lump * eval_dda(pdeinfo, y) * p * tmp2

	solve_lin_state!(problem, s, tmp2, tmp)
	x .= b + tmp2
end
