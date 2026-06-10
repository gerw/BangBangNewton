abstract type SubproblemInformation end

struct BoundInformation <: SubproblemInformation
	ua::Float64
	ub::Float64
end

abstract type PDEInformation end

# Broadcast PDEInformation as scalar
Base.Broadcast.broadcastable(x::PDEInformation) = Ref(x)

@with_kw struct LinearPDEInformation <: PDEInformation
	# zeroth order term
	c::Float64 = 10.
	dirichletBC::Bool = false
end

abstract type NonlinearPDEInformation <: PDEInformation end

@with_kw struct CubicPDEInformation <: NonlinearPDEInformation
	# zeroth order term
	c::Float64 = 10.
	# Cubic term
	a::Float64 = 3.
	dirichletBC::Bool = false
end

@with_kw struct TanhPDEInformation <: NonlinearPDEInformation
	# zeroth order term
	c::Float64 = 10.
	# Tanh term
	a::Float64 = 1.
	dirichletBC::Bool = false
end

mutable struct AbstractBangProblem{GD,GD2,I <: SubproblemInformation, PDEI <: PDEInformation}
	mesh::Mesh{GD,GD2}

	K::SparseMatrixCSC{Float64, Int64}
	M::SparseMatrixCSC{Float64, Int64}

	M_lump::Vector{Float64}

	# Factorization of stiffness matrix
	K_fac::SparseArrays.CHOLMOD.Factor{Float64,Int64}

	# Factorization of mass matrix
	M_fac::SparseArrays.CHOLMOD.Factor{Float64,Int64}

	# Indicies of the diagonal in K
	K_diag_idx::Vector{Int64}

	yd::Vector{Float64}

	info::I

	pdeinfo::PDEI

	tmp::Vector{Float64}
	P::SparseMatrixCSC{Float64,Int64}

	# Count number of factorizations (of stiffness matrix) and system solves
	n_factorizations::Int64
	n_system_solves::Int64

	function AbstractBangProblem( mesh::Mesh{GD,GD2}, yd_fun::Function, info::I, pdeinfo::PDEI) where {GD, GD2, I <: SubproblemInformation, PDEI <: PDEInformation}
		fe = FE_Lagrange{1}()
		quad = quadrature_unit_triangle_area(2)

		# zeroth order term
		c = pdeinfo.c

		K, ~ = area_integrator(mesh, fe, quad, 1.0, nothing,   c, nothing)
		M, ~ = area_integrator(mesh, fe, quad, 0.0, nothing, 1.0, nothing)

		@assert issymmetric(K)
		@assert issymmetric(M)

		M_lump = dropdims(sum(M, dims=2), dims=2)

		if pdeinfo.dirichletBC
			# Gather inner nodes
			inner_idx = trues(mesh.np)
			for i = 1:length(mesh.be)
				inner_idx[mesh.e[mesh.be[i][1]][1]] = false
				inner_idx[mesh.e[mesh.be[i][1]][2]] = false
			end

			K = K[inner_idx, inner_idx]

			M_lump = M_lump[inner_idx]

			n_inner = sum(inner_idx)

			tmp = Vector{Float64}(undef, n_inner)
			P = sparse((1:mesh.np)[inner_idx],1:n_inner,ones(n_inner), mesh.np, n_inner)
		else
			tmp = Vector{Float64}()
			P = sparse(LinearAlgebra.I, mesh.np, mesh.np)
		end

		n = size(K)[1]
		K_diag_idx = Vector{Int64}(undef, n)
		for i = 1:n
			for j = nzrange(K, i)
				if rowvals(K)[j] == i
					K_diag_idx[i] = j
					break
				end
			end
		end

		K_fac = cholesky(K)
		M_fac = cholesky(M)

		yd = Vector{Float64}(undef, mesh.np)


		for i = 1:mesh.np
			yd[i] = yd_fun( mesh.p[i] )
		end

		# We already factored the stiffness matrix above
		n_factorizations = 1
		n_system_solves = 0

		return new{GD,GD2,I,PDEI}(mesh, K, M, M_lump, K_fac, M_fac, K_diag_idx, yd, info, pdeinfo, tmp, P, n_factorizations, n_system_solves)
	end
end

# Aliases
const BangBangProblem{GD,GD2} = AbstractBangProblem{GD, GD2, BoundInformation}

const LinearBangProblem{GD,GD2,I} = AbstractBangProblem{GD, GD2, I, LinearPDEInformation}
const NonlinearBangProblem{GD,GD2,I} = AbstractBangProblem{GD, GD2, I, <:NonlinearPDEInformation}

# Convenience constructors
BangBangProblem(mesh, yd_fun, ua, ub, pdeinfo = LinearPDEInformation()) = AbstractBangProblem(mesh, yd_fun, BoundInformation(ua, ub), pdeinfo)

# Easy access to ua and ub
function Base.getproperty(obj::BangBangProblem, sym::Symbol)
	if sym === :ua || sym === :ub
		return getproperty(getfield(obj, :info), sym)
	else # fallback to getfield
		return getfield(obj, sym)
	end
end

mutable struct State
	# The state vector
	y::Vector{Float64}

	# Linearized stiffness matrix
	Ka::SparseMatrixCSC{Float64, Int64}

	# Factorization of linearized state matrix
	Ka_fac::SparseArrays.CHOLMOD.Factor{Float64,Int64}

	# Value of the objective function
	obj::Float64

	function State( problem::AbstractBangProblem )
		y = zeros(problem.mesh.np)

		if typeof(problem) <: LinearBangProblem
			# For a linear problem, 'Ka' will not be overwritten, thus we do not need a copy
			Ka = problem.K
		else
			# For a nonlinear problem, 'Ka' will be overwritten in solve_state!, thus we need a copy
			Ka = copy(problem.K)
		end

		# Make a copy of the factorization since we reuse its symbolic part
		Ka_fac = copy(problem.K_fac)

		new(y, Ka, Ka_fac, 0.)
	end
end

function solve_lin_state!(problem::AbstractBangProblem, s::Union{State,Nothing}, y, u)
	if s === nothing
		Ka_fac = problem.K_fac
	else
		Ka_fac = s.Ka_fac
	end
	if problem.pdeinfo.dirichletBC
		mul!( problem.tmp, problem.P', u)
		Y = Ka_fac \ problem.tmp
		mul!( y, problem.P, Y )
	else
		y .= Ka_fac \ u
	end

	problem.n_system_solves += 1

	y
end

function solve_state!(problem::LinearBangProblem, s::State, u)
	solve_lin_state!(problem, s, s.y, u)
end

function solve_state!(problem::NonlinearBangProblem, s::State, u)
	y = s.y

	# Allocate memory for residual
	residual = zero(u)

	# Solves the state equation and updates the factorization Ka_fac
	maxiter = 15
	for k = 1:maxiter
		# residual = problem.K * (problem.P' * y) .+ problem.M_lump .* eval_a.(problem.pdeinfo, y) .- u
		if problem.pdeinfo.dirichletBC
			mul!( problem.tmp, problem.P', y)
			mul!( residual, problem.K, problem.tmp)
			# TODO: Multiplication with P is missing here
		else
			mul!( residual, problem.K, y)
		end
		@. residual += problem.M_lump * eval_a(problem.pdeinfo, y) - u

		# s.Ka .= problem.K .+ spdiagm( 0 => problem.M_lump .* eval_da.(problem.pdeinfo, y) )
		@. $nonzeros(s.Ka)[problem.K_diag_idx] = @view($nonzeros(problem.K)[problem.K_diag_idx]) + problem.M_lump * eval_da(problem.pdeinfo, y)

		# Compute factorization by reusing the symbolic factorization
		problem.n_factorizations += 1
		cholesky!( s.Ka_fac, s.Ka )

		problem.n_system_solves += 1
		dy = s.Ka_fac \ residual

		y .-= dy

		if dot(residual, dy) < 1e-24
			break
		end
		if k == maxiter
			throw(ErrorException("solve_state! did not converge"))
		end
	end

	y
end

function eval_a(pdeinfo::CubicPDEInformation, y)
	pdeinfo.a * y^3
end

function eval_da(pdeinfo::CubicPDEInformation, y)
	3 * pdeinfo.a * y^2
end

function eval_dda(pdeinfo::CubicPDEInformation, y)
	6 * pdeinfo.a * y
end

function eval_a(pdeinfo::TanhPDEInformation, y)
	pdeinfo.a * tanh(y)
end

function eval_da(pdeinfo::TanhPDEInformation, y)
	pdeinfo.a * (1. - tanh(y)^2)
end

function eval_dda(pdeinfo::TanhPDEInformation, y)
	- pdeinfo.a * 2 * tanh(y) * (1 - tanh(y)^2)
end

function eval_dda(pdeinfo::LinearPDEInformation, y)
	0.
end
