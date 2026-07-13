using LinearAlgebra

struct CGMemory{T}
	r::Vector{T}
	z::Vector{T}
	d::Vector{T}
	Ad::Vector{T}
	MAd::Vector{T}
	Md::Vector{T} # Only needed for Steihaug variant

	function CGMemory{T}(n::Int) where {T}
		# Init memory
		r = Vector{T}(undef, n)
		z = Vector{T}(undef, n)
		d = Vector{T}(undef, n)
		Ad = Vector{T}(undef, n)
		MAd = Vector{T}(undef, n)
		Md = Vector{T}(undef, n)
		new{T}(r, z, d, Ad, MAd, Md)
	end
end

function Base.resize!( mem::CGMemory, n)
	resize!(mem.r, n)
	resize!(mem.z, n)
	resize!(mem.d, n)
	resize!(mem.Ad, n)
	resize!(mem.MAd, n)
	resize!(mem.Md, n)
end

function cg(A::AbstractMatrix, b::AbstractVector, x0=nothing, atol=1e-6, M=I)
	n = size(A, 2)
	T = eltype(A)
	x = Vector{T}(undef, n)
	mem = CGMemory{T}( n )
	cg!(A, b, x, mem, x0, atol, M)
end

function cg!(A::AbstractMatrix, b::AbstractVector, x::AbstractVector, mem::CGMemory, x0=nothing, atol=1e-6, M=I)
	if typeof(x0) == Nothing
		x .= zero(eltype(x))
	else
		x .= x0
	end

	# Init memory
	r = mem.r
	z = mem.z
	d = mem.d
	Ad = mem.Ad
	MAd = mem.MAd

	# Initialize the variables
	# r = A*x - b
	r .= mul!(r, A, x) .- b

	# z = M*r
	mul!(z, M, r)
	@. d = -r

	# Compute the convergence criterion.
	tolerance = atol^2

	iters = 0

	# Residual norm
	norm_r = r'*z

	initial_norm_r = norm_r

	while iters < 3000 && norm_r > tolerance
		# Ad = A*d
		mul!(Ad, A, d)

		# MAd = M*Ad
		mul!(MAd, M, Ad)

		# Compute exact step length.
		alpha = (r'*z)/(d'*MAd)

		# Save the old variables
		old_norm_r = norm_r

		# Perform the step
		@. x = x + alpha * d
		@. r = r + alpha * Ad
		# r .= mul!(r, A, x) .- b
		@. z = z + alpha * MAd
		# mul!(z, M, r)

		norm_r = r'*z

		beta = norm_r/old_norm_r

		# Compute new search direction
		@. d = -r + beta * d

		iters += 1

	end

	#= println("CG")
	println(sqrt(norm_r))
	println(sqrt(norm_r/initial_norm_r)) =#

	return x, iters
end

function steihaug_cg(A::AbstractMatrix, b::AbstractVector, Delta, atol=1e-6, M=I)
	n = size(A, 2)
	T = eltype(A)
	x = Vector{T}(undef, n)
	mem = CGMemory{T}( n )
	steihaug_cg!(A, b, x, mem, Delta, atol, M)
end

"""
    steihaug_cg!(A::AbstractMatrix, b::AbstractVector, x::AbstractVector, mem::CGMemory, Delta, atol=1e-6, M=I)

Solves

    Minimize x'*M*( b + .5*A*x)
"""
function steihaug_cg!(A::AbstractMatrix, b::AbstractVector, x::AbstractVector, mem::CGMemory, Delta, atol=1e-6, M=I)
	# Initialize iterate with zero
	x .= zero(eltype(x))

	# Save squared trust-region radius
	Delta2 = Delta^2

	# Init memory
	r = mem.r
	z = mem.z
	d = mem.d
	Ad = mem.Ad
	MAd = mem.MAd
	Md = mem.Md

	# Initialize the variables
	# r = b - A*0
	@. r = b

	# z = M*r
	mul!(z, M, r)
	@. d = -r
	@. Md = -z

	# Compute the convergence criterion.
	tolerance = atol^2

	# Save length of step
	M_norm_x2 = 0.

	iters = 0

	# Residual norm
	norm_r = r'*z

	while iters < 3000 && norm_r > tolerance
		iters += 1

		# Ad = A*d
		mul!(Ad, A, d)

		# MAd = M*Ad
		mul!(MAd, M, Ad)

		# Auxiliary scalars
		gamma = d'*MAd
		xMd = x'*Md
		dMd = d'*Md

		if gamma <= 0
			# Negative curvature detected, go to boundary of trust-region
			alpha = find_alpha( M_norm_x2 - Delta2, xMd, dMd )
			@. x = x + alpha * d
			println("Negative curvature")
			return x, iters, Delta
		end

		# Compute exact step length.
		alpha = (r'*z)/gamma

		# Save the old variables
		old_norm_r = norm_r

		# Compute (squared) length of iterate after step
		M_norm_x2_new = M_norm_x2 + 2 * alpha * xMd + alpha^2 * dMd

		if M_norm_x2_new > Delta2
			alpha = find_alpha( M_norm_x2 - Delta2, xMd, dMd )
			@. x = x + alpha * d
			# println("Step too long")
			return x, iters, Delta
		end

		# Perform the step
		@. x = x + alpha * d
		@. r = r + alpha * Ad
		# r .= mul!(r, A, x) .- b
		@. z = z + alpha * MAd
		# mul!(z, M, r)

		# New (squared) length of x
		M_norm_x2 = M_norm_x2_new

		# New residual
		norm_r = r'*z

		beta = norm_r/old_norm_r

		# Compute new search direction
		@. d  = -r + beta *  d
		@. Md = -z + beta * Md

	end

	return x, iters, sqrt(M_norm_x2)
end

"""
    function find_alpha( c, b, a )

Find a positive root of

    c + 2*b*alpha + a*alpha^2 = 0

under the assumptions

    c < 0, a > 0
"""
function find_alpha( c, b, a )
	(-b + sqrt(b^2 - a * c))/a
end
