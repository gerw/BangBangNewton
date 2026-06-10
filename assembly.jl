function assemble_u(prob::AbstractBangProblem, p::AbstractVector, it::Union{AbstractVector{Bool},Nothing} = nothing)

	I = Vector{Int64}(undef, prob.mesh.nt)
	J = Vector{Int64}(undef, prob.mesh.nt)
	S = Vector{Float64}(undef, prob.mesh.nt)

	F = Vector{Float64}(undef, prob.mesh.np)

	L = @time assemble_u!(prob, p, F, I, J, S, it)

	return F, L
end

function assemble_u!(prob::BangBangProblem{GD}, p::AbstractVector, F::AbstractVector, I::AbstractVector{Int64}, J::AbstractVector{Int64}, S::AbstractVector{Float64}, it::Union{AbstractVector{Bool},Nothing} = nothing, scale::Union{AbstractVector,Nothing} = nothing ) where GD

	F .= 0.

	mesh = prob.mesh

	# Evaluate the shape functions and their derivatives on the reference
	# cell in the midpoint of the cell
	~, dshape_array = shape(FE_Lagrange(1), @SVector[1.0/3.0,1.0/3.0,1.0/3.0], Val(true))
	dshape = dshape_array[1]

	# Clear the arrays I, J, S
	empty!(I)
	empty!(J)
	empty!(S)

	if scale != nothing
		empty!(scale)
	end

	# Check that p does not contain NaN
	@assert !any(isnan, p)

	for i = 1:mesh.nt
		ind = mesh.t[i]
		local_p = p[ind]

		# Compute area of triangle
		if GD == 2
			area = abs(det(mesh.affine_matrix[i])) / 2.
		else
			area = sqrt(det(mesh.affine_matrix[i]'*mesh.affine_matrix[i])) / 2.
		end

		if all( local_p .>= 0 )
			# Integrate over v
			val = prob.ub*area/3;
			F[ind] .+= val
			continue
		elseif all( local_p .<= 0 )
			# Integrate over -v
			val = prob.ua*area/3;
			F[ind] .+= val
			continue
		end

		if typeof(it) != Nothing
			it[i] = true
		end

		# Gradient of p
		grad = mesh.affine_invmatrixT[i] * dshape * local_p
		norm_gp = norm(grad)

		if sum( local_p .> 0 ) == 1
			# Flip p
			mult = -1
			local_p = -local_p

			vb = prob.ua
			va = prob.ub
		else
			mult = 1

			vb = prob.ub
			va = prob.ua
		end

		# Now, local_p has one negative entry and two non-negative entries
		j = findfirst(x -> x < 0, local_p);
		if j > 1
			# Swap j and 1
			if j == 2
				ind = @SVector[ ind[2], ind[1], ind[3] ]
			else
				ind = @SVector[ ind[3], ind[2], ind[1] ]
			end

			local_p = p[ind]
		end

		# Barycentric coordinates of zeros
		lambda2 = (0 - local_p[1]) / (local_p[2] - local_p[1])
		l1 = @SVector[1-lambda2, lambda2, 0]
		lambda3 = (0 - local_p[1]) / (local_p[3] - local_p[1])
		l2 = @SVector[1-lambda3, 0, lambda3]

		# Barycentric coordinates of midpoint of small triangle
		c_mid = @SVector[(1 + 1-lambda2 + 1-lambda3)/3, lambda2/3, lambda3/3]

		# Area of small triangle
		c_area = area * lambda2 * lambda3

		# Add integral to F
		@. F[ind] += -(vb-va)*c_area*c_mid + vb * area / 3 * @SVector[1., 1., 1.]

		# Length of edge
		p1 = hcat(mesh.p[ind]...) * @SVector[1-lambda2, lambda2, 0]
		p2 = hcat(mesh.p[ind]...) * @SVector[1-lambda3, 0, lambda3]
		l = norm(p1 - p2)

		# Quadrature formula for edge
		#= mu1 = l1
		mu2 = l2
		mu3 = l1 + l2
		w1 = l/6
		w2 = l/6
		w3 = l*4/6
		local_L = (w1*(mu1*mu1') + w2*(mu2*mu2') + w3*(mu3*mu3')) / norm_gp =#

		if scale == nothing
			# Assemble second derivative of sign
			w = l/6
			local_L = hcat(l1, l2)*@SMatrix[2 1; 1 2]*hcat(l1, l2)' * (w / norm_gp)


			append!(I, vcat(ind,ind,ind))
			append!(J, hcat(ind,ind,ind)')
			append!(S, local_L)
		else
			# Collect scaling of local mass matrix
			push!(scale, l/(6 * norm_gp))

			k = length(scale)

			# Assemble trace operator
			append!(I, @SVector[2*k-1, 2*k-1, 2*k, 2*k])
			append!(J, @SVector[ind[1], ind[2], ind[1], ind[3]])
			append!(S, @SVector[l1[1],  l1[2],  l2[1],  l2[3]])
		end

	end

	if scale == nothing
		L = sparse(I, J, S, mesh.np, mesh.np)
		L .*= prob.ub - prob.ua
	else
		L = sparse(I, J, S, 2*length(scale), mesh.np)
		scale .*= prob.ub - prob.ua
	end

	return L, 0.
end
