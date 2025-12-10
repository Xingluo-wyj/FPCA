#function get_SQP_direction_CG(
#    pb, M, x::Vector{Tf}, structderivative; info=Dict()
#) where {Tf}
    #J=[structderivative.Jacₕ;2*x']
    #Z = nullspace(J)    # tangent space basis
   
    ## 1. Restoration step
    #r = zeros(Tf, length(x))
    #if norm(structderivative.hx) > 1e1 * eps(Tf) 
    #    r = IterativeSolvers.lsmr(J, -[structderivative.hx;x'x-1])
      
    #end
    
    ## 2. Reduced gradient and RHS
    #g = Z' * structderivative.∇Fx
    #v = -g - Z' * structderivative.∇²Lx * r
    ## 3. Linear system solve (robust)

    #u = pinv(Z' * structderivative.∇²Lx * Z )*v
    
    ## 4. Compute d
    ## linear search return d
    #d = r + Z * u
    #return d
#end

# function get_SQP_direction_JuMP(pb, M, x::Vector; regularize=false)
#     @assert manifold_codim(M) > 0

#     ## Oracle calls
#     n = length(x)
#     hx, Jacₕ, ∇Fx, λ, ∇²Lx = oracles(pb, x, M)

#     model = Model(optimizer_with_attributes(OSQP.Optimizer, "polish" => true))
#     set_silent(model)

#     d = @variable(model, d[1:n])
#     @constraint(model, hx + Jacₕ * d .== 0)

#     if regularize
#         gradFx = ∇Fx - Jacₕ' * λ
#         ∇²Lx += ((norm(gradFx))^(0.8))I
#     end

#     @objective(model, Min, ∇Fx' * d + 0.5 * d' * ∇²Lx * d)

#     # @info "Solving for SQP step"
#     JuMP.optimize!(model)
#     d = value.(d)
#     if termination_status(model) != MathOptInterface.OPTIMAL
#         @warn "Problem in SQP direction computation" termination_status(model) primal_status(
#             model
#         ) dual_status(model)
#         d .= 0
#     end

#     return d, Jacₕ
# end
function get_SQP_direction_inexactNewton(
    pb, M, x::Vector{Tf}, structderivative; 
    η_k=0.1, max_cg_iter=100, info=Dict()
) where {Tf}
    J = [structderivative.Jacₕ; 2*x']
    Z = nullspace(J)
    # Restoration step
    r = zeros(Tf, length(x))
    if norm(structderivative.hx) > 1e1 * eps(Tf) 
        r = IterativeSolvers.lsmr(J, -[structderivative.hx; x'x - 1])
    end
    
    # Reduced system
    g = Z' * structderivative.∇Fx
    v = -g - Z' * structderivative.∇²Lx * r
    A = Z' * structderivative.∇²Lx * Z
    
    # Inexact solve with CG
    u, ch = IterativeSolvers.cg(
        A, v;
        abstol=η_k * norm(v),
        maxiter=max_cg_iter,
        log=true
    )
    
    # 兼容性处理：获取残差
    info[:cg_converged] = ch.isconverged
    info[:cg_iters] = ch.iters
    info[:cg_residual] = hasproperty(ch, :residual_norm) ? ch.residual_norm[end] : norm(A*u - v)
    
    return r + Z * u
end
function addMaratoscorrection!(d::Vector{Tf}, pb, M, x, Jacₕ) where {Tf}
 
    hxd= zeros(Tf, size(Jacₕ,1))
    hxd[1:end-1] =NSP.h(M, x .+ d) 
    hxd[end] = dot(x .+ d, x .+ d) - 1
    
    norm(hxd) < 1e1 * eps(Tf) && return
    dMaratos = IterativeSolvers.lsmr(Jacₕ, -hxd)
    @debug "Maratos SOC: " norm(hx + Jacₕ * dMaratos)
    d .+= dMaratos
    return
end

