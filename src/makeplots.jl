
PlotsOptim.get_legendname(obj::LocalCompositeNewtonOpt) = "LocalNewton"

# Prefer a raster backend to avoid LaTeX/PGF dependency when saving figures
try
    @eval using Plots
    gr()
catch e
    @warn "Could not switch Plots backend to GR; LaTeX-based output may still be used." e
end

getabsc_time(optimizer, trace) = [state.time for state in trace]

function runnumexps()
    expe_maxquad()
    expe_eigmax()
    return nothing
end

function treatproxsteps(pb, tr, Mopt::EigmaxManifold, xopt)
    stepinfo = Any[]
    for os in tr[1:end]
        x = os.additionalinfo.x
        gx = eigvals(NSP.g(pb, x))

        # Computing steps low, up
        r = Mopt.eigmult.r
        γlow, γup = get_γlowγupmax(gx, r)
        γₖ = os.additionalinfo.γ
        distopt = norm(x - xopt)
        push!(stepinfo, (; γlow, γup, γₖ, distopt))
    end
    return stepinfo
end
function treatproxsteps(pb, tr, Mopt::MaxQuadManifold, xopt::Vector{Tf}) where {Tf}
    stepinfo = Any[]
    for os in tr[1:end]
        x = os.additionalinfo.x
        gx = NSP.g(pb, x)
        # Computing steps low, up
        r = length(Mopt.active_fᵢ_indices)
        γlow, γup = get_γlowγupmax(gx, r)
        γₖ = os.additionalinfo.γ
        distopt = norm(x - xopt)
        distopt == 0 && (distopt = eps(Tf))
        push!(stepinfo, (; γlow, γup, γₖ, distopt))
    end
    return stepinfo
end
function get_γlowγupmax(gx, r)
    gxsort = sort(gx; rev=true)

    γlow = 0
    for k in 1:(r - 1)
        γlow += k * (gxsort[k] - gxsort[k + 1])
    end
    γup = γlow + r * (gxsort[r] - gxsort[r + 1])

    return γlow, γup
end

function buildfigures(optimdata, tr, pb, xopt, Mopt, Fopt, pbname::String; NUMEXPS_OUTDIR, plotgamma = true, includelegend = false)
    @info "building figures for $pbname"

    # 1. 绘制步长参数γ图（可选）
    if plotgamma
        stepinfo = treatproxsteps(pb, tr, Mopt, xopt)
        optimdatagamma = OrderedDict(
            L"\gamma low(x_k)" => [(itstepinfo.γlow, itstepinfo.distopt) for itstepinfo in stepinfo],
            L"\bar{\gamma}(x_k)" => [(itstepinfo.γup, itstepinfo.distopt) for itstepinfo in stepinfo],
            L"\gamma_k" => [(itstepinfo.γₖ, itstepinfo.distopt) for itstepinfo in stepinfo],
        )

        getabsc_distopt(o, trace) = [o[2] for o in trace]
        getord_gamma(o, trace) = [o[1] for o in trace]
        fig = plot_curves(
            optimdatagamma,
            getabsc_distopt,
            getord_gamma;
            xlabel=L"\| x_k - x^\star\|",
            ylabel=L"",
            xmode="log",
            includelegend=false,
        )
        try
            PlotsOptim.savefig(fig, joinpath(NUMEXPS_OUTDIR, pbname * "_gamma"))
        catch e
            @warn "Error while building gamma figure" e
        end
    end
    
    # 2. 绘制子优性随时间变化图
    getabsc_time(optimizer, trace) = [state.time for state in trace]
    getord_subopt(optimizer, trace) = [state.Fx - Fopt for state in trace]
    fig_subopt = plot_curves(
        optimdata,
        getabsc_time,
        getord_subopt;
        xlabel="time (s)",
        ylabel=L"F(x_k) - F^\star",
        nmarks=1000,
        includelegend,
    )
    try
        PlotsOptim.savefig(fig_subopt, joinpath(NUMEXPS_OUTDIR, pbname * "_time_subopt"))
    catch e
        @warn "Error while building suboptimality figure" e
    end
    
    # 3. 新增：绘制距离最优解随迭代次数的变化图
    getabsc_iter(optimizer, trace) = 0:(length(trace)-1)  # 迭代次数从0开始
    getord_dist(optimizer, trace) = [norm(state.additionalinfo.x - xopt)+1e-15 for state in trace]
    
    fig_dist = plot_curves(
        optimdata,
        getabsc_iter,
        getord_dist;
        xlabel="Iteration ",
        ylabel=L"\| x_k - x^\star \|",
        nmarks=1000,
        ymode="log",  # 使用对数坐标便于观察收敛
        includelegend=includelegend,
    )
    try
        PlotsOptim.savefig(fig_dist, joinpath(NUMEXPS_OUTDIR, pbname * "_iter_dist"))
    catch e
        @warn "Error while building distance figure" e
    end
    
    return true
end
