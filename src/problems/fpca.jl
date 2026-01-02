
function fpca(NUMEXPS_OUTDIR=NUMEXPS_OUTDIR_DEFAULT)
    Tf = Float64

    function MaxQuadFPCA(Tf = Float64)

    n =200
    k = 200
    As = [zeros(Tf, n, n) for i in 1:k]
    bs = [zeros(Tf, n) for i in 1:k]
   
    return MaxQuadPb{Tf}(n, k,-As, bs, zeros(k))
end
    
    pb =MaxQuadFPCA(Tf)
    
    x=rand(Tf, pb.n)

        x=LinearAlgebra.normalize(x)

        optparams_precomp = OptimizerParams(;
            iterations_limit=5,trace_length=0, time_limit=500
        )

        ## Define solvers to be run on problem
        optimdata = DataStructures.OrderedDict()
        time_limit = 500
        
       
        # Local Newton method
        getx(o, os, optimstate_additionalinfo) = deepcopy(os.x)
        getγ(o, os, optimstate_additionalinfo) = deepcopy(os.γ)
        optimstate_extensions = OrderedDict{Symbol,Function}(:x => getx, :γ => getγ)

        # find the smaller γ which gives maximal structure
        gx = sort(g(pb, x))
      
        γ = 0.0
        for i in 1:(length(gx)-1)
            γ += (gx[end - i + 1] - gx[end - i]) * i
        end

        o = LocalCompositeNewtonOpt{Tf}()
        optparams = OptimizerParams(; iterations_limit=40, trace_length=50, time_limit)
        _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
        state = initial_state(o, x, pb; γ)
        xfinal_localNewton, tr = NSS.optimize!(
            pb, o, x; state, optparams, optimstate_extensions
        )
        optimdata[o] = tr

        ## Build figures
        xopt = xfinal_localNewton
        Mopt = MaxQuadManifold(pb, [1])
        Fopt = prevfloat(F(pb, xopt))

       

        
    

  
    return true
end

