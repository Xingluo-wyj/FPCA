module LocalCompositeNewton
using CSV, DataFrames, Statistics
using IterativeSolvers
using LinearAlgebra
using Printf
using Random
using EigenDerivatives
using NonSmoothProblems
using NonSmoothSolvers
using IterativeSolvers
using DelimitedFiles
import DataStructures
using DataStructures
import NonSmoothSolvers:
    initial_state,
    print_header,
    display_logs_header_post,
    display_logs_post,
    update_iterate!,
    get_minimizer_candidate,
    has_converged
import PlotsOptim: 
    get_legendname
using PlotsOptim    


using LaTeXStrings
using DocStringExtensions
include("c:\\Users\\wuyoujia\\Desktop\\manifold optimazation\\LocalCompositeNewton.jl\\src\\prox_max.jl")
include("oracles.jl")
include("guessstructure.jl")
include("SQP.jl")
include("localNewton.jl")
include("problems/fpca.jl")
include("makeplots.jl")

# Setting numerical experiments default output directory
const NUMEXPS_OUTDIR_DEFAULT = joinpath(
    @__DIR__, "..", "numexps_output"
)

function __init__()
    if !isdir(NUMEXPS_OUTDIR_DEFAULT)
        mkdir(NUMEXPS_OUTDIR_DEFAULT)
    end
    @info "default output directory for numerical experiments is: " NUMEXPS_OUTDIR_DEFAULT
    return nothing
end



export optimize!

end 
