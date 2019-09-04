using Random
using Distributions
using LinearAlgebra
using CSVFiles
using DataFrames
using DelimitedFiles
using Optim, NLSolversBase
import DataStructures.OrderedDict
using LineSearches
import NLopt


include("feasibleSet.jl")
include("estimation_functions.jl")

function estimateModel()
        Random.seed!(4)

        #T = 3          #I start with t =1, the paper starts with t = 0
        #N = 100
        MC = 2000
        sim = 100000
        ApproxS = 200
        lambda = 500.0
        # use true parameters as guess to test
        param = 1       #which parameter set to use
        df = DataFrame(load("output/df$(param)_MC$sim.csv"))
        state, wage, choice = convertDF(df)
        T = size(wage, 2)

        # Change to Parameter1.jl, Parameter2.jl or Parameters3.jl to use another set of parameters
        st = [10,0,0,1] # Initial state at t=1

        Domain_set = StateSpace(st, T)

        #MC_ϵ = rand(MvNormal([0,0,0,0], sigma),sim) #Take S draws from the Multivariate Normal

        Draws = rand(MvNormal(zeros(4), Matrix{Float64}(I, 4, 4)), MC * T)
        Draws = reshape(Draws, 4, MC, T)

        θ = makeInitialGuess(param)

        return state, wage, choice, Draws, Domain_set, lambda, θ
end


function wrapll(θ::Vector, grad::Vector)
        if length(grad) > 0
                nothing
        end
        test = likelihood(state, wage, choice, Draws, Domain_set, lambda, θ)
        println("value $test")
        return test
end
####################

# check type stability
@code_warntype likelihood(state, wage, choice, Draws, Domain_set, lambda, θ)
@code_warntype wrapll(θ, [])
####################
# USING NLOPT
state, wage, choice, Draws, Domain_set, lambda, θ = estimateModel()

opt = NLopt.Opt(:LN_NEWUOA, size(θ)[1])
opt.min_objective = wrapll
opt.xtol_rel = 1e-06

(optf,optx,ret) = NLopt.optimize(opt, θ)


#########################################################
# USING OPTIM
using LineSearches

function Optimwrapll(θ::Vector)
        res = likelihood(state, wage, choice, Draws, Domain_set, lambda, θ)
        return res
end

# forward for Automatic Diff, finite for Finite Diff
func = OnceDifferentiable(Optimwrapll, θ; autodiff = :forward)
@time mini = optimize(func, θ, LBFGS(;linesearch = BackTracking()), Optim.Options(store_trace = true,show_trace=true, iterations = 5000))
