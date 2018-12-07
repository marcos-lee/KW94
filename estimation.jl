using Random
using Distributions
using LinearAlgebra
using CSVFiles
using DataFrames
using DelimitedFiles
using Optim
import DataStructures.OrderedDict

include("feasibleSet.jl")
include("estimation_functions.jl") #Smoothed AR
#include("estimation_approx.jl")

T = 40          #I start with t =1, the paper starts with t = 0
MC = 50

# use true parameters as guess to test
param = 1       #which parameter set to use

# Change to Parameter1.jl, Parameter2.jl or Parameters3.jl to use another set of parameters
include("Parameters$param.jl")



st = [10,0,0,1] # Initial state at t=1

@time Domain_set, tStateSpace = @timed StateSpace(st, T)


sigma = [p.σ11^2 p.σ12 p.σ13 p.σ14;
        p.σ12 p.σ22^2 p.σ23 p.σ24;
        p.σ13 p.σ23 p.σ33^2 p.σ34;
        p.σ14 p.σ24 p.σ34 p.σ44^2]
A = cholesky(sigma)
LT = A.L
θ = [p.α10, p.α11, p.α12, p.α13, p.α14, p.α15,
p.α20, p.α21, p.α22, p.α23, p.α24, p.α25,
p.β0, p.β1, p.β2, p.γ0, log(LT[1,1]), log(LT[2,2]), log(LT[3,3]), log(LT[4,4]),
LT[2,1], LT[3,1], LT[4,1], LT[3,2], LT[4,2], LT[4,3]]


Random.seed!(4)
Draws = rand(MvNormal(zeros(4), Matrix{Float64}(I, 4, 4)),MC)

df = DataFrame(load("kw94.csv"))

ApproxS = 200
lambda = 0.5

@time likelihood(df, Draws, Domain_set, lambda, θ)

function wrapll(θ)
    return likelihood(df, Draws, Domain_set, lambda, θ)
end



#########################################################
#garbage

@time mini2 = optimize(wrapll, Optim.minimizer(mini1), Optim.Options(store_trace = true,show_trace=true))

θ = a["mini1"].minimizer

func = TwiceDifferentiable(wrapll, θ1; autodiff = :forward)
@time mini = optimize(func, θ1, Newton(), Optim.Options(store_trace = true,show_trace=true))
#17hours

using JLD

save("minimization1.jld", "mini1", mini1)
θ1 = Optim.minimizer(mini1)
func2 = TwiceDifferentiable(wrapll, θ1; autodiff = :forward)
@time mini2 = optimize(func2, θ1, Newton(), Optim.Options(store_trace = true,show_trace=true))

using Pkg
Pkg.add("LineSearches")
using LineSearches

func3 = TwiceDifferentiable(wrapll, θ; autodiff = :forward)
@time mini3 = optimize(func3, θ, Newton(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.MoreThuente()), Optim.Options(store_trace = true,show_trace=true))
func4 = OnceDifferentiable(wrapll, θ1; autodiff = :forward)
@time mini4 = optimize(func4, θ1, LBFGS(;linesearch = BackTracking(order=2)), Optim.Options(store_trace = true,show_trace=true))

save("minimization2.jld", "mini2")
