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

T = 10          #I start with t =1, the paper starts with t = 0
N = 100
MC = 2000

# use true parameters as guess to test
param = 4       #which parameter set to use

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

#df = DataFrame(load("kw94.csv"))
df = DataFrame(load("output/df4_MC100000.csv"))

df.choice = Array{Int64}(undef, size(df,1))
for i = 1:size(df,1)
        if df.school_c[i] == 1
                df.choice[i] = 3
        elseif df.work1c[i] == 1
                df.choice[i] = 1
        elseif df.work2c[i] == 1
                df.choice[i] = 2
        else
                df.choice[i] = 4
        end
end
df.id = df.idx

ApproxS = 200
lambda = 0.5

@time likelihood(df, Draws, Domain_set, lambda, θ)

function wrapll(θ)
    return likelihood(df, Draws, Domain_set, lambda, θ)
end



#########################################################
#garbage

@time mini = optimize(wrapll, Optim.minimizer(mini), Optim.Options(store_trace = true,show_trace=true))


func = TwiceDifferentiable(wrapll, θ, autodiff = :forward)
@time mini = optimize(func, θ, Newton(), Optim.Options(store_trace = true,show_trace=true))
#17hours

func1 = OnceDifferentiable(wrapll, θ; autodiff = :forward)
@time mini1 = optimize(func1, θ, LBFGS(), Optim.Options(store_trace = true,show_trace=true))

using Pkg
Pkg.add("LineSearches")
using LineSearches

func3 = TwiceDifferentiable(wrapll, θ; autodiff = :forward)
@time mini3 = optimize(func3, θ, Newton(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.MoreThuente()), Optim.Options(store_trace = true,show_trace=true))

θ1 = θ .- 0.5 .* rand(26) .- 0.25
func4 = OnceDifferentiable(wrapll, θ1; autodiff = :forward)
@time mini4 = optimize(func4, θ1, LBFGS(;linesearch = BackTracking()), Optim.Options(store_trace = true,show_trace=true))

func3 = TwiceDifferentiable(wrapll, θ; autodiff = :forward)
@time mini3 = optimize(func3, θ, Newton(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.MoreThuente()), Optim.Options(store_trace = true,show_trace=true))

@time mini5 = optimize(func4, θ1, Optim.Options(store_trace = true,show_trace=true))

save("minimization2.jld", "mini2")

Optim.minimizer(mini4)
Optim.minimizer(mini4) .- θ
