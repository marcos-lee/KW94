using Random
using Distributions
using LinearAlgebra
using CSVFiles
using DataFrames
using DelimitedFiles

# This code heavily relies on Dictionaries. In Julia (and maybe elsewhere)
# dictionary entries are randomly ordered. In the full solution problem,
# this causes no problem since dictionaries are created once and that is it.
# However, in the approximation routine, it causes a lot of problems since there
# is a lot of extracting, changes and rebuilding of dictionaries. Therefore, I
# need to rely on OrderedDict structures, which as implied by name, preserves order.
# Why does Julia doesn't do this as standard is unknown to me. Faster to randomly assign, perhaps?
import DataStructures.OrderedDict


include("feasibleSet.jl")
include("functions.jl")


N = 1000        #Number of people
T = 40          #I start with t =1, the paper starts with t = 0
#MC = 1000     #Number of MC draws

param = 1       #which parameter set to use

# Change to Parameter1.jl, Parameter2.jl or Parameters3.jl to use another set of parameters
include("Parameters$param.jl")
st = [10,0,0,1] # Initial state at t=1
include("maxe.jl")


import StatsBase.sample

# Generates a dictionary, where the key is the time t,
# and the value is an Array of Arrays, with each array being a possible state point at that period
@time Domain_set, tStateSpace = @timed StateSpace(st, T)


Random.seed!(10)
N_ϵ = Vector{Array{Float64,2}}(undef,N)
for i = 1:N
    N_ϵ[i] = rand(MvNormal(mu, sigma),T)
end



iter = [100000 2000 1000 250]
iter2 = [2000 500]
for i = iter
    println("\n Solving for MC draws = $i \n")
    global MC = i
    mu = [0, 0, 0, 0] #Mean of ϵ
    sigma = [p.σ11^2 p.σ12 p.σ13 p.σ14;
            p.σ12 p.σ22^2 p.σ23 p.σ24;
            p.σ13 p.σ23 p.σ33^2 p.σ34;
            p.σ14 p.σ24 p.σ34 p.σ44^2]

    Random.seed!(10)
    MC_ϵ = rand(MvNormal(mu, sigma),MC) #Take S draws from the Multivariate Normal
    benchDraws(MC_ϵ, Domain_set)
    if i == 2000
        for j = iter2
            println("\n Approximating for $j state points\n")
            global ApproxS = j
            benchApprox(MC_ϵ, Domain_set, ApproxS)
        end
    end
end


#Old version
#=
N = 1000        #Number of people
T = 40          #I start with t =1, the paper starts with t = 0
MC = 2000     #Number of MC draws

param = 1       #which parameter set to use

# Change to Parameter1.jl, Parameter2.jl or Parameters3.jl to use another set of parameters
include("Parameters$param.jl")



mu = [0, 0, 0, 0] #Mean of ϵ
sigma = [p.σ11^2 p.σ12 p.σ13 p.σ14;
        p.σ12 p.σ22^2 p.σ23 p.σ24;
        p.σ13 p.σ23 p.σ33^2 p.σ34;
        p.σ14 p.σ24 p.σ34 p.σ44^2]

Random.seed!(10)
MC_ϵ = rand(MvNormal(mu, sigma),MC) #Take S draws from the Multivariate Normal

N_ϵ = Vector{Array{Float64,2}}(undef,N)
for i = 1:N
    N_ϵ[i] = rand(MvNormal(mu, sigma),T)
end


# The notation of states is the following:
# State Space S(t) = {s(t), x1(t), x2(t), d3(t-1)}

st = [10,0,0,1] # Initial state at t=1

# Generates a dictionary, where the key is the time t,
# and the value is an Array of Arrays, with each array being a possible state point at that period
@time Domain_set, tStateSpace = @timed StateSpace(st, T)

# Full solution of the model:
@time Emaxall, timeEmax = genEmaxAll(Domain_set,MC_ϵ, T)
#about 11-12 minutes when using 100k MC draws
writedlm("output/timeEmax$(param)_MC$MC.txt", timeEmax)
##########################################################################

# Simulate the model for N people

df = SimulateAll(N, T, N_ϵ, Emaxall)


df |> save("output/df$(param)_MC$MC.csv")



Ch = by(df, :period) do x
    DataFrame(avgschool = mean(x.school_c), avgw1 = mean(x.work1), avgw2 = mean(x.work2), avghome = mean(x.home))
end



##########################################################################
# Use approximation method described in Keane and Wolpin 1994

include("maxe.jl")


import StatsBase.sample
ApproxS = 2000
st = [10,0,0,1] # Initial state at t=1


# solves for Emax using the approximation
@time Emaxallhat, timeEmaxhat = genEmaxAllHat(Domain_set, ApproxS)
writedlm("output/timeEmaxhat$(param)_MC$(MC)_S$(ApproxS).txt", timeEmaxhat)


# simulates the model using the same draws from the distribution of errors
df1 = SimulateAll(N, T, N_ϵ, Emaxallhat)

df1 |> save("output/df$(param)_MC$(MC)_S$(ApproxS).csv")


Ch1 = by(df1, :period) do x
    DataFrame(avgschool = mean(x.school_c), avgw1 = mean(x.work1), avgw2 = mean(x.work2), avghome = mean(x.home))
end
=#
