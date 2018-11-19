using Random
using Distributions
using LinearAlgebra
using CSVFiles
using DataFrames
using DelimitedFiles
import DataStructures.OrderedDict


include("feasibleSet.jl")
#include("functions.jl")
include("functions_alt.jl")


N = 1000        #Number of people
T = 40          #I start with t =1, the paper starts with t = 0
S = 2000      #Number of MC draws

param = 1       #which parameter set to use

# Change to Parameter1.jl, Parameter2.jl or Parameters3.jl to use another set of parameters
include("Parameters$param.jl")

mu = [0, 0, 0, 0] #Mean of ϵ
sigma = [p.σ11^2 p.σ12 p.σ13 p.σ14;
        p.σ12 p.σ22^2 p.σ23 p.σ24;
        p.σ13 p.σ23 p.σ33^2 p.σ34;
        p.σ14 p.σ24 p.σ34 p.σ44^2]

#Random.seed!(1)
MC_ϵ = rand(MvNormal(mu, sigma),S) #Take S draws from the Multivariate Normal

# The notation of states is the following:
# State Space S(t) = {s(t), x1(t), x2(t), d3(t-1)}

st = [10,0,0,1] # Initial state at t=1

# Generates a dictionary, where the key is the time t,
# and the value is an Array of Arrays, with each array being a possible state point at that period
@time Domain_set, tStateSpace = @timed StateSpace(st, T)

# Full solution of the model:
@time Emaxall, timeEmax = genEmaxAll(Domain_set,MC_ϵ, T)
#about 11-12 minutes
writedlm("output/timeEmax$(param)_MC$S.txt", timeEmax)
##########################################################################

# Simulate the model for N people
N_ϵ = Vector{Array{Float64,2}}(undef,N)
for i = 1:N
    N_ϵ[i] = rand(MvNormal(mu, sigma),T)
end
df = SimulateAll(N, T, N_ϵ, Emaxall)


df |> save("output/df$(param)_MC$S.csv")



Ch = by(df, :period) do x
    DataFrame(avgschool = mean(x.school_c), avgw1 = mean(x.work1), avgw2 = mean(x.work2), avghome = mean(x.home))
end
