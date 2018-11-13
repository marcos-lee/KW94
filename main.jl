using Random
using Distributions
using LinearAlgebra
using CSVFiles
using DataFrames

Random.seed!(1)


include("feasibleSet.jl")
include("functions.jl")


n = 1000        #Number of people
T = 41          #I start with t =1, the paper starts with t = 0
S = 100000      #Number of MC draws

# Change to Parameter1.jl, Parameter2.jl or Parameters3.jl to use another set of parameters
include("Parameters1.jl")

mu = [0, 0, 0, 0] #Mean of ϵ
sigma = [p.σ11^2 p.σ12 p.σ13 p.σ14;
        p.σ12 p.σ22^2 p.σ23 p.σ24;
        p.σ13 p.σ23 p.σ33^2 p.σ34;
        p.σ14 p.σ24 p.σ34 p.σ44^2]

MC_ϵ = rand(MvNormal(mu, sigma),S) #Take S draws from the Multivariate Normal



# Define an auxiliary vector to calculate the feasible set
# It adds 1 to the state vector according to the action taken
# First element: go to school
# Second: Work at 1
# Third: Work at 2
# Stay at home
up = Vector{Vector{Int64}}(undef,4)

up[1] = [1,0,0,0] #
up[2] = [0,1,0,0]
up[3] = [0,0,1,0]
up[4] = [0,0,0,0]


# The notation of states is the following:
# State Space S(t) = {s(t), x1(t), x2(t), d3(t-1)}

st = [10,0,0,0] # Initial state at t=1

# Generates a dictionary, where the key is the time t,
# and the value is an Array of Arrays, with each array being a possible state point at that period
@time Domain_set = StateSpace(st, T, up)


# Full solution of the model:

# Calculates the terminal period Value Function
@time fEmax = EmaxT(T,MC_ϵ)
#about one minute


# Store it in a dictionary with key = t, value = Emax
Emaxall = Dict(T => fEmax)

# Calculates the Emax at every period t=2,...,T-1
fEmaxT = fEmax
@time Emaxall = genEmaxAll(fEmax,Domain_set,Emaxall,MC_ϵ)
#about 11-12 minutes


##########################################################################

# Simulate the model for N people
stateHistory = Array{Array{Int64}}(undef,1000)
choiceHistory = Array{Array{Int64}}(undef,1000)

for i = 1:1000
    N_ϵ =  rand(MvNormal(mu, sigma),T)
    stateHistory[i], choiceHistory[i] = SimulateModel(T,st,N_ϵ,Emaxall)
end


# Transforms output into DataFrame for easy handling
stateDF = stateHistory[1]
choiceDF = choiceHistory[1]

for i = 2:1000
    global stateDF = hcat(stateDF, stateHistory[i])
    global choiceDF = hcat(choiceDF, choiceHistory[i])
end

period = repeat(collect(1:T),outer = 1000)
idx = repeat(collect(1:1000), inner = T)
df = DataFrame(idx = idx, period=period, school = stateDF[1,:], exp1 = stateDF[2,:], exp2 = stateDF[3,:],
            school_c = choiceDF[3,:], work1 = choiceDF[1,:], work2 = choiceDF[2,:], home = choiceDF[4,:])

# very bad way of saving different file names according to which parameter set I am using

if p.σ44 == 1500.0
    df |> save("df1.csv")
elseif p.σ44 == 6000.0
    df |> save("df2.csv")
elseif p.σ44 == 8500.0
    df |> save("df3.csv")
end
