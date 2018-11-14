using Random
using Distributions
using LinearAlgebra
using CSVFiles
using DataFrames


include("feasibleSet.jl")
#include("functions.jl")
include("functions_alt.jl")


n = 1000        #Number of people
T = 40          #I start with t =1, the paper starts with t = 0
S = 1000      #Number of MC draws

# Change to Parameter1.jl, Parameter2.jl or Parameters3.jl to use another set of parameters
include("Parameters1.jl")

mu = [0, 0, 0, 0] #Mean of ϵ
sigma = [p.σ11^2 p.σ12 p.σ13 p.σ14;
        p.σ12 p.σ22^2 p.σ23 p.σ24;
        p.σ13 p.σ23 p.σ33^2 p.σ34;
        p.σ14 p.σ24 p.σ34 p.σ44^2]
Random.seed!(123)
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
@time Emaxall = genEmaxAll(Domain_set,MC_ϵ, T)
#about 11-12 minutes

Emaxall[2]

##########################################################################

# Simulate the model for N people
stateHistory = Array{Array{Int64}}(undef,1000)
choiceHistory = Array{Array{Int64}}(undef,1000)


Random.seed!(1)
N_ϵ = Vector{Array{Float64,2}}(undef,1000)
for i = 1:1000
    N_ϵ[i] = rand(MvNormal(mu, sigma),T)
end

stateHistory = Array{Array{Int64}}(undef,1000)
choiceHistory = Array{Array{Int64}}(undef,1000)

for i = 1:1000
    stateHistory[i], choiceHistory[i] = SimulateModel(T,st,N_ϵ[i],Emaxall)
end

# Transforms output into DataFrame for easy handling
stateDF = stateHistory[2]
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

Ch = by(df, :period) do x
    DataFrame(avgschool = mean(x.school_c), avgw1 = mean(x.work1), avgw2 = mean(x.work2), avghome = mean(x.home))
end
Ch


N_ϵ[2]

N_ϵ1= N_ϵ[2]


choice = Array{Int64}(undef, 4, 40)
stf = Array{Int64}(undef, 4, 40)
state = st

d3 = state + [1,0,0,1-state[4]]
d3[1] = min(20,d3[1])
v1 = exp.(R1(state[1],state[2],state[3],N_ϵ1[1,t])) .+ p.β*Emaxall[t+1][state+[0,1,0,-state[4]]]
v2 = exp.(R2(state[1],state[2],state[3],N_ϵ1[2,t])) .+ p.β*Emaxall[t+1][state+[0,0,1,-state[4]]]
v3 = R3(state[1],state[4],N_ϵ1[3,t]) .+ p.β*Emaxall[t+1][d3]
v4 = R4(N_ϵ1[4,t]) .+ p.β*Emaxall[t+1][st]
rmax = max(v1,v2,v3,v4)
if v1 == rmax
    state = state+[0,1,0,-state[4]]
    choice[:,t] = [1, 0, 0, 0]
elseif v2 == rmax
    state = state+[0,0,1,-state[4]]
    choice[:,t] = [0, 1, 0, 0]
elseif v3 == rmax
    state = d3
    choice[:,t] = [0, 0, 1, 0]
elseif v4 == rmax
    state = state
    choice[:,t] = [0, 0, 0, 1]
end
stf[:,t] = state
