using Random
using Distributions
using LinearAlgebra
using CSVFiles

mutable struct Parameters{T <: Real}
    α10::T;α11::T;α12::T;α13::T;α14::T;α15::T;
    α20::T;α21::T;α22::T;α23::T;α24::T;α25::T;
    β0::T;β1::T;β2::T;
    γ0::T;
    σ11::T;σ12::T;σ13::T;σ14::T;
    σ22::T;σ23::T;σ24::T;
    σ33::T;σ34::T;
    σ44::T;
    β::T;
    T::T;
end

p = Parameters(9.21,0.038,0.033,0.0005,0.0,0.0,
            8.48,0.07,0.067,0.001,0.022,0.0005,
            0.0,0.0,4000.0,
            17750.0,
            0.2,0.0,0.0,0.0,
            0.25,0.0,0.0,
            1500.0,0.0,
            1500.0,
            0.95,
            40.0)


include("functions.jl")

Random.seed!(1)

n = 1000
T = 40
S = 100000
mu = [0, 0, 0, 0]
sigma = diagm(0 => [p.σ11, p.σ22, p.σ33, p.σ44])
MC_ϵ = rand(MvNormal(mu, sigma),S)
epsilon = rand(MvNormal(mu, sigma),S)


function EmaxT(T::Int64,epsilon::Array)
    SST = Domain_set[T]
    Emax = zeros(size(SST,1),1)
    for i = 1:size(SST,1)
        state = SST[i]
        r1 = exp.(R1(state[1],state[2],state[3],epsilon[1,:]))
        r2 = exp.(R2(state[1],state[2],state[3],epsilon[2,:]))
        r3 = R3(state[1],state[4],epsilon[3,:])
        r4 = R4(epsilon[4,:])
        Emax[i] = sum(max.(r1, r2, r3, r4))/S
    end
    return fEmax = Dict(zip(SST,Emax))
end

function Emaxt(T::Int64, Domain_set::Dict, fEmax::Dict, epsilon::Array)
    SST = Domain_set[T]
    Emaxteste = zeros(size(SST,1),1)
    for i = 1:size(SST,1)
        state = SST[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = exp.(R1(state[1],state[2],state[3],epsilon[1,:])) .+ p.β*fEmax[state+[0,1,0,-state[4]]]
        v2 = exp.(R2(state[1],state[2],state[3],epsilon[2,:])) .+ p.β*fEmax[state+[0,0,1,-state[4]]]
        v3 = R3(state[1],state[4],epsilon[3,:]) .+ p.β*fEmax[d3]
        v4 = R4(epsilon[4,:]) .+ p.β*fEmax[state]
        Emaxteste[i] = sum(max.(v1, v2, v3, v4))/S
    end
    return fEmaxt = Dict(zip(SST,Emaxteste))
end


function genEmaxAll(fEmax::Dict,Domain_set::Dict,Emaxall::Dict,epsilon::Array)
    for t = reverse(2:T-1)
        fEmax = Emaxt(t, Domain_set,fEmax,epsilon)
        tempDict = Dict(t => fEmax)
        Emaxall = merge(Emaxall,tempDict)
        println(t)
    end
    return Emaxall
end


function sim(T,st,N_ϵ,Emaxall)
    choice = [0, 0, 0, 0]
    stf = st
    state = st
    for t = 2:T-1
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = exp.(R1(state[1],state[2],state[3],N_ϵ[1,t])) .+ p.β*Emaxall[t][state+[0,1,0,-state[4]]]
        v2 = exp.(R2(state[1],state[2],state[3],N_ϵ[2,t])) .+ p.β*Emaxall[t][state+[0,0,1,-state[4]]]
        v3 = R3(state[1],test[4],N_ϵ[3,t]) .+ p.β*Emaxall[t][d3]
        v4 = R4(N_ϵ[4,t]) .+ p.β*Emaxall[t][state]
        rmax = max(v1,v2,v3,v4)
        if v1 == rmax
            state = state+[0,1,0,-state[4]]
            choice = hcat(choice, [1, 0, 0, 0])
        elseif v2 == rmax
            state = state+[0,0,1,-state[4]]
            choice = hcat(choice, [0, 1, 0, 0])
        elseif v3 == rmax
            state = d3
            choice = hcat(choice, [0, 0, 1, 0])
        elseif v4 == rmax
            state = state
            choice = hcat(choice, [0, 0, 0, 1])
        end
        stf = hcat(stf, state)
    end
    state = stf[:,T-1]
    v1 = exp.(R1(state[1],state[2],state[3],N_ϵ[1,T]))
    v2 = exp.(R2(state[1],state[2],state[3],N_ϵ[2,T]))
    v3 = R3(state[1],test[4],N_ϵ[3,T])
    v4 = R4(N_ϵ[4,T])
    rmax = max(v1,v2,v3,v4)
    if v1 == rmax
        state = state+[0,1,0,-state[4]]
        choice = hcat(choice, [1, 0, 0, 0])
    elseif v2 == rmax
        state = state+[0,0,1,-state[4]]
        choice = hcat(choice, [0, 1, 0, 0])
    elseif v3 == rmax
        state = d3
        choice = hcat(choice, [0, 0, 1, 0])
    elseif v4 == rmax
        state = state
        choice = hcat(choice, [0, 0, 0, 1])
    end
    stf = hcat(stf, state)
    return stf, choice
end


@time fEmax = EmaxT(40,MC_ϵ)
Emaxall = Dict(40 => fEmax)
fEmaxT = fEmax

@time Emaxall = genEmaxAll(fEmax,Domain_set,Emaxall,MC_ϵ)




st = [10,0,0,0]

teste = Array{Array{Int64}}(undef,1000)
choice = Array{Array{Int64}}(undef,1000)

for i = 1:1000
    N_ϵ =  rand(MvNormal(mu, sigma),T)
    teste[i], choice[i] = sim(40,st,N_ϵ,Emaxall)
end



for i = 2:1000
    global testf = hcat(testf, teste[i])
    global choicef = hcat(choicef, choice[i])
end


period = repeat(collect(1:40),outer = 1000)
idx = repeat(collect(1:1000), inner = 40)

df = DataFrame(idx = idx, period=period, school = testf[1,:], exp1 = testf[2,:], exp2 = testf[3,:],
            school_c = choicef[3,:], work1 = choicef[1,:], work2 = choicef[2,:], home = choicef[4,:])

df |> save("df.csv")
