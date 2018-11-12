using Random
using Distributions
using LinearAlgebra
using Combinatorics

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
sigma = diagm(0 => [p.σ11^2, p.σ22^2, p.σ33^2, p.σ44^2])
epsilon = rand(MvNormal(mu, sigma),S)


fstate = Dict(zip(Domain_set[40],collect(1:size(Domain_set[40],1))))

epsilon = rand(MvNormal(mu, sigma),S)
function EmaxT(T::Int64,epsilon::Array)
    SST = Domain_set[T]
    Emax = zeros(size(SST,1),1)
    for i = 1:size(SST,1)
        state = SST[i]
        r1 = exp.(R1(state[1],state[2],state[3],epsilon[1,:]))
        r2 = exp.(R2(state[1],state[2],state[3],epsilon[2,:]))
        r3 = R3(state[1],test[4],epsilon[3,:])
        r4 = R4(epsilon[4,:])
        Emax[i] = sum(max.(r1, r2, r3, r4))/S
    end
    return fEmax = Dict(zip(SST,Emax))
end

@time fEmax = EmaxT(40,epsilon)
Emaxall = Dict(40 => fEmax)

function Emaxt(T::Int64, Domain_set, fEmax)
    SST = Domain_set[T]
    Emaxteste = zeros(size(SST,1),1)
    epsilon = rand(MvNormal(mu, sigma),S)
    for i = 1:size(SST,1)
        state = SST[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        r1 = exp.(R1(state[1],state[2],state[3],epsilon[1,:])) .+fEmax[state+[0,1,0,-state[4]]]
        r2 = exp.(R2(state[1],state[2],state[3],epsilon[2,:])) .+fEmax[state+[0,0,1,-state[4]]]
        r3 = R3(state[1],test[4],epsilon[3,:]) .+fEmax[d3]
        r4 = R4(epsilon[4,:]) .+fEmax[state]
        Emaxteste[i] = sum(max.(r1, r2, r3, r4))/S
    end
    return fEmaxt = Dict(zip(SST,Emaxteste))
end


function genEmaxAll(fEmax,Domain_set,Emaxall)
    for t = reverse(2:T-1)
        fEmax = Emaxt(t, Domain_set,fEmax)
        tempDict = Dict(t => fEmax)
        Emaxall = merge(Emaxall,tempDict)
    end
    return Emaxall
end

@time fEmax = EmaxT(40,epsilon)
Emaxall = Dict(40 => fEmax)
fEmaxT = fEmax

@time Emaxall = genEmaxAll(fEmax,Domain_set,Emaxall)
