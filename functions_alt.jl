function R1(s,x1,x2,ϵ1)
    r1 = p.α10 + p.α11 * s + p.α12 * x1 - p.α13 * (x1^2) + p.α14 *x2 - p.α15 *(x2^2) .+ ϵ1
end

function R2(s,x1,x2,ϵ2)
    r2 = p.α20 + p.α21 * s + p.α22 * x2 - p.α23 * (x2^2) + p.α24 *x1 - p.α25 *(x1^2) .+ ϵ2
end

function R3(s,slag,ϵ3)
    if s <= 19
        r3 = p.β0 - p.β1 * (s >= 12) - p.β2 * (1 - slag) .+ ϵ3
    else
        r3 = - p.β2 .+ ϵ3 .- 40000
    end
end

function R4(ϵ4)
    r4 = p.γ0 .+ ϵ4
end

# Calculates Emax at terminal period
function EmaxT(T::Int64, MC_ϵ::Array, Domain_set::Array)
    SST = Domain_set
    Emax = zeros(size(SST,1),1)
    for i = 1:size(SST,1)
        state = SST[i]
        r1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:]))
        r2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:]))
        r3 = R3(state[1],state[4],MC_ϵ[3,:])
        r4 = R4(MC_ϵ[4,:])
        Emax[i] = mean(max.(r1, r2, r3, r4))
    end
    return fEmax = Dict(zip(SST,Emax))
end

# Calculates Emax at t=2,...,T-1
function Emaxt(SSt::Array, fEmax::Dict, MC_ϵ::Array)
    Emax = zeros(size(SSt,1),1)
    for i = 1:size(SSt,1)
        state = SSt[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:])) .+ p.β*fEmax[state+[0,1,0,-state[4]]]
        v2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:])) .+ p.β*fEmax[state+[0,0,1,-state[4]]]
        v3 = R3(state[1],state[4],MC_ϵ[3,:]) .+ p.β*fEmax[d3]
        v4 = R4(MC_ϵ[4,:]) .+ p.β*fEmax[state+[0,0,0,-state[4]]]
        Emax[i] = mean(max.(v1, v2, v3, v4))
    end
    return fEmaxt = Dict(zip(SSt,Emax))
end
# Combines both together
function genEmaxAll(Domain_set::Dict, MC_ϵ::Array, T)
    println(T)
    @time fEmax, tEmax = @timed EmaxT(T,MC_ϵ, Domain_set[T])
    Emaxall = Dict(T => fEmax)
    timeEmax = Array{Float64}(undef, 40, 2)
    timeEmax[40,:] = [40 tEmax]
    for t = reverse(2:T-1)
        println(t)
        @time fEmax, tEmax = @timed Emaxt(Domain_set[t], fEmax, MC_ϵ)
        timeEmax[t,:] = [t tEmax]
        tempDict = Dict(t => fEmax)
        Emaxall = merge(Emaxall,tempDict)
    end
    return Emaxall, timeEmax
end

# Simulates one person in the model
function SimulateModel(T,st,N_ϵ,Emaxall)
    choice = Array{Int64}(undef, 4, 40)
    stf = Array{Int64}(undef, 4, 40)
    state = st
    for t = 1:T-1
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = exp.(R1(state[1],state[2],state[3],N_ϵ[1,t])) .+ p.β*Emaxall[t+1][state+[0,1,0,-state[4]]]
        v2 = exp.(R2(state[1],state[2],state[3],N_ϵ[2,t])) .+ p.β*Emaxall[t+1][state+[0,0,1,-state[4]]]
        v3 = R3(state[1],state[4],N_ϵ[3,t]) .+ p.β*Emaxall[t+1][d3]
        v4 = R4(N_ϵ[4,t]) .+ p.β*Emaxall[t+1][state+[0,0,0,-state[4]]]
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
            state = state + [0,0,0,-state[4]]
            choice[:,t] = [0, 0, 0, 1]
        end
        stf[:,t] = state
    end
    state = stf[:,T-1]
    r1 = exp.(R1(state[1],state[2],state[3],N_ϵ[1,T]))
    r2 = exp.(R2(state[1],state[2],state[3],N_ϵ[2,T]))
    r3 = R3(state[1],state[4],N_ϵ[3,T])
    r4 = R4(N_ϵ[4,T])
    rmax = max(r1,r2,r3,r4)
    if r1 == rmax
        state = state+[0,1,0,-state[4]]
        choice[:,T] = [1, 0, 0, 0]
    elseif r2 == rmax
        state = state+[0,0,1,-state[4]]
        choice[:,T] = [0, 1, 0, 0]
    elseif r3 == rmax
        d3 = state + [1,0,0,1-state[4]]
        choice[:,T] = [0, 0, 1, 0]
    elseif r4 == rmax
        state = state + [0,0,0,-state[4]]
        choice[:,T] = [0, 0, 0, 1]
    end
    stf[:,T] = state
    return stf, choice
end

function SimulateAll(N::Int64, T::Int64, N_ϵ::Array, Emaxall::Dict)
    stateHistory = Array{Array{Int64}}(undef,N)
    choiceHistory = Array{Array{Int64}}(undef,N)
    for i = 1:N
        stateHistory[i], choiceHistory[i] = SimulateModel(T,st,N_ϵ[i],Emaxall)
    end
    # Transforms output into DataFrame for easy handling
    stateDF = stateHistory[1]
    choiceDF = choiceHistory[1]
    for i = 2:N
        stateDF = hcat(stateDF, stateHistory[i])
        choiceDF = hcat(choiceDF, choiceHistory[i])
    end
    period = repeat(collect(1:T),outer = N)
    idx = repeat(collect(1:N), inner = T)
    return df = DataFrame(idx = idx, period=period, school = stateDF[1,:], exp1 = stateDF[2,:], exp2 = stateDF[3,:],
                school_c = choiceDF[3,:], work1 = choiceDF[1,:], work2 = choiceDF[2,:], home = choiceDF[4,:])
end