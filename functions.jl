function R1(s,x1,x2,ϵ1)
    r1 = p.α10 + p.α11 * s + p.α12 * x1 - p.α13 * x1^2 + p.α14 *x2 - p.α15 *x2^2 .+ ϵ1
end

function R2(s,x1,x2,ϵ2)
    r2 = p.α20 + p.α21 * s + p.α22 * x2 - p.α23 * x2^2 + p.α24 *x1 - p.α25 *x1^2 .+ ϵ2
end

function R3(s,slag,ϵ3)
    r3 = p.β0 - p.β1 * (s >= 12) - p.β2 * (1 - slag) .+ ϵ3
end

function R4(ϵ4)
    r4 = p.γ0 .+ ϵ4
end


# Calculates Emax at terminal period
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

# Calculates Emax at t=2,...,T-1
function Emaxt(T::Int64, Domain_set::Dict, fEmax::Dict, epsilon::Array)
    SST = Domain_set[T]
    Emax = zeros(size(SST,1),1)
    for i = 2:size(SST,1)
        state = SST[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = exp.(R1(state[1],state[2],state[3],epsilon[1,:])) .+ p.β*fEmax[state+[0,1,0,-state[4]]]
        v2 = exp.(R2(state[1],state[2],state[3],epsilon[2,:])) .+ p.β*fEmax[state+[0,0,1,-state[4]]]
        v3 = R3(state[1],state[4],epsilon[3,:]) .+ p.β*fEmax[d3]
        v4 = R4(epsilon[4,:]) .+ p.β*fEmax[state]
        Emax[i] = sum(max.(v1, v2, v3, v4))/S
    end
    return fEmaxt = Dict(zip(SST,Emax))
end

# Combines both together
function genEmaxAll(fEmax::Dict,Domain_set::Dict,Emaxall::Dict,epsilon::Array)
    for t = reverse(2:T-1)
        fEmax = Emaxt(t, Domain_set,fEmax,epsilon)
        tempDict = Dict(t => fEmax)
        Emaxall = merge(Emaxall,tempDict)
        println(t)
    end
    return Emaxall
end

# Simulates one person in the model
function SimulateModel(T,st,N_ϵ,Emaxall)
    choice = [0,0,0,0]
    stf = st
    state = st
    for t = 2:T-1
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = exp.(R1(state[1],state[2],state[3],N_ϵ[1,t])) .+ p.β*Emaxall[t][state+[0,1,0,-state[4]]]
        v2 = exp.(R2(state[1],state[2],state[3],N_ϵ[2,t])) .+ p.β*Emaxall[t][state+[0,0,1,-state[4]]]
        v3 = R3(state[1],state[4],N_ϵ[3,t]) .+ p.β*Emaxall[t][d3]
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
    r1 = exp.(R1(state[1],state[2],state[3],N_ϵ[1,T]))
    r2 = exp.(R2(state[1],state[2],state[3],N_ϵ[2,T]))
    r3 = R3(state[1],state[4],N_ϵ[3,T])
    r4 = R4(N_ϵ[4,T])
    rmax = max(r1,r2,r3,r4)
    if r1 == rmax
        state = state+[0,1,0,-state[4]]
        choice = hcat(choice, [1, 0, 0, 0])
    elseif r2 == rmax
        state = state+[0,0,1,-state[4]]
        choice = hcat(choice, [0, 1, 0, 0])
    elseif r3 == rmax
        state = d3
        choice = hcat(choice, [0, 0, 1, 0])
    elseif r4 == rmax
        state = state
        choice = hcat(choice, [0, 0, 0, 1])
    end
    stf = hcat(stf, state)
    return stf, choice
end
