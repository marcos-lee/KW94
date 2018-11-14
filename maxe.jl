# Calculates Emax at terminal period
function maxET(T::Int64,epsilon::Array)
    SST = Domain_set[T]
    maxE = zeros(size(SST,1),1)
    vbar = zeros(size(SST,1),4)
    for i = 1:size(SST,1)
        state = SST[i]
        r1 = sum(exp.(R1(state[1],state[2],state[3],epsilon[1,:])))/S
        r2 = sum(exp.(R2(state[1],state[2],state[3],epsilon[2,:])))/S
        r3 = sum(R3(state[1],state[4],epsilon[3,:]))/S
        r4 = sum(R4(epsilon[4,:]))/S
        maxE[i] = max(r1, r2, r3, r4)
        vbar[i,:] = [r1 r2 r3 r4]
    end
    fmaxE = Dict(zip(SST,maxE))
    Ev1 = Dict(zip(SST,vbar[:,1]))
    Ev2 = Dict(zip(SST,vbar[:,2]))
    Ev3 = Dict(zip(SST,vbar[:,3]))
    Ev4 = Dict(zip(SST,vbar[:,4]))
    return fmaxE, Ev1, Ev2, Ev3, Ev4
end

# Calculates Emax at t=2,...,T-1
function maxEt(T::Int64, Domain_set::Dict, Emaxall::Dict, epsilon::Array)
    SST = Domain_set[T]
    maxE = zeros(size(SST,1),1)
    vbar = zeros(size(SST,1),4)
    for i = 1:size(SST,1)
        state = SST[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = sum(exp.(R1(state[1],state[2],state[3],epsilon[1,:])) .+ p.β*Emaxall[T+1][state+[0,1,0,-state[4]]])/S
        v2 = sum(exp.(R2(state[1],state[2],state[3],epsilon[2,:])) .+ p.β*Emaxall[T+1][state+[0,0,1,-state[4]]])/S
        v3 = sum(R3(state[1],state[4],epsilon[3,:]) .+ p.β*Emaxall[T+1][d3])/S
        v4 = sum(R4(epsilon[4,:]) .+ p.β*Emaxall[T+1][state])/S
        maxE[i] = max(v1, v2, v3, v4)
        vbar[i,:] = [v1 v2 v3 v4]
    end
    fmaxEt = Dict(zip(SST,maxE))
    Ev1 = Dict(zip(SST,vbar[:,1]))
    Ev2 = Dict(zip(SST,vbar[:,2]))
    Ev3 = Dict(zip(SST,vbar[:,3]))
    Ev4 = Dict(zip(SST,vbar[:,4]))
    return fmaxEt, Ev1, Ev2, Ev3, Ev4
end

# Combines both together
function genmaxEAll(Emaxall::Dict,Domain_set::Dict,epsilon::Array)
    fmaxE, Ev1, Ev2, Ev3, Ev4 = maxET(T,MC_ϵ)
    # Store it in a dictionary with key = t, value = Emax
    maxEall = Dict(T => fmaxE)
    Ev1all = Dict(T => Ev1)
    Ev2all = Dict(T => Ev2)
    Ev3all = Dict(T => Ev3)
    Ev4all = Dict(T => Ev4)
    for t = reverse(2:T-1)
        fmaxE, Ev1, Ev2, Ev3, Ev4 = maxEt(t, Domain_set, Emaxall, MC_ϵ)
        tempDict = Dict(t => fmaxE)
        tempDict1 = Dict(t => Ev1)
        tempDict2 = Dict(t => Ev2)
        tempDict3 = Dict(t => Ev3)
        tempDict4 = Dict(t => Ev4)
        maxEall = merge(maxEall,tempDict)
        Ev1all = merge(Ev1all,tempDict1)
        Ev2all = merge(Ev2all,tempDict2)
        Ev3all = merge(Ev3all,tempDict3)
        Ev4all = merge(Ev4all,tempDict4)
        println(t)
    end
    return maxEall, Ev1all, Ev2all, Ev3all, Ev4all
end


@time maxEall, Ev1all, Ev2all, Ev3all, Ev4all = genmaxEAll(Emaxall,Domain_set,MC_ϵ)

maxEall[40]


function OLS()
