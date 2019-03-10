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
# SST = State Space at T
function EmaxT(SST::Array, MC_ϵ::Array)
    Emax = zeros(size(SST,1),1)
    for i = 1:size(SST,1)
        state = SST[i]
        r1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:]))
        r2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:]))
        r3 = R3(state[1],state[4],MC_ϵ[3,:])
        r4 = R4(MC_ϵ[4,:])
        Emax[i] = mean(max.(r1, r2, r3, r4))
    end
    return fEmax = OrderedDict(zip(SST,Emax))
end


# Calculates Emax at t=2,...,T-1
# SSt = State Space at t
function Emaxt(SSt::Array, fEmax::OrderedDict, MC_ϵ::Array)
    Emax = zeros(size(SSt,1),1)
    for i = 1:size(SSt,1)
        state = SSt[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:])) .+ p.β*fEmax[state+[0,1,0,-state[4]]]
        v2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:])) .+ p.β*fEmax[state+[0,0,1,-state[4]]]
        if d3[1] < 20
            v3 = R3(state[1],state[4],MC_ϵ[3,:]) .+ p.β*fEmax[d3]
        else
            v3 = R3(state[1],state[4],MC_ϵ[3,:])
        end
        v4 = R4(MC_ϵ[4,:]) .+ p.β*fEmax[state+[0,0,0,-state[4]]]
        Emax[i] = mean(max.(v1, v2, v3, v4))
    end
    return fEmaxt = OrderedDict(zip(SSt,Emax))
end

# Combines both together
function genEmaxAll(Domain_set::OrderedDict, MC_ϵ::Array, T)
    println("\n Backward induction \n")
    println("\n Solving Exact Model \n")
    println("== Iteration t=$T ==\n")
    @time fEmax, tEmax = @timed EmaxT(Domain_set[T], MC_ϵ)
    Emaxall = OrderedDict(T => fEmax)
    timeEmax = Array{Float64}(undef, T-1, 2)
    timeEmax[T-1,:] = [T tEmax]
    for t = reverse(2:T-1)
        println("== Iteration t=$t ==\n")
        @time fEmax, tEmax = @timed Emaxt(Domain_set[t], fEmax, MC_ϵ)
        timeEmax[t,:] = [t tEmax]
        tempDict = OrderedDict(t => fEmax)
        Emaxall = merge(Emaxall,tempDict)
    end
    return Emaxall, timeEmax
end

# Simulates one person in the model
function SimulateModel(T, st, N_ϵ, Emaxall::OrderedDict)
    choice = Array{Int64}(undef, 4, T)
    stf = Array{Int64}(undef, 4, T)
    state = st
    wage =  Array{Float64}(undef, T)
    for t = 1:T-1
        stf[:,t] = state
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = exp.(R1(state[1],state[2],state[3],N_ϵ[1,t])) .+ p.β*Emaxall[t+1][state+[0,1,0,-state[4]]]
        v2 = exp.(R2(state[1],state[2],state[3],N_ϵ[2,t])) .+ p.β*Emaxall[t+1][state+[0,0,1,-state[4]]]
        v3 = R3(state[1],state[4],N_ϵ[3,t]) .+ p.β*Emaxall[t+1][d3]
        v4 = R4(N_ϵ[4,t]) .+ p.β*Emaxall[t+1][state+[0,0,0,-state[4]]]
        rmax = max(v1,v2,v3,v4)
        if v1 == rmax
            wage[t] = exp.(R1(state[1],state[2],state[3],N_ϵ[1,t]))
            state = state + [0,1,0,-state[4]]
            choice[:,t] = [1, 0, 0, 0]
        elseif v2 == rmax
            wage[t] = exp.(R2(state[1],state[2],state[3],N_ϵ[2,t]))
            state = state + [0,0,1,-state[4]]
            choice[:,t] = [0, 1, 0, 0]
        elseif v3 == rmax
            wage[t] = R3(state[1],state[4],N_ϵ[3,t])
            state = d3
            choice[:,t] = [0, 0, 1, 0]
        elseif v4 == rmax
            wage[t] = R4(N_ϵ[4,t])
            state = state + [0,0,0,-state[4]]
            choice[:,t] = [0, 0, 0, 1]
        end
    end
    stf[:,T] = state
    r1 = exp.(R1(state[1],state[2],state[3],N_ϵ[1,T]))
    r2 = exp.(R2(state[1],state[2],state[3],N_ϵ[2,T]))
    r3 = R3(state[1],state[4],N_ϵ[3,T])
    r4 = R4(N_ϵ[4,T])
    rmax = max(r1,r2,r3,r4)
    if r1 == rmax
        wage[T] = exp.(R1(state[1],state[2],state[3],N_ϵ[1,T]))
        state = state + [0,1,0,-state[4]]
        choice[:,T] = [1, 0, 0, 0]
    elseif r2 == rmax
        wage[T] = exp.(R2(state[1],state[2],state[3],N_ϵ[2,T]))
        state = state + [0,0,1,-state[4]]
        choice[:,T] = [0, 1, 0, 0]
    elseif r3 == rmax
        wage[T] = R3(state[1],state[4],N_ϵ[3,T])
        state = state + [1,0,0,1-state[4]]
        choice[:,T] = [0, 0, 1, 0]
    elseif r4 == rmax
        wage[T] = R4(N_ϵ[4,T])
        state = state + [0,0,0,-state[4]]
        choice[:,T] = [0, 0, 0, 1]
    end
    return stf, choice, wage
end

function SimulateAll(N::Int64, T::Int64, N_ϵ::Array, Emax::OrderedDict)
    stateHistory = Array{Array{Int64}}(undef,N)
    choiceHistory = Array{Array{Int64}}(undef,N)
    wageHistory = Array{Float64}(undef,N, T)
    st = [10, 0, 0, 1]
    for i = 1:N
        stateHistory[i], choiceHistory[i], wageHistory[i,:] = SimulateModel(T, st, N_ϵ[i], Emax)
    end
    # Transforms output into DataFrame for easy handling
    stateDF = stateHistory[1]
    choiceDF = choiceHistory[1]
    wageDF = wageHistory[1, :]
    for i = 2:N
        stateDF = hcat(stateDF, stateHistory[i])
        choiceDF = hcat(choiceDF, choiceHistory[i])
        wageDF = vcat(wageDF, wageHistory[i,:])
    end
    period = repeat(collect(1:T),outer = N)
    idx = repeat(collect(1:N), inner = T)
    return df = DataFrame(idx = idx, period=period, educ = stateDF[1,:], work1 = stateDF[2,:], work2 = stateDF[3,:], lag = stateDF[4,:],
                school_c = choiceDF[3,:], work1c = choiceDF[1,:], work2c = choiceDF[2,:], home = choiceDF[4,:], wage = wageDF)
end


function benchDraws(MC_ϵ, Domain_set)
    @time Emaxall, timeEmax = genEmaxAll(Domain_set,MC_ϵ, T)
    #about 11-12 minutes when using 100k MC draws
    writedlm("output/timeEmax$(param)_MC$MC.txt", timeEmax)
    ##########################################################################
    # Simulate the model for N people
    df = SimulateAll(N, T, N_ϵ, Emaxall)
    df |> save("output/df$(param)_MC$MC.csv")
end

function benchApprox(MC_ϵ, Domain_set, ApproxS)
    @time Emaxallhat, timeEmaxhat = genEmaxAllHat(Domain_set, ApproxS)
    writedlm("output/timeEmaxhat$(param)_MC$(MC)_S$(ApproxS).txt", timeEmaxhat)
    # simulates the model using the same draws from the distribution of errors
    df1 = SimulateAll(N, T, N_ϵ, Emaxallhat)
    df1 |> save("output/df$(param)_MC$(MC)_S$(ApproxS).csv")
end
