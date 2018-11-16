# Calculates Emax at terminal period
function maxET(epsilon::Array, Domain_set)
    SST = Domain_set
    maxE = zeros(size(SST,1),1)
    vbar = zeros(size(SST,1),4)
    for i = 1:size(SST,1)
        state = SST[i]
        r1 = mean(exp.(R1(state[1],state[2],state[3],epsilon[1,:])))
        r2 = mean(exp.(R2(state[1],state[2],state[3],epsilon[2,:])))
        r3 = R3(state[1],state[4],0.0)
        r4 = R4(0.0)
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
function maxEt(Domain_set::Array, Emaxall::Dict, epsilon::Array)
    SST = Domain_set
    maxE = zeros(size(SST,1),1)
    vbar = zeros(size(SST,1),4)
    for i = 1:size(SST,1)
        state = SST[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = mean(exp.(R1(state[1],state[2],state[3],epsilon[1,:])) .+ p.β*Emaxall[state+[0,1,0,-state[4]]])
        v2 = mean(exp.(R2(state[1],state[2],state[3],epsilon[2,:])) .+ p.β*Emaxall[state+[0,0,1,-state[4]]])
        if d3[1] < 20
            v3 = R3(state[1],state[4],0.0) .+ p.β*Emaxall[d3]
        else
            v3 = R3(state[1],state[4],0.0)
        end
        v4 = R4(0.0) .+ p.β*Emaxall[state+[0,0,0,-state[4]]]
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
        fmaxE, Ev1, Ev2, Ev3, Ev4 = maxEt(Domain_set[t], Emaxall[t+1], MC_ϵ)
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

# auxiliary functions
function createX(maxE, Ev1val, Ev2val, Ev3val, Ev4val)
    x1 = maxE .- Ev1val
    x2 = maxE .- Ev2val
    #x3 = maxE .- Ev3val
    x4 = maxE .- Ev4val
    x5 = (maxE .- Ev1val).^0.5
    x6 = (maxE .- Ev2val).^0.5
    x7 = (maxE .- Ev3val).^0.5
    x8 = (maxE .- Ev4val).^0.5
    return x = [ones(size(maxE,1),1) x1 x2 x4 x5 x6 x7 x8]
end
function OLS(y,x)
    β = (x'*x)^(-1) * (x' * y)
end

function ApproximateTerminal(ApproxS::Int64)
    rngDomain = sample(Domain_set[T], ApproxS; replace=false)
    y, x, fEmax, maxE, Emax = genBothMax(rngDomain)
    βap = OLS(y, x)
    yf, xf, fEmaxTemp, maxEf, Emaxf, Ev1val, Ev2val, Ev3val, Ev4val = genBothMax(Domain_set[T])
    yhat = xf*βap
    for i = 1:size(yhat,1)
        if yhat[i] < 0
            yhat[i] = 0
        else
            yhat[i] = yhat[i]
        end
    end
    Emaxhat = yhat .+ maxEf
    df = DataFrame(Emaxap = Emaxf, maxEap = maxEf,
                Ev1val = Ev1val, Ev2val = Ev2val, Ev3val = Ev3val, Ev4val = Ev4val,
                y=yf, yhat = yhat, Emaxhat = Emaxhat, test = Domain_set[T])
    df |> save("df$(param)aprox.csv")
    fEmaxhat = Dict(zip(Domain_set[T], Emaxhat))
    return fEmaxhat, fEmax
    #return Emaxhat, fEmaxhat
end

function genBothMax(Domain::Array)
    fEmax, tEmax = @timed EmaxT(T, MC_ϵ, Domain)
    fmaxE, Ev1, Ev2, Ev3, Ev4 = maxET(MC_ϵ, Domain)
    maxE = collect(values(fmaxE))
    Emax = collect(values(fEmax))
    Ev1val = collect(values(Ev1))
    Ev2val = collect(values(Ev2))
    Ev3val = collect(values(Ev3))
    Ev4val = collect(values(Ev4))
    y = Emax .- maxE
    x = createX(maxE, Ev1val, Ev2val, Ev3val, Ev4val)
    return y, x, fEmax, maxE, Emax, Ev1val, Ev2val, Ev3val, Ev4val
end

function genBothMaxt(Domain::Array, Emaxallhat)
    fEmax, tEmax = @timed Emaxt(Domain, Emaxallhat, MC_ϵ)
    fmaxE, Ev1, Ev2, Ev3, Ev4 = maxEt(Domain, Emaxallhat, MC_ϵ)
    maxE = collect(values(fmaxE))
    Emax = collect(values(fEmax))
    Ev1val = collect(values(Ev1))
    Ev2val = collect(values(Ev2))
    Ev3val = collect(values(Ev3))
    Ev4val = collect(values(Ev4))
    y = Emax .- maxE
    x = createX(maxE, Ev1val, Ev2val, Ev3val, Ev4val)
    return y, x, fEmax, maxE, Emax
end

function ApproximateOnce(ApproxS::Int64, Domain_set::Array, Emaxallhat::Dict)
    rngDomain = sample(Domain_set, ApproxS; replace=false)
    y, x, fEmax = genBothMaxt(rngDomain, Emaxallhat)
    βap = OLS(y, x)
    yf, xf, fEmaxTemp, maxEf, Emaxf = genBothMaxt(Domain_set, Emaxallhat)
    yhat = xf*βap
    for i = 1:size(yhat,1)
        if yhat[i] < 0
            yhat[i] = 0
        else
            yhat[i] = yhat[i]
        end
    end
    Emaxhat = yhat .+ maxEf
    df = DataFrame(Emaxap = Emaxf, maxEap = maxEf,
                y=yf, yhat = yhat, Emaxhat = Emaxhat, test = Domain_set)
    df |> save("dfteste$(param)aprox.csv")
    fEmaxhat = Dict(zip(Domain_set, Emaxhat))
    return fEmaxhat, fEmax
    #return fEmaxhat = merge(Dict(zip(Domain_set, Emaxhat)), fEmax)
end


# since I am working with dictionaries, and in Julia dictionary keys
# are randomly ordered, if I want to compare the values from the OLS
# approximation and thef full solution, I have to create them both again
# so that the key orders are the same in both dictionaries

import StatsBase.sample
ApproxS = 2000
st = [10,0,0,1] # Initial state at t=1

fEmaxhat, fEmaxTemp = ApproximateTerminal(ApproxS)
#fEmaxhat = merge(fEmaxhat, fEmaxTemp)
#fEmaxhat = Dict(zip(Domain_set[T], Emaxhat))
Emaxallhat = Dict(T => fEmaxhat)

#t=39
#fEmaxhat1, fEmax1 = ApproximateOnce(ApproxS, Domain_set[t], Emaxallhat[t+1])


for t = reverse(2:T-1)
    println(t)
    if size(Domain_set[t],1) >= ApproxS
        fEmaxhat, fEmaxTemp = ApproximateOnce(ApproxS, Domain_set[t], Emaxallhat[t+1])
        #fEmaxhat = merge(fEmaxhat, fEmaxTemp)
    else
        fEmaxhat = Emaxt(Domain_set[t], Emaxallhat[t+1], MC_ϵ)
    end
    tempDict = Dict(t => fEmaxhat)
    global Emaxallhat = merge(Emaxallhat,tempDict)
end



df1 = SimulateAll(N, T, N_ϵ, Emaxallhat)


Ch1 = by(df1, :period) do x
    DataFrame(avgschool = mean(x.school_c), avgw1 = mean(x.work1), avgw2 = mean(x.work2), avghome = mean(x.home))
end


### EMAXALLHAT IS BUGGY, NOT STORING CORRECT INFORMATION


#=
#this is kind of awkward, maybe better to use DataFrames?
maxE_Aprox = Dict()
for i = keys(fEmax)
  tempVal = get(fmaxE, i, -10000)
  tempDict = Dict(zip([i], tempVal))
  global maxE_Aprox = merge(maxE_Aprox, tempDict)
end
# to check there are no -10000, minimum(collect(values(maxE_Aprox)))
=#
