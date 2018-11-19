# Calculates Emax at terminal period
function maxET(SST::Array, MC_ϵ::Array)
    maxE = zeros(size(SST,1),1)
    vbar = zeros(size(SST,1),4)
    for i = 1:size(SST,1)
        state = SST[i]
        r1 = mean(exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:])))
        r2 = mean(exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:])))
        r3 = R3(state[1],state[4],0.0)
        r4 = R4(0.0)
        maxE[i] = max(r1, r2, r3, r4)
        vbar[i,:] = [r1 r2 r3 r4]
    end
    fmaxE = OrderedDict(zip(SST,maxE))
    fEv1 = OrderedDict(zip(SST,vbar[:,1]))
    fEv2 = OrderedDict(zip(SST,vbar[:,2]))
    fEv3 = OrderedDict(zip(SST,vbar[:,3]))
    fEv4 = OrderedDict(zip(SST,vbar[:,4]))
    return fmaxE, fEv1, fEv2, fEv3, fEv4
end

# Calculates Emax at t=2,...,T-1
function maxEt(SSt::Array, fEmax::OrderedDict, MC_ϵ::Array)
    maxE = zeros(size(SSt,1),1)
    vbar = zeros(size(SSt,1),4)
    for i = 1:size(SSt,1)
        state = SSt[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = mean(exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:])) .+ p.β*fEmax[state+[0,1,0,-state[4]]])
        v2 = mean(exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:])) .+ p.β*fEmax[state+[0,0,1,-state[4]]])
        if d3[1] < 20
            v3 = R3(state[1],state[4],0.0) .+ p.β*fEmax[d3]
        else
            v3 = R3(state[1],state[4],0.0)
        end
        v4 = R4(0.0) .+ p.β*fEmax[state+[0,0,0,-state[4]]]
        maxE[i] = max(v1, v2, v3, v4)
        vbar[i,:] = [v1 v2 v3 v4]
    end
    fmaxEt = OrderedDict(zip(SSt,maxE))
    fEv1 = OrderedDict(zip(SSt,vbar[:,1]))
    fEv2 = OrderedDict(zip(SSt,vbar[:,2]))
    fEv3 = OrderedDict(zip(SSt,vbar[:,3]))
    fEv4 = OrderedDict(zip(SSt,vbar[:,4]))
    return fmaxEt, fEv1, fEv2, fEv3, fEv4
end

# Combines both together
function genmaxEAll(Emaxall::OrderedDict,Domain_set::OrderedDict,epsilon::Array)
    fmaxE, fEv1, fEv2, fEv3, fEv4 = maxET(T,MC_ϵ)
    # Store it in a dictionary with key = t, value = Emax
    maxEall = OrderedDict(T => fmaxE)
    Ev1all = OrderedDict(T => fEv1)
    Ev2all = OrderedDict(T => fEv2)
    Ev3all = OrderedDict(T => fEv3)
    Ev4all = OrderedDict(T => fEv4)
    for t = reverse(2:T-1)
        fmaxE, fEv1, fEv2, fEv3, fEv4 = maxEt(Domain_set[t], Emaxall[t+1], MC_ϵ)
        tempDict = OrderedDict(t => fmaxE)
        tempDict1 = OrderedDict(t => fEv1)
        tempDict2 = OrderedDict(t => fEv2)
        tempDict3 = OrderedDict(t => fEv3)
        tempDict4 = OrderedDict(t => fEv4)
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
function createX(fmaxE::OrderedDict, fEv1::OrderedDict, fEv2::OrderedDict, fEv3::OrderedDict, fEv4::OrderedDict)
    n = size(collect(values(fmaxE)),1)
    x1 = collect(values(fmaxE)) .- collect(values(fEv1))
    x2 = collect(values(fmaxE)) .- collect(values(fEv2))
    #x3 = fmaxE .- Ev3
    x4 = collect(values(fmaxE)) .- collect(values(fEv4))
    x5 = (collect(values(fmaxE)) .- collect(values(fEv1))).^0.5
    x6 = (collect(values(fmaxE)) .- collect(values(fEv2))).^0.5
    x7 = (collect(values(fmaxE)) .- collect(values(fEv3))).^0.5
    x8 = (collect(values(fmaxE)) .- collect(values(fEv4))).^0.5
    x = [ones(n,1) x1 x2 x4 x5 x6 x7 x8]
    return x
end

function OLS(y,x)
    β = (x'*x)^(-1) * (x' * y)
end

function genApproxDataT(Domain::Array, MC_ϵ::Array)
    fmaxE, fEv1, fEv2, fEv3, fEv4 = maxET(Domain, MC_ϵ)
    fEmax = EmaxT(Domain, MC_ϵ)
    y = collect(values(fEmax)) .- collect(values(fmaxE))
    x = createX(fmaxE, fEv1, fEv2, fEv3, fEv4)
    βap = OLS(y,x)
    return βap, fEmax
end

function ApproximateTerminal(ApproxS::Int64)
    rngDomain = sample(Domain_set[T], ApproxS; replace=false)
    βap, fEmaxS = genApproxDataT(rngDomain, MC_ϵ)
    fmaxE, fEv1, fEv2, fEv3, fEv4 = maxET(Domain_set[T], MC_ϵ)
    fEmax = EmaxT(Domain_set[T], MC_ϵ)
    yfull = collect(values(fEmax)) .- collect(values(fmaxE))
    xfull = createX(fmaxE, fEv1, fEv2, fEv3, fEv4)
    yhat = xfull*βap
    for i = 1:size(yhat,1)
        if yhat[i] < 0
            yhat[i] = 0
        else
            yhat[i] = yhat[i]
        end
    end
    Emaxhat = yhat .+ collect(values(fmaxE))
    fEmaxhat = OrderedDict(zip(Domain_set[T], Emaxhat))
    fEmaxhat = merge(fEmaxhat, fEmaxS)
    Domain = collect(keys(fEmaxhat))
    df = DataFrame(Emaxap = collect(values(fEmax)), maxEap = collect(values(fmaxE)), yf = yfull,
                Ev1val = collect(values(fEv1)), Ev2val = collect(values(fEv2)), Ev3val = collect(values(fEv3)),
                Ev4val = collect(values(fEv4)), yhat = yhat, Emaxhat = Emaxhat, test = Domain)
    df |> save("output/df$(param)aprox.csv")
    return fEmaxhat
    #return Emaxhat, fEmaxhat
end

function genApproxData(Domain::Array, MC_ϵ::Array, fEmaxhat::OrderedDict)
    fmaxE, fEv1, fEv2, fEv3, fEv4 = maxEt(Domain, fEmaxhat, MC_ϵ)
    fEmax = Emaxt(Domain, fEmaxhat, MC_ϵ)
    y = collect(values(fEmax)) .- collect(values(fmaxE))
    x = createX(fmaxE, fEv1, fEv2, fEv3, fEv4)
    βap = OLS(y,x)
    return βap, fEmax
end


function ApproximateOnce(ApproxS::Int64, Domain::Array, fEmaxhat::OrderedDict)
    rngDomain = sample(Domain, ApproxS; replace=false)
    βap, fEmaxS = genApproxData(rngDomain, MC_ϵ, fEmaxhat)
    fmaxE, fEv1, fEv2, fEv3, fEv4 = maxEt(Domain, fEmaxhat, MC_ϵ)
    fEmax = EmaxT(Domain, MC_ϵ)
    yfull = collect(values(fEmax)) .- collect(values(fmaxE))
    xfull = createX(fmaxE, fEv1, fEv2, fEv3, fEv4)
    yhat = xfull*βap
    #for i = 1:size(yhat,1)
    #    if yhat[i] < 0
    #        yhat[i] = 0
    #    else
    #        yhat[i] = yhat[i]
    #    end
    #end
    Emaxhat = yhat .+ collect(values(fmaxE))
    for i = 1:size(Emaxhat,1)
        if Emaxhat[i] < collect(values(fmaxE))[i]
            Emaxhat[i] = collect(values(fmaxE))[i]
        end
    end
    fEmaxhat = OrderedDict(zip(Domain, Emaxhat))
    fEmaxhat = merge(fEmaxhat, fEmaxS)
    Domain = collect(keys(fEmaxhat))
    df = DataFrame(Emaxap = collect(values(fEmax)), maxEap = collect(values(fmaxE)), yf = yfull,
                Ev1val = collect(values(fEv1)), Ev2val = collect(values(fEv2)), Ev3val = collect(values(fEv3)),
                Ev4val = collect(values(fEv4)), yhat = yhat, Emaxhat = Emaxhat, test = Domain)
    df |> save("output/df39$(param)aprox.csv")
    return fEmaxhat
end


import StatsBase.sample
ApproxS = 2000
st = [10,0,0,1] # Initial state at t=1

fEmaxhat = ApproximateTerminal(ApproxS)
Emaxallhat = OrderedDict(T => fEmaxhat)
#ApproximateOnce(ApproxS, Domain_set[t], Emaxallhat[t+1])
for t = reverse(2:T-1)
    println(t)
    if size(Domain_set[t],1) >= ApproxS
        fEmaxhat = ApproximateOnce(ApproxS, Domain_set[t], Emaxallhat[t+1])
    else
        fEmaxhat = Emaxt(Domain_set[t], Emaxallhat[t+1], MC_ϵ)
    end
    tempDict = OrderedDict(t => fEmaxhat)
    global Emaxallhat = merge(Emaxallhat,tempDict)
end



df1 = SimulateAll(N, T, N_ϵ, Emaxallhat)

Ch1 = by(df1, :period) do x
    DataFrame(avgschool = mean(x.school_c), avgw1 = mean(x.work1), avgw2 = mean(x.work2), avghome = mean(x.home))
end
