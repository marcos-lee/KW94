function unpack(θ::Array)
    α1 = θ[1:6]
    α2 = θ[7:12]
    β = θ[13:15]
    γ0 = θ[16]
    ch = exp.(θ[17:20])
    #println("\n ln guess \n")
    #show(θ[17:26])
    L = [ch[1] 0 0 0;
        θ[21] ch[2] 0 0;
        θ[22] θ[24] ch[3] 0;
        θ[23] θ[25] θ[26] ch[4]]
    return α1, α2, β, γ0, L
end

function R1(s::Int64, x1::Int64, x2::Int64, ϵ1, α1::Array)
    r1 = α1[1] + α1[2] * s + α1[3] * x1 - α1[4] * (x1^2) + α1[5] *x2 - α1[6] *(x2^2) .+ ϵ1
end

function R2(s::Int64, x1::Int64, x2::Int64, ϵ2, α2::Array)
    r2 = α2[1] + α2[2] * s + α2[3] * x2 - α2[4] * (x2^2) + α2[5] *x1 - α2[6] *(x1^2) .+ ϵ2
end

function R3(s::Int64, slag::Int64, ϵ3, β::Array)
        r3 = β[1] - β[2] * (s >= 12) - β[3] * (1 - slag) .+ ϵ3
end

function R4(ϵ4, γ0)
    r4 = γ0 .+ ϵ4
end


# Calculates Emax at terminal period
# SST = State Space at T
function EmaxT(SST::Array, MC_ϵ::Array, α1::Array, α2::Array, β::Array, γ0)
    Emax = Array{eltype(α1)}(undef, size(SST,1),1)
    for i = 1:size(SST,1)
        state = SST[i]
        r1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:],α1))
        r2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:],α2))
        if state[1] < 20
            r3 = R3(state[1],state[4],MC_ϵ[3,:],β)
        else
            r3 = R3(state[1],state[4],MC_ϵ[3,:],β) .- 50000.0
        end
        r4 = R4(MC_ϵ[4,:],γ0)
        Emax[i] = mean(max.(r1, r2, r3, r4))
    end
    return fEmax = OrderedDict{Array{Int64,1},eltype(α1)}(zip(SST,Emax))
end


# Calculates Emax at t=2,...,T-1
# SSt = State Space at t
function Emaxt(SSt::Array, fEmax::OrderedDict, MC_ϵ::Array, α1::Array, α2::Array, β::Array, γ0)
    Emax = Array{eltype(α1)}(undef, size(SSt,1),1)
    for i = 1:size(SSt,1)
        state = SSt[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:],α1)) .+ 0.95*fEmax[state+[0,1,0,-state[4]]]
        v2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:],α2)) .+ 0.95*fEmax[state+[0,0,1,-state[4]]]
        if state[1] < 20
            v3 = R3(state[1],state[4],MC_ϵ[3,:],β) .+ 0.95*fEmax[d3]
        else
            v3 = R3(state[1],state[4],MC_ϵ[3,:],β) .- 50000.0
        end
        v4 = R4(MC_ϵ[4,:],γ0) .+ 0.95*fEmax[state+[0,0,0,-state[4]]]
        Emax[i] = mean(max.(v1, v2, v3, v4))
    end
    return fEmaxt = OrderedDict{Array{Int64,1},eltype(α1)}(zip(SSt,Emax))
end
#println("\n Backward induction \n")
#println("\n Solving Exact Model \n")
#println("== Iteration t=$T ==\n")
# Combines both together
function genEmaxAll(Domain_set::OrderedDict, MC_ϵ::Array, T::Int64, α1::Array, α2::Array, β::Array, γ0)
    fEmax = EmaxT(Domain_set[T], MC_ϵ, α1, α2, β, γ0)
    Emaxall = OrderedDict{Int64, OrderedDict{Array{Int64,1},eltype(α1)}}(T => fEmax)
    for t = reverse(2:T-1)
        fEmax = Emaxt(Domain_set[t], fEmax, MC_ϵ, α1, α2, β, γ0)
        tempDict = OrderedDict{Int64, OrderedDict{Array{Int64,1},eltype(α1)}}(t => fEmax)
        Emaxall = merge(Emaxall,tempDict)
    end
    return Emaxall
end


# Calculates Emax at terminal period
function maxET(SST::Array, MC_ϵ::Array, α1, α2, β, γ0)
    maxE = Array{eltype(α1)}(undef, size(SST,1), 1)
    vbar = Array{eltype(α1)}(undef, size(SST,1),4)
    for i = 1:size(SST,1)
        state = SST[i]
        r1 = mean(exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:],α1)))
        r2 = mean(exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:],α2)))
        if state[1] < 20
            r3 = R3(state[1],state[4],0.0,β)
        else
            r3 = R3(state[1],state[4],0.0,β) .- 50000.0
        end
        r4 = R4(0.0,γ0)
        maxE[i] = max(r1, r2, r3, r4)
        vbar[i,:] = [r1 r2 r3 r4]
    end
    fmaxE = OrderedDict{Array{Int64,1},eltype(α1)}(zip(SST,maxE))
    fEv1 = OrderedDict{Array{Int64,1},eltype(α1)}(zip(SST,vbar[:,1]))
    fEv2 = OrderedDict{Array{Int64,1},eltype(α1)}(zip(SST,vbar[:,2]))
    fEv3 = OrderedDict{Array{Int64,1},eltype(α1)}(zip(SST,vbar[:,3]))
    fEv4 = OrderedDict{Array{Int64,1},eltype(α1)}(zip(SST,vbar[:,4]))
    return fmaxE, fEv1, fEv2, fEv3, fEv4
end

# Calculates Emax at t=2,...,T-1
function maxEt(SSt::Array, fEmax::OrderedDict, MC_ϵ::Array, α1, α2, β, γ0)
    maxE = Array{eltype(α1)}(undef, size(SSt,1),1)
    vbar = Array{eltype(α1)}(undef, size(SSt,1),4)
    for i = 1:size(SSt,1)
        state = SSt[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = mean(exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:],α1)) .+ 0.95*fEmax[state+[0,1,0,-state[4]]])
        v2 = mean(exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:],α2)) .+ 0.95*fEmax[state+[0,0,1,-state[4]]])
        if state[1] < 20
            v3 = R3(state[1],state[4],0.0,β) .+ 0.95*fEmax[d3]
        else
            v3 = R3(state[1],state[4],0.0,β) .- 50000.0
        end
        v4 = R4(0.0,γ0) .+ 0.95*fEmax[state+[0,0,0,-state[4]]]
        maxE[i] = max(v1, v2, v3, v4)
        vbar[i,:] = [v1 v2 v3 v4]
    end
    fmaxEt = OrderedDict{Array{Int64,1},eltype(α1)}(zip(SSt,maxE))
    fEv1 = OrderedDict{Array{Int64,1},eltype(α1)}(zip(SSt,vbar[:,1]))
    fEv2 = OrderedDict{Array{Int64,1},eltype(α1)}(zip(SSt,vbar[:,2]))
    fEv3 = OrderedDict{Array{Int64,1},eltype(α1)}(zip(SSt,vbar[:,3]))
    fEv4 = OrderedDict{Array{Int64,1},eltype(α1)}(zip(SSt,vbar[:,4]))
    return fmaxEt, fEv1, fEv2, fEv3, fEv4
end


# auxiliary functions
function createX(fmaxE::OrderedDict, fEv1::OrderedDict, fEv2::OrderedDict, fEv3::OrderedDict, fEv4::OrderedDict)
    n = size(collect(values(fmaxE)),1)
    x1 = collect(values(fmaxE)) .- collect(values(fEv1))
    x2 = collect(values(fmaxE)) .- collect(values(fEv2))
    x4 = collect(values(fmaxE)) .- collect(values(fEv4))
    x5 = (collect(values(fmaxE)) .- collect(values(fEv1))).^0.5
    x6 = (collect(values(fmaxE)) .- collect(values(fEv2))).^0.5
    x7 = (collect(values(fmaxE)) .- collect(values(fEv3))).^0.5
    x8 = (collect(values(fmaxE)) .- collect(values(fEv4))).^0.5
    if param > 1
        x3 = collect(values(fmaxE)) .- collect(values(fEv3))
        x = [ones(n,1) x1 x2 x3 x4 x5 x6 x7 x8]
    else
        x = [ones(n,1) x1 x2 x4 x5 x6 x7 x8]
    end
    return x
end

function OLS(y::Array, x::Array)
    β = (x'*x)^(-1) * (x' * y)
end

function genApproxDataT(Domain::Array, MC_ϵ::Array, α1, α2, β, γ0)
    fmaxE, fEv1, fEv2, fEv3, fEv4 = maxET(Domain, MC_ϵ, α1, α2, β, γ0)
    fEmax = EmaxT(Domain, MC_ϵ, α1, α2, β, γ0)
    y = collect(values(fEmax)) .- collect(values(fmaxE))
    x = createX(fmaxE, fEv1, fEv2, fEv3, fEv4)
    βap = OLS(y,x)
    return βap, fEmax
end

function ApproximateTerminal(ApproxS::Int64, MC_ϵ::Array, α1, α2, β, γ0)
    rngDomain = sample(Domain_set[T], ApproxS; replace=false)
    βap, fEmaxS = genApproxDataT(rngDomain, MC_ϵ, α1, α2, β, γ0)
    fmaxE, fEv1, fEv2, fEv3, fEv4 = maxET(Domain_set[T], MC_ϵ, α1, α2, β, γ0)
    xfull = createX(fmaxE, fEv1, fEv2, fEv3, fEv4)
    yhat = xfull*βap
    yhat[yhat .< 0] .= 0.0
    Emaxhat = yhat .+ collect(values(fmaxE))
    fEmaxhat = OrderedDict{Array{Int64,1},eltype(α1)}(zip(Domain_set[T], Emaxhat))
    fEmaxhat = merge(fEmaxhat, fEmaxS)
    return fEmaxhat
    #return Emaxhat, fEmaxhat
end

function genApproxData(Domain::Array, MC_ϵ::Array, fEmaxhat::OrderedDict, α1, α2, β, γ0)
    fmaxE, fEv1, fEv2, fEv3, fEv4 = maxEt(Domain, fEmaxhat, MC_ϵ, α1, α2, β, γ0)
    fEmax = Emaxt(Domain, fEmaxhat, MC_ϵ, α1, α2, β, γ0)
    y = collect(values(fEmax)) .- collect(values(fmaxE))
    x = createX(fmaxE, fEv1, fEv2, fEv3, fEv4)
    βap = OLS(y,x)
    return βap, fEmax
end


function ApproximateOnce(ApproxS::Int64, Domain::Array, fEmaxhat::OrderedDict, MC_ϵ,  α1, α2, β, γ0)
    rngDomain = sample(Domain, ApproxS; replace=false)
    βap, fEmaxS = genApproxData(rngDomain, MC_ϵ, fEmaxhat, α1, α2, β, γ0)
    fmaxE, fEv1, fEv2, fEv3, fEv4 = maxEt(Domain, fEmaxhat, MC_ϵ, α1, α2, β, γ0)
    xfull = createX(fmaxE, fEv1, fEv2, fEv3, fEv4)
    yhat = xfull*βap
    yhat[yhat .< 0] .= 0.0
    Emaxhat = yhat .+ collect(values(fmaxE))
    fEmaxhat = OrderedDict{Array{Int64,1},eltype(α1)}(zip(Domain, Emaxhat))
    fEmaxhat = merge(fEmaxhat, fEmaxS)
    return fEmaxhat
end

function genEmaxAllHat(Domain_set::OrderedDict, ApproxS::Int64, MC_ϵ, α1, α2, β, γ0)
    fEmaxhat = ApproximateTerminal(ApproxS, MC_ϵ, α1, α2, β, γ0)
    Emaxallhat = OrderedDict{Int64, OrderedDict{Array{Int64,1},eltype(α1)}}(T => fEmaxhat)
    for t = reverse(2:T-1)
        if size(Domain_set[t],1) >= ApproxS
            fEmaxhat = ApproximateOnce(ApproxS, Domain_set[t], Emaxallhat[t+1], MC_ϵ, α1, α2, β, γ0)
        else
            fEmaxhat = Emaxt(Domain_set[t], Emaxallhat[t+1], MC_ϵ, α1, α2, β, γ0)
        end
        tempDict = OrderedDict{Int64, OrderedDict{Array{Int64,1},eltype(α1)}}(t => fEmaxhat)
        Emaxallhat = merge(Emaxallhat,tempDict)
    end
    return Emaxallhat
end




function likelihood(df::DataFrame, Draws::Array, Domain_set::OrderedDict, ApproxS::Int64, θ::Array)
    α1, α2, β, γ0, L = unpack(θ)
    MC_ϵ = Array{eltype(α1)}(undef, 4, size(Draws)[2])
    MC_ϵ[1,:] = Draws[1,:] .* L[1,1]
    MC_ϵ[2,:] = Draws[1,:] .* L[2,1] .+ Draws[2,:] .* L[2,2]
    MC_ϵ[3,:] = Draws[1,:] .* L[3,1] .+ Draws[2,:] .* L[3,2] .+ Draws[3,:] .* L[3,3]
    MC_ϵ[4,:] = Draws[1,:] .* L[4,1] .+ Draws[2,:] .* L[4,2] .+ Draws[3,:] .* L[4,3] .+ Draws[4,:] .* L[4,4]
    Emaxall = genEmaxAllHat(Domain_set, ApproxS, MC_ϵ, α1, α2, β, γ0)
    tempdf = df[df.period .== 40,:]
    fstate = [tempdf.educ tempdf.work1 tempdf.work2 tempdf.lag]
    probT = llcontribT(fstate, tempdf.wage, tempdf.choice, MC_ϵ[:,1:200], α1, α2, β, γ0, L)
    probt = Array{eltype(α1)}(undef, maximum(df.id), T-1)
    for t = reverse(1:T-1)
        tempdf = df[df.period .== t,:]
        fstate = [tempdf.educ tempdf.work1 tempdf.work2 tempdf.lag]
        probt[:,t] = llcontrib(fstate, tempdf.wage, tempdf.choice, MC_ϵ[:,1:200], Emaxall[t+1], α1, α2, β, γ0, L)
    end
    prob = hcat(probt, probT)
    prob[prob.==0] .= 1e-100
    #println("\n exp guess \n")
    #show(L)
    ll = -sum(log.(prob))
end

function llcontrib(fstate::Array{Int64,2}, wage::Array{Float64,1}, fchoice::Array{Int64,1}, MC_ϵ::Array, fEmax::OrderedDict, α1::Array, α2::Array, β::Array, γ0, L::Array)
    prob = Array{eltype(α1)}(undef,size(wage)[1])
    for i = 1:size(wage)[1]
        state = fstate[i,:]
        choice = fchoice[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:],α1)) .+ 0.95*fEmax[state+[0,1,0,-state[4]]]
        v2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:],α2)) .+ 0.95*fEmax[state+[0,0,1,-state[4]]]
        if state[1] < 20
            v3 = R3(state[1],state[4],MC_ϵ[3,:],β) .+ 0.95*fEmax[d3]
        else
            v3 = R3(state[1],state[4],MC_ϵ[3,:],β) .- 50000.0
        end
        v4 = R4(MC_ϵ[4,:],γ0) .+ 0.95*fEmax[state+[0,0,0,-state[4]]]
        if choice == 1
            v1_obs = wage[i] .+ 0.95*fEmax[state+[0,1,0,-state[4]]]
            vmax = max.(v1_obs, v2, v3, v4)
            logit = exp.(v1_obs .- vmax) .+ exp.(v2 .- vmax) .+ exp.(v3 .- vmax) .+ exp.(v4 .- vmax)
            probt = exp.(v1_obs .- vmax)./logit
            denw = pdf.(Normal(0,L[1,1]), log(wage[i]) .- R1(state[1],state[2],state[3],0,α1))
        elseif choice == 2
            v2_obs = wage[i] .+ 0.95*fEmax[state+[0,0,1,-state[4]]]
            vmax = max.(v1, v2_obs, v3, v4)
            logit = exp.(v1 .- vmax) .+ exp.(v2_obs .- vmax) .+ exp.(v3 .- vmax) .+ exp.(v4 .- vmax)
            probt = exp.(v2_obs .- vmax)./logit
            denw = pdf.(Normal(0,L[2,2]), log(wage[i]) .- R2(state[1],state[2],state[3],0,α2))
        elseif choice == 3
            vmax = max.(v1, v2, v3, v4)
            logit = exp.(v1 .- vmax) .+ exp.(v2 .- vmax) .+ exp.(v3 .- vmax) .+ exp.(v4 .- vmax)
            probt = exp.(v3 .- vmax)./logit
            denw = 1.0
        elseif choice == 4
            vmax = max.(v1, v2, v3, v4)
            logit = exp.(v1 .- vmax) .+ exp.(v2 .- vmax) .+ exp.(v3 .- vmax) .+ exp.(v4 .- vmax)
            probt = exp.(v4 .- vmax)./logit
            denw = 1.0
        end
        #probt[isnan.(probt)] .= 1.0
        prob[i] = mean(probt) .* denw
    end
    return prob
end

function llcontribT(fstate::Array{Int64,2}, wage::Array{Float64,1}, fchoice::Array{Int64,1}, MC_ϵ::Array, α1::Array, α2::Array, β::Array, γ0, L::Array)
    prob = Array{eltype(α1)}(undef,size(wage)[1])
    for i = 1:size(wage)[1]
        state = fstate[i,:]
        choice = fchoice[i]
        r1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:],α1))
        r2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:],α2))
        if state[20] < 20
            r3 = R3(state[1],state[4],MC_ϵ[3,:],β)
        else
            r3 = R3(state[1],state[4],MC_ϵ[3,:],β) .- 50000.0
        end
        r4 = R4(MC_ϵ[4,:],γ0)
        if choice == 1
            rmax = max.(wage[i], r2, r3, r4)
            logit = exp.(wage[i] .- rmax) .+ exp.(r2 .- rmax) .+ exp.(r3 .- rmax) .+ exp.(r4 .- rmax)
            probt = exp.(wage[i] .- rmax)./logit
            denw = pdf.(Normal(0,L[1,1]), log(wage[i]) .- R1(state[1],state[2],state[3],0,α1))
        elseif choice == 2
            rmax = max.(r1, wage[i], r3, r4)
            logit = exp.(r1 .- rmax) .+ exp.(wage[i] .- rmax) .+ exp.(r3 .- rmax) .+ exp.(r4 .- rmax)
            probt = exp.(wage[i] .- rmax)./logit
            denw = pdf.(Normal(0,L[2,2]), log(wage[i]) .- R2(state[1],state[2],state[3],0,α2))
        elseif choice == 3
            rmax = max.(r1, r2, r3, r4)
            logit = exp.(r1 .- rmax) .+ exp.(r2 .- rmax) .+ exp.(r3 .- rmax) .+ exp.(r4 .- rmax)
            probt = exp.(r3 .- rmax)./logit
            denw = 1.0
        elseif choice == 4
            rmax = max.(r1, r2, r3, r4)
            logit = exp.(r1 .- rmax) .+ exp.(r2 .- rmax) .+ exp.(r3 .- rmax) .+ exp.(r4 .- rmax)
            probt = exp.(r4 .- rmax)./logit
            denw = 1.0
        end
        #probt[isnan.(probt)] .= 1.0
        prob[i] = mean(probt) .* denw
    end
    return prob
end
