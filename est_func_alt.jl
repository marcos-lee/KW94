function unpack(θ::Array)
    α1 = θ[1:6]
    α2 = θ[7:12]
    β = θ[13:15]
    γ0 = θ[16]
    ch = exp.(θ[17:26])
    L = [ch[1] 0 0 0;
        ch[2] ch[5] 0 0;
        ch[3] ch[6] ch[8] 0;
        ch[4] ch[7] ch[9] ch[10]]
    return α1, α2, β, γ0, L
end

function R1(s::Int64, x1::Int64, x2::Int64, ϵ1, α1::Array)
    r1 = α1[1] + α1[2] * s + α1[3] * x1 - α1[4] * (x1^2) + α1[5] *x2 - α1[6] *(x2^2) .+ ϵ1
end

function R2(s::Int64, x1::Int64, x2::Int64, ϵ2, α2::Array)
    r2 = α2[1] + α2[2] * s + α2[3] * x2 - α2[4] * (x2^2) + α2[5] *x1 - α2[6] *(x1^2) .+ ϵ2
end

function R3(s::Int64, slag::Int64, ϵ3, β::Array)
    if s <= 19
        r3 = β[1] - β[2] * (s >= 12) - β[3] * (1 - slag) .+ ϵ3
    else
        r3 = - β[3] .+ ϵ3 .- 40000
    end
end

function R4(ϵ4, γ0::Float64)
    r4 = γ0 .+ ϵ4
end

# Calculates Emax at terminal period
# SST = State Space at T
function EmaxT(SST::Array, MC_ϵ::Array, θ::Array)
    α1, α2, β, γ0, L = unpack(θ)
    Emax = zeros(size(SST,1),1)
    for i = 1:size(SST,1)
        state = SST[i]
        r1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:],α1))
        r2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:],α2))
        r3 = R3(state[1],state[4],MC_ϵ[3,:],β)
        r4 = R4(MC_ϵ[4,:],γ0)
        Emax[i] = mean(max.(r1, r2, r3, r4))
    end
    return fEmax = OrderedDict(zip(SST,Emax))
end


# Calculates Emax at t=2,...,T-1
# SSt = State Space at t
function Emaxt(SSt::Array, fEmax::OrderedDict, MC_ϵ::Array, θ::Array)
    α1, α2, β, γ0, L = unpack(θ)
    Emax = zeros(size(SSt,1),1)
    for i = 1:size(SSt,1)
        state = SSt[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:],α1)) .+ 0.95*fEmax[state+[0,1,0,-state[4]]]
        v2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:],α2)) .+ 0.95*fEmax[state+[0,0,1,-state[4]]]
        if d3[1] < 20
            v3 = R3(state[1],state[4],MC_ϵ[3,:],β) .+ 0.95*fEmax[d3]
        else
            v3 = R3(state[1],state[4],MC_ϵ[3,:],β)
        end
        v4 = R4(MC_ϵ[4,:],γ0) .+ 0.95*fEmax[state+[0,0,0,-state[4]]]
        Emax[i] = mean(max.(v1, v2, v3, v4))
    end
    return fEmaxt = OrderedDict(zip(SSt,Emax))
end
#println("\n Backward induction \n")
#println("\n Solving Exact Model \n")
#println("== Iteration t=$T ==\n")
# Combines both together
function genEmaxAll(Domain_set::OrderedDict, MC_ϵ::Array, T::Int64, θ::Array)
    fEmax, tEmax = @timed EmaxT(Domain_set[T], MC_ϵ, θ)
    Emaxall = OrderedDict(T => fEmax)
    timeEmax = Array{Float64}(undef, 39, 2)
    timeEmax[39,:] = [40 tEmax]
    for t = reverse(2:T-1)
        fEmax, tEmax = @timed Emaxt(Domain_set[t], fEmax, MC_ϵ, θ)
        timeEmax[t,:] = [t tEmax]
        tempDict = OrderedDict(t => fEmax)
        Emaxall = merge(Emaxall,tempDict)
    end
    return Emaxall, timeEmax
end
function likelihood(df::DataFrame, Draws::Array, Domain_set::OrderedDict, θ::Array)
    α1, α2, β, γ0, L = unpack(θ)
    MC_ϵ = Array{Any}(undef, 4, size(Draws)[2])
    MC_ϵ[1,:] = Draws[1,:] .* L[1,1]
    MC_ϵ[2,:] = Draws[1,:] .* L[2,1] .+ Draws[2,:] .* L[2,2]
    MC_ϵ[3,:] = Draws[1,:] .* L[3,1] .+ Draws[2,:] .* L[3,2] .+ Draws[3,:] .* L[3,3]
    MC_ϵ[4,:] = Draws[1,:] .* L[4,1] .+ Draws[2,:] .* L[4,2] .+ Draws[3,:] .* L[4,3] .+ Draws[4,:] .* L[4,4]
    Emaxall, timeEmax = genEmaxAll(Domain_set,MC_ϵ, T, θ)
    probT = llcontribT(df[df.period .== 40,:], MC_ϵ, θ)
    probt = Array{Float64}(undef, maximum(df.id), T-1)
    for t = reverse(1:T-1)
        global probt[:,t] = llcontrib(df[df.period .== t,:], MC_ϵ, Emaxall[t+1], θ)
    end
    prob = hcat(probt, probT)
    ll = -sum(log.(prob))
end

function llcontrib(tempdf::DataFrame, MC_ϵ::Array, fEmax::OrderedDict, θ::Array)
    α1, α2, β, γ0, L = unpack(θ)
    fstate = [tempdf.educ tempdf.work1 tempdf.work2 tempdf.lag]
    prob = Array{Float64}(undef,size(tempdf)[1])
    for i = 1:size(tempdf)[1]
        state = fstate[i,:]
        choice = tempdf.choice[i]
        d3 = state + [1,0,0,1-state[4]]
        d3[1] = min(20,d3[1])
        v1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:],α1)) .+ 0.95*fEmax[state+[0,1,0,-state[4]]]
        v2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:],α2)) .+ 0.95*fEmax[state+[0,0,1,-state[4]]]
        if d3[1] < 20
            v3 = R3(state[1],state[4],MC_ϵ[3,:],β) .+ 0.95*fEmax[d3]
        else
            v3 = R3(state[1],state[4],MC_ϵ[3,:],β)
        end
        v4 = R4(MC_ϵ[4,:],γ0) .+ 0.95*fEmax[state+[0,0,0,-state[4]]]
        vmax = max.(v1,v2,v3,v4)
        if choice == 1
            probt =  (v1 .== vmax)
            denw = pdf.(Normal(0,L[1,1]), log(tempdf.wage[i]) .- R1(state[1],state[2],state[3],0,α1))
        elseif choice == 2
            probt = (v2 .== vmax)
            denw = pdf.(Normal(0,sqrt(L[2,1]^2 + L[2,2]^2)), log(tempdf.wage[i]) .- R2(state[1],state[2],state[3],0,α2))
        elseif choice == 3
            probt = (v3 .== vmax)
            denw = 1
        elseif choice == 4
            probt = (v4 .== vmax)
            denw = 1
        end
        prob[i] = mean(probt) .* denw
    end
    return prob
end

function llcontribT(tempdf::DataFrame, MC_ϵ::Array, θ::Array)
    α1, α2, β, γ0, L = unpack(θ)
    fstate = [tempdf.educ tempdf.work1 tempdf.work2 tempdf.lag]
    prob = Array{Float64}(undef,size(tempdf)[1])
    for i = 1:size(tempdf)[1]
        state = fstate[i,:]
        choice = tempdf.choice[i]
        r1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:],α1))
        r2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:],α2))
        r3 = R3(state[1],state[4],MC_ϵ[3,:],β)
        r4 = R4(MC_ϵ[4,:],γ0)
        rmax = max.(r1,r2,r3,r4)
        if choice == 1
            probt = (r1 .== rmax)
            denw = pdf.(Normal(0,L[1,1]), log(tempdf.wage[i]) .- R1(state[1],state[2],state[3],0,α1))
        elseif choice == 2
            probt = (r2 .== rmax)
            denw = pdf.(Normal(0,sqrt(L[2,1]^2 + L[2,2]^2)), log(tempdf.wage[i]) .- R2(state[1],state[2],state[3],0,α2))
        elseif choice == 3
            probt = (r3 .== rmax)
            denw = 1
        elseif choice == 4
            probt = (r4 .== rmax)
            denw = 1
        end
        prob[i] = mean(probt) .* denw
    end
    return prob
end
