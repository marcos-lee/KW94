function unpack(θ::Array)
    α1 = θ[1:6]
    α2 = θ[7:12]
    β = θ[13:15]
    γ0 = θ[16]
    ch = exp.(θ[17:20])
    for i = 1:4
        if ch[i] < 1.0e-8
            ch[i] =1.0e-8
        end
    end
    #println("\n ln guess \n")
    #show(θ[17:26])
    L = [ch[1] 0 0 0;
        θ[21] ch[2] 0 0;
        θ[22] θ[24] ch[3] 0;
        θ[23] θ[25] θ[26] ch[4]]
    #L = [ch[1] 0 0 0;
    #    0 ch[2] 0 0;
    #    0 0 ch[3] 0;
    #    0 0 0 ch[4]]
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
        if state[1]  < 20
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
    fEmax = EmaxT(Domain_set[T], MC_ϵ[:,:,T], α1, α2, β, γ0)
    Emaxall = OrderedDict{Int64, OrderedDict{Array{Int64,1},eltype(α1)}}(T => fEmax)
    for t = reverse(2:T-1)
        fEmax = Emaxt(Domain_set[t], fEmax, MC_ϵ[:,:,t], α1, α2, β, γ0)
        tempDict = OrderedDict{Int64, OrderedDict{Array{Int64,1},eltype(α1)}}(t => fEmax)
        Emaxall = merge(Emaxall,tempDict)
    end
    return Emaxall
end


function convertDF(df::DataFrame)
    df.choice = Array{Int64}(undef, size(df,1))
    for i = 1:size(df,1)
            if df.school_c[i] == 1
                    df.choice[i] = 3
            elseif df.work1c[i] == 1
                    df.choice[i] = 1
            elseif df.work2c[i] == 1
                    df.choice[i] = 2
            else
                    df.choice[i] = 4
            end
    end
    df.id = df.idx
    fstate = Array{Int64}(undef, maximum(df.id), maximum(df.period), 4)
    wage = Array{Float64}(undef, maximum(df.id), maximum(df.period))
    choice = Array{Int64}(undef, maximum(df.id), maximum(df.period))
    for t = reverse(1:maximum(df.period))
        tempdf = df[df.period .== t,:]
        fstate[:,t,:] = [tempdf.educ tempdf.work1 tempdf.work2 tempdf.lag]
        wage[:,t] = tempdf.wage
        choice[:,t] = tempdf.choice
    end
    return fstate, wage, choice
end



function likelihood(state, wage, choice, Draws::Array, Domain_set::OrderedDict, lambda::Float64, θ::Array)
    α1, α2, β, γ0, L = unpack(θ)
    MC_ϵ = Array{eltype(α1)}(undef, 4, size(Draws)[2], size(Draws)[3])
    N = size(wage, 1)
    T = size(wage, 2)
    for t = 1:T
        MC_ϵ[:,:,t] = L * Draws[:,:,t]
    end
    Emaxall = genEmaxAll(Domain_set, MC_ϵ, T, α1, α2, β, γ0)
    probT = llcontribT(state[:,T,:], wage[:,T], choice[:,T], MC_ϵ[:,:,T], lambda, α1, α2, β, γ0, L)
    probt = Array{eltype(α1)}(undef, N, T-1)
    for t = reverse(1:T-1)
        probt[:,t] = llcontrib(state[:,t,:], wage[:,t], choice[:,t], MC_ϵ[:,:,t], Emaxall[t+1], lambda, α1, α2, β, γ0, L)
    end
    prob = hcat(probt, probT)
    ll = -mean(clamp!(log.(prob),-1.0e20, 1.0e20))
end
function llcontrib(fstate::Array{Int64,2}, wage::Array{Float64,1}, fchoice::Array{Int64,1}, MC_ϵ::Array, fEmax::OrderedDict, lambda::Float64, α1::Array, α2::Array, β::Array, γ0, L::Array)
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
            logit = exp.((v1_obs .- vmax)./lambda) .+ exp.((v2 .- vmax)./lambda) .+ exp.((v3 .- vmax)./lambda) .+ exp.((v4 .- vmax)./lambda)
            probt = exp.((v1_obs .- vmax)./lambda)./logit
            denw = pdf.(Normal(0,L[1,1]), log(wage[i]) .- R1(state[1],state[2],state[3],0,α1))
        elseif choice == 2
            v2_obs = wage[i] .+ 0.95*fEmax[state+[0,0,1,-state[4]]]
            vmax = max.(v1, v2_obs, v3, v4)
            logit = exp.((v1 .- vmax)./lambda) .+ exp.((v2_obs .- vmax)./lambda) .+ exp.((v3 .- vmax)./lambda) .+ exp.((v4 .- vmax)./lambda)
            probt = exp.((v2_obs .- vmax)./lambda)./logit
            denw = pdf.(Normal(0,max(sqrt(L[2,1]^2 + L[2,2]^2), 1.0e-8)), log(wage[i]) .- R2(state[1],state[2],state[3],0,α2))
        elseif choice == 3
            vmax = max.(v1, v2, v3, v4)
            logit = exp.((v1 .- vmax)./lambda) .+ exp.((v2 .- vmax)./lambda) .+ exp.((v3 .- vmax)./lambda) .+ exp.((v4 .- vmax)./lambda)
            probt = exp.((v3 .- vmax)./lambda)./logit
            denw = 1.0
        elseif choice == 4
            vmax = max.(v1, v2, v3, v4)
            logit = exp.((v1 .- vmax)./lambda) .+ exp.((v2 .- vmax)./lambda) .+ exp.((v3 .- vmax)./lambda) .+ exp.((v4 .- vmax)./lambda)
            probt = exp.((v4 .- vmax)./lambda)./logit
            denw = 1.0
        end
        prob[i] = mean(probt .* denw)
    end
    return prob
end

function llcontribT(fstate::Array{Int64,2}, wage::Array{Float64,1}, fchoice::Array{Int64,1}, MC_ϵ::Array, lambda::Float64, α1::Array, α2::Array, β::Array, γ0, L::Array)
    prob = Array{eltype(α1)}(undef,size(wage)[1])
    for i = 1:size(wage)[1]
        state = fstate[i,:]
        choice = fchoice[i]
        r1 = exp.(R1(state[1],state[2],state[3],MC_ϵ[1,:],α1))
        r2 = exp.(R2(state[1],state[2],state[3],MC_ϵ[2,:],α2))
        if state[1]  < 20
            r3 = R3(state[1],state[4],MC_ϵ[3,:],β)
        else
            r3 = R3(state[1],state[4],MC_ϵ[3,:],β) .- 50000.0
        end
        r4 = R4(MC_ϵ[4,:],γ0)
        if choice == 1
            rmax = max.(wage[i], r2, r3, r4)
            logit = exp.((wage[i] .- rmax)./lambda) .+ exp.((r2 .- rmax)./lambda) .+ exp.((r3 .- rmax)./lambda) .+ exp.((r4 .- rmax)./lambda)
            probt = exp.((wage[i] .- rmax)./lambda)./logit
            denw = pdf.(Normal(0,L[1,1]), log(wage[i]) .- R1(state[1],state[2],state[3],0,α1))
        elseif choice == 2
            rmax = max.(r1, wage[i], r3, r4)
            logit = exp.((r1 .- rmax)./lambda) .+ exp.((wage[i] .- rmax)./lambda) .+ exp.((r3 .- rmax)./lambda) .+ exp.((r4 .- rmax)./lambda)
            probt = exp.((wage[i] .- rmax)./lambda)./logit
            denw = pdf.(Normal(0,max(sqrt(L[2,1]^2 + L[2,2]^2), 1.0e-8)), log(wage[i]) .- R2(state[1],state[2],state[3],0,α2))
        elseif choice == 3
            rmax = max.(r1, r2, r3, r4)
            logit = exp.((r1 .- rmax)./lambda) .+ exp.((r2 .- rmax)./lambda) .+ exp.((r3 .- rmax)./lambda) .+ exp.((r4 .- rmax)./lambda)
            probt = exp.((r3 .- rmax)./lambda)./logit
            denw = 1.0
        elseif choice == 4
            rmax = max.(r1, r2, r3, r4)
            logit = exp.((r1 .- rmax)./lambda) .+ exp.((r2 .- rmax)./lambda) .+ exp.((r3 .- rmax)./lambda) .+ exp.((r4 .- rmax)./lambda)
            probt = exp.((r4 .- rmax)./lambda)./logit
            denw = 1.0
        end
        #probt[isnan.(probt)] .= 1.0
        prob[i] = mean(probt) .* denw
    end
    return prob
end




function makeInitialGuess(param::Int64)
        include("Parameters$param.jl")
        sigma = [p.σ11^2 p.σ12 p.σ13 p.σ14;
                p.σ12 p.σ22^2 p.σ23 p.σ24;
                p.σ13 p.σ23 p.σ33^2 p.σ34;
                p.σ14 p.σ24 p.σ34 p.σ44^2]
        A = cholesky(sigma)
        LT = A.L
        θ = [p.α10, p.α11, p.α12, p.α13, p.α14, p.α15,
        p.α20, p.α21, p.α22, p.α23, p.α24, p.α25,
        p.β0, p.β1, p.β2, p.γ0, log(LT[1,1]), log(LT[2,2]), log(LT[3,3]), log(LT[4,4]),
        LT[2,1], LT[3,1], LT[4,1], LT[3,2], LT[4,2], LT[4,3]]
        return θ
end


function respydata()
        df = DataFrame(load("output/respy_data.csv"))
        df.Identifier .+= 1
        df.Period .+= 1
        state = Array{Int64}(undef, maximum(df.Identifier), maximum(df.Period), 4)
        wage = Array{Float64}(undef, maximum(df.Identifier), maximum(df.Period))
        choice = Array{Int64}(undef, maximum(df.Identifier), maximum(df.Period))
        df.lag = zeros(Int64,40000)
        df.lag[df.Lagged_Choice .== 3] .= 1
        for t = reverse(1:maximum(df.Period))
                tempdf = df[df.Period .== t,:]
                state[:,t,:] = [tempdf.Years_Schooling tempdf.Experience_A tempdf.Experience_B tempdf.lag]
                wage[:,t] = tempdf.Wage
                choice[:,t] = tempdf.Choice
        end
        return state, wage, choice
end
