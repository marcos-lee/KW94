# Sen Lu. This script is used to define the Keane and Wolpin model
#--- add processor
using Distributed
numb_procs = 11
# numb_procs = 5
addprocs(numb_procs)
nprocs()

#--- import packages and define functions
@everywhere using Distributions

# export KW94,KW94_primitives,feasibleSet,StateSpace,IS_Schooling,Value_update_T,find_feasible_indx

@everywhere mutable struct KW94
    """
    Return specifications:
    R_1(t) = w_1t = exp(α_10 + α_11*s_t + α_12*x1_t - α_13*(x1_t)^2 + α_14*x2_t -α_15*(x2_t)^2 + ϵ_1t)
    R_2(t) = w_2t = exp(α_20 + α_21*s_t + α_22*x1_t - α_23*(x1_t)^2 + α_24*x2_t -α_25*(x2_t)^2 + ϵ_2t)
    R_3(t) = β_0 -β_1*Bool(s_t>=13)-β_2(1-d_3(t-1)) + ϵ_3t
    R_4(t) = γ_0 + ϵ_4t

    """
    N::Int64    # Number of individuals
    T::Int64    # Number of periods
    N_draws::Int64  # Number of draws to calculate numerical integration
    α_10::Float64   # return parameters of occupation 1
    α_11::Float64   # return parameters of occupation 1
    α_12::Float64   # return parameters of occupation 1
    α_13::Float64   # return parameters of occupation 1
    α_14::Float64   # return parameters of occupation 1
    α_15::Float64   # return parameters of occupation 1
    α_20::Float64   # return parameters of occupation 2
    α_21::Float64   # return parameters of occupation 2
    α_22::Float64   # return parameters of occupation 2
    α_23::Float64   # return parameters of occupation 2
    α_24::Float64   # return parameters of occupation 2
    α_25::Float64   # return parameters of occupation 2
    β_0::Float64    # return parameters of study
    β_1::Float64    # return parameters of study
    β_2::Float64    # return parameters of study
    γ_0::Float64    # return parameters of staying at home
    R_1::Function   # return function of occupation 1
    R_2::Function   # return function of occupation 2
    R_3::Function   # return function of schooling
    R_4::Function   # return function of staying at home
    dist_eps::MultivariateNormal
    S_max::Int64       # Maximum of additional years of schooling
    Init_state::Vector{Int64}   # Initial state variables
end

@everywhere function KW94_primitives(;N::Int64=1000,
        T::Int64=40,    # Number of periods
        N_draws::Int64=100000,  # Number of draws to calculate numerical integration
        α_10::Float64=9.21,   # return parameters of occupation 1
        α_11::Float64=0.038,   # return parameters of occupation 1
        α_12::Float64=0.033,   # return parameters of occupation 1
        α_13::Float64=0.0005,   # return parameters of occupation 1
        α_14::Float64=0.0,   # return parameters of occupation 1
        α_15::Float64=0.0,   # return parameters of occupation 1
        α_20::Float64=8.48,   # return parameters of occupation 2
        α_21::Float64=0.07,   # return parameters of occupation 2
        α_22::Float64=0.067,   # return parameters of occupation 2
        α_23::Float64=0.001,   # return parameters of occupation 2
        α_24::Float64=0.022,   # return parameters of occupation 2
        α_25::Float64=0.0005,   # return parameters of occupation 2
        β_0::Float64=0.0,    # return parameters of study
        β_1::Float64=0.0,    # return parameters of study
        β_2::Float64=4000.0,    # return parameters of study
        γ_0::Float64=17750.0,    # return parameters of staying at home
        σ_11_sqrt::Float64=0.2,      # primitive of covariance matrix
        σ_22_sqrt::Float64=0.25,      # primitive of covariance matrix
        σ_33_sqrt::Float64=1500.0,      # primitive of covariance matrix
        σ_44_sqrt::Float64=1500.0,      # primitive of covariance matrix
        σ_12::Float64=0.0,      # primitive of covariance matrix
        σ_13::Float64=0.0,      # primitive of covariance matrix
        σ_14::Float64=0.0,      # primitive of covariance matrix
        σ_23::Float64=0.0,      # primitive of covariance matrix
        σ_24::Float64=0.0,      # primitive of covariance matrix
        σ_34::Float64=0.0,      # primitive of covariance matrix
        S_max::Int64=10,
        Init_state::Vector{Int64}=[10,0,0,0])  # entrance: 1- s_t; 2- x1_t; 3- x2_t; 4- d3_(t-1)

    function R_1(ste::Vector{Int64}, t::Int64, ϵ::Vector{Float64}) # entrance: 1- s_t; 2- x1_t; 3- x2_t; 4- d3_(t-1)
        w_1t = exp(α_10 + α_11*ste[1] + α_12*ste[2] - α_13*(ste[2])^2 + α_14*ste[3] -α_15*(ste[3])^2 + ϵ[1])
        return w_1t
    end

    function R_2(ste::Vector{Int64}, t::Int64, ϵ::Vector{Float64}) # entrance: 1- s_t; 2- x1_t; 3- x2_t; 4- d3_(t-1)
        w_2t = exp(α_20 + α_21*ste[1] + α_22*ste[3] - α_23*(ste[3])^2 + α_24*ste[2] -α_25*(ste[2])^2 + ϵ[2])
        return w_2t
    end

    schooling_year = Init_state[1] + S_max
    function R_3(ste::Vector{Int64}, t::Int64, ϵ::Vector{Float64})
        if ste[1]<schooling_year
            R3 = β_0 -β_1*Bool(ste[1]>=13)- β_2*(1-ste[4]) + ϵ[3]
        else
            R3 = - β_2
        end
        return R3
    end

    function R_4(ste::Vector{Int64}, t::Int64, ϵ::Vector{Float64})
        R4 = γ_0 + ϵ[4]
        return R4
    end
    Σ = [σ_11_sqrt^2 σ_12 σ_13 σ_14;
        σ_12 σ_22_sqrt^2 σ_23 σ_24;
        σ_13 σ_23 σ_33_sqrt^2 σ_34;
        σ_14 σ_24 σ_34 σ_44_sqrt^2]
    dist_eps = MultivariateNormal(zeros(4),Σ)

    #N,T,N_draws,α_10,α_11,α_12,α_13,α_14,α_15,α_20,α_21,α_22,α_23,α_24,α_25,β_0,β_1,β_2,γ_0,R_1,R_2,R_3,R_4,dist_eps,S_max,Init_state
    m = KW94(N,T,N_draws,α_10,α_11,α_12,α_13,α_14,α_15,α_20,α_21,α_22,α_23,α_24,α_25,β_0,β_1,β_2,γ_0,R_1,R_2,R_3,R_4,dist_eps,S_max,Init_state)
    return m
end

@everywhere function feasibleSet(st::Vector{Int64}, kwm::KW94)
    up = Vector{Vector{Int64}}(undef,4)
    up[1] = [1,0,0,0]   #studying
    up[2] = [0,1,0,0]   #occupation 1
    up[3] = [0,0,1,0]   #occupation 2
    up[4] = [0,0,0,0]   #staying at home
    if sum([st[1] st[2] st[3]]) < (kwm.T + kwm.Init_state[1])
        if st[1] < (kwm.Init_state[1] + kwm.S_max)
            output = Vector{Vector{Int64}}(undef,4)
            for i=1:4
                output[i] = st + up[i]
                if output[i][1] - st[1] == 1
                    output[i][4] = 1
                else
                    output[i][4] = 0
                end
            end
        else
            output = Vector{Vector{Int64}}(undef,3)
            for i = 1:3
                output[i] = st + up[i+1]
                output[i][4] = 0
            end
        end
    else
        st[4] = 0
        output = st
    end
    return output
end

@everywhere function StateSpace(kwm::KW94)
    Domain_set = Dict{Int64,Vector}()
    Domain_set[1] = kwm.Init_state
    D = Vector{Vector{Int64}}(undef,0)
    Domain_set[2] = vcat(feasibleSet(kwm.Init_state,kwm),D)
    for t = 2:(kwm.T-1)
        D = Vector{Vector{Int64}}(undef,0)
        for i in Domain_set[t]
            D = vcat(feasibleSet(i,kwm),D)
        end
        Domain_set[t+1] = unique(D)
    end
    return Domain_set
end

@everywhere function IS_Schooling(st::Vector{Int64},kwm::KW94)
    output =  (st[1] < (kwm.Init_state[1] + kwm.S_max))
    return output
end

@everywhere function Value_update_T(st::Vector{Int64})
    global mont_carlo_eps;
    global kwm;
    # global mont_carlo_eps
    V_sp_mtvec=Array{Float64}(undef,kwm.N_draws,4)
    for i=1:kwm.N_draws
        V_sp_mtvec[i,1] = kwm.R_1(st,kwm.T,mont_carlo_eps[i,:])
        V_sp_mtvec[i,2] = kwm.R_2(st,kwm.T,mont_carlo_eps[i,:])
        V_sp_mtvec[i,3] = kwm.R_3(st,kwm.T,mont_carlo_eps[i,:])
        V_sp_mtvec[i,4] = kwm.R_4(st,kwm.T,mont_carlo_eps[i,:])
    end
    V_mtvec = maximum.([V_sp_mtvec[i,:] for i =1:kwm.N_draws])
    V_sp = mean(V_mtvec)
    return V_sp
end

@everywhere function pmap_Value_update_T(st_vec::Vector{Vector{Int64}})
    np = nprocs()  # determine the number of processes available
    n = length(st_vec)
    results = Vector{Float64}(undef,n)
    i = 1
    # function to produce the next work item from the queue.
    # in this case it's just an index.
    nextidx() = (idx=i; i+=1; idx)
    @sync begin
        for p=1:np
            if p != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        if idx > n
                            break
                        end
                        results[idx] = remotecall_fetch(Value_update_T,p,st_vec[idx])
                    end
                end
            end
        end
    end
    results
end

@everywhere function find_feasible_indx(st::Vector{Int64},t::Int64,kwm::KW94,V_domain::Dict,V_domain_card::Vector{Int64})
    # in period t, given a state st, find indx of all feasible next period states.
    target_domain = V_domain[t+1]
    states_cache = feasibleSet(st,kwm)
    n_options = size(states_cache)[1]
    indx_cache = Vector{Int64}(undef,n_options)
    for i=1:n_options
        target_cache = states_cache[i]
        for j=1:V_domain_card[t+1]
            if target_domain[j]==target_cache
                indx_cache[i] = j
            end
        end
    end
    return indx_cache
end

#=

@everywhere function Value_update_T(st::Vector{Int64},kwm::KW94,mont_carlo_eps::Array{Float64})
    global mont_carlo_eps
    # global mont_carlo_eps
    V_sp_mtvec=Array{Float64}(undef,kwm.N_draws,4)
    for i=1:kwm.N_draws
        V_sp_mtvec[i,1] = kwm.R_1(st,kwm.T,mont_carlo_eps[i,:])
        V_sp_mtvec[i,2] = kwm.R_2(st,kwm.T,mont_carlo_eps[i,:])
        V_sp_mtvec[i,3] = kwm.R_3(st,kwm.T,mont_carlo_eps[i,:])
        V_sp_mtvec[i,4] = kwm.R_4(st,kwm.T,mont_carlo_eps[i,:])
    end
    V_mtvec = maximum.([V_sp_mtvec[i,:] for i =1:kwm.N_draws])
    V_sp = mean(V_mtvec)
    return V_sp
end


@everywhere function pmap_Value_update_T(st_vec::Vector{Vector{Int64}},kwm::KW94,mont_carlo_eps::Array{Float64})
    np = nprocs()  # determine the number of processes available
    n = length(st_vec)
    results = Vector{Float64}(undef,n)
    i = 1
    # function to produce the next work item from the queue.
    # in this case it's just an index.
    nextidx() = (idx=i; i+=1; idx)
    @sync begin
        for p=1:np
            if p != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        if idx > n
                            break
                        end
                        results[idx] = remotecall_fetch(Value_update_T,p,st_vec[idx],kwm,mont_carlo_eps)
                    end
                end
            end
        end
    end
    results
end

function update_step_T(V_value::Dict,V_domain::Dict,kwm::KW94,mont_carlo_eps::Array{Float64})
    sp_vec = V_domain[kwm.T]
    # sp_vec = sp_vec[1:4]
    V_final = pmap_Value_update_T(sp_vec,kwm,mont_carlo_eps)
    V_value[kwm.T]=V_final
end

function Emax(st::Vector{Int64},t::Int64,V_value::Dict,V_domain::Dict)
    target_domain = V_domain[t]
    indx_cache = 0
    for j=1:size(target_domain,1)
        if target_domain[j]==st
            indx_cache = j
        end
    end
    return V_value[t][indx_cache]
end

@everywhere function value_update_t(st::Vector{Int64},t::Int64,kwm::KW94,mont_carlo_eps::Array{Float64},V_value::Dict,V_domain::Dict,V_domain_card::Vector{Int64})
    states_cache = feasibleSet(st,kwm)
    #If cardi = 4, then order 1:studing ; 2: occup 1 ; 3: occup 2 ;  4: staying at home
    #If cardi = 3, then order 1:occup 1; 2:occup2 ; 3:staying at home.
    states_indx_cache = find_feasible_indx(st,t,kwm,V_domain,V_domain_card)
    # states_value_cache1 = Emax.(states_cache,t+1,V_value,V_domain)
    states_value_cache2 = [(V_value[t+1][k]) for k in states_indx_cache]
    if IS_Schooling(st,kwm)
        V_sp_mtvec=Array{Float64}(undef,kwm.N_draws,4)
        for i=1:kwm.N_draws
            V_sp_mtvec[i,1] = kwm.R_1(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[2]    #occupation 1
            V_sp_mtvec[i,2] = kwm.R_2(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[3]    #occupation 2
            V_sp_mtvec[i,3] = kwm.R_3(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[1]    #studying
            V_sp_mtvec[i,4] = kwm.R_4(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[4]    #staying at home
        end
        V_mtvec = maximum.([V_sp_mtvec[i,:] for i =1:kwm.N_draws])
        V_sp = mean(V_mtvec)
    else
        V_sp_mtvec=Array{Float64}(undef,kwm.N_draws,3)
        for i=1:kwm.N_draws
            V_sp_mtvec[i,1] = kwm.R_1(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[1]    #occupation 1
            V_sp_mtvec[i,2] = kwm.R_2(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[2]    #occupation 2
            V_sp_mtvec[i,3] = kwm.R_4(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[3]    #staying at home
        end
        V_mtvec = maximum.([V_sp_mtvec[i,:] for i =1:kwm.N_draws])
        V_sp = mean(V_mtvec)
    end
    return V_sp
end

@everywhere function pmap_Value_update_t(st_vec::Vector{Vector{Int64}},t::Int64,kwm::KW94,mont_carlo_eps::Array{Float64},V_value::Dict,V_domain::Dict,V_domain_card::Vector{Int64})
    np = nprocs()  # determine the number of processes available
    n = length(st_vec)
    results = Vector{Float64}(undef,n)
    i = 1
    # function to produce the next work item from the queue.
    # in this case it's just an index.
    nextidx() = (idx=i; i+=1; idx)
    @sync begin
        for p=1:np
            if p != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        if idx > n
                            break
                        end
                        results[idx] = remotecall_fetch(value_update_t,p,st_vec[idx],t,kwm,mont_carlo_eps,V_value,V_domain,V_domain_card)
                    end
                end
            end
        end
    end
    results
end
=#
function update_step_T()
    global V_value;
    global V_domain;
    global kwm;
    sp_vec = V_domain[kwm.T]
    # sp_vec = sp_vec[1:4]
    V_final = pmap_Value_update_T(sp_vec)
    V_value[kwm.T]=V_final
end

@everywhere function value_update_t_v2(st::Vector{Int64},t::Int64)
    global kwm;
    global mont_carlo_eps;
    global V_value;
    global V_domain;
    global V_domain_card;
    states_cache = feasibleSet(st,kwm)
    #If cardi = 4, then order 1:studing ; 2: occup 1 ; 3: occup 2 ;  4: staying at home
    #If cardi = 3, then order 1:occup 1; 2:occup2 ; 3:staying at home.
    states_indx_cache = find_feasible_indx(st,t,kwm,V_domain,V_domain_card)
    # states_value_cache1 = Emax.(states_cache,t+1,V_value,V_domain)
    states_value_cache2 = [(V_value[t+1][k]) for k in states_indx_cache]
    if IS_Schooling(st,kwm)
        V_sp_mtvec=Array{Float64}(undef,kwm.N_draws,4)
        for i=1:kwm.N_draws
            V_sp_mtvec[i,1] = kwm.R_1(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[2]    #occupation 1
            V_sp_mtvec[i,2] = kwm.R_2(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[3]    #occupation 2
            V_sp_mtvec[i,3] = kwm.R_3(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[1]    #studying
            V_sp_mtvec[i,4] = kwm.R_4(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[4]    #staying at home
        end
        V_mtvec = maximum.([V_sp_mtvec[i,:] for i =1:kwm.N_draws])
        V_sp = mean(V_mtvec)
    else
        V_sp_mtvec=Array{Float64}(undef,kwm.N_draws,3)
        for i=1:kwm.N_draws
            V_sp_mtvec[i,1] = kwm.R_1(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[1]    #occupation 1
            V_sp_mtvec[i,2] = kwm.R_2(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[2]    #occupation 2
            V_sp_mtvec[i,3] = kwm.R_4(st,t,mont_carlo_eps[i,:]) + 0.95*states_value_cache2[3]    #staying at home
        end
        V_mtvec = maximum.([V_sp_mtvec[i,:] for i =1:kwm.N_draws])
        V_sp = mean(V_mtvec)
    end
    return V_sp
end

@everywhere function pmap_Value_update_t_v2(st_vec::Vector{Vector{Int64}},t::Int64)
    np = nprocs()  # determine the number of processes available
    n = length(st_vec)
    results = Vector{Float64}(undef,n)
    i = 1
    # function to produce the next work item from the queue.
    # in this case it's just an index.
    nextidx() = (idx=i; i+=1; idx)
    @sync begin
        for p=1:np
            if p != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        if idx > n
                            break
                        end
                        results[idx] = remotecall_fetch(value_update_t_v2,p,st_vec[idx],t)
                    end
                end
            end
        end
    end
    results
end

function update_step_t(t::Int64)
    global V_value;
    global V_domain;
    if t>1
        sp_vec = V_domain[t]
        # sp_vec = sp_vec[1:4]
        V_final = pmap_Value_update_t_v2(sp_vec,t)
        V_value[t]=V_final
    else
        sp_vec = V_domain[t]
        # sp_vec = sp_vec[1:4]
        V_final = value_update_t_v2(sp_vec,t)
        V_value[t]=V_final
    end
end

function update_step_t_test(t::Int64)
    global V_value;
    global V_domain;
    sp_vec = V_domain[t][1:300]
    # sp_vec = sp_vec[1:4]
    V_final = pmap_Value_update_t_v2(sp_vec,t)
    V_value[t]=V_final
end

function update_v_value()
    global kwm;
    T = kwm.T;
    println("\n Backward induction \n")
    println("== Iteration t=$T ==\n")
    update_step_T()
    process_v_value(T)
    for t_indx = 1:(T-1)
        t = T-t_indx
        println("== Iteration t=$t ==\n")
        update_step_t(t)
        process_v_value(t)
    end
end


function processor_init()
    global kwm;
    global mont_carlo_eps;
    global V_domain;
    global V_domain_card;
    numb_procs= nprocs()
    let kwm = kwm
       for i=2:numb_procs
          remotecall_fetch(()->kwm,i)
       end
    end
    let mont_carlo_eps = mont_carlo_eps
       for i=2:numb_procs
          remotecall_fetch(()->mont_carlo_eps,i)
       end
    end

    let V_domain = V_domain
       for i=2:numb_procs
          remotecall_fetch(()->V_domain,i)
       end
    end
    let V_domain_card = V_domain_card
       for i=2:numb_procs
          remotecall_fetch(()->V_domain_card,i)
       end
    end
    for i=2:numb_procs
        remotecall_fetch(()->(kwm.T,mont_carlo_eps[1],V_domain[1],V_domain_card[1]),i)
    end
end

function process_v_value(t::Int64)
    global V_value;
    numb_procs = nprocs()
    let V_value = V_value
       for i=2:numb_procs
          remotecall_fetch(()->V_value,i)
       end
    end
    for i=2:numb_procs
        remotecall_fetch(()->V_value[t][1],i)
    end
end

V_domain[2]
@everywhere function agent_choice(individual_eps::Array{Float64,2})
    global kwm;
    global V_domain;
    global V_domain_card;
    global V_value;
    agent_output = Array{Int64,2}(undef,kwm.T,4)
    s0 = kwm.Init_state
    st = kwm.Init_state
    for t= 1:(kwm.T-1)
        if IS_Schooling(st,kwm)
            option_set = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            states_indx_cache = find_feasible_indx(st,t,kwm,V_domain,V_domain_card)
            states_sp_cache = [(V_domain[t+1][k]) for k in states_indx_cache]
            states_value_cache2 = [(V_value[t+1][k]) for k in states_indx_cache]
            option_value = Vector{Float64}(undef,4)
            option_value[1] = kwm.R_1(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[2]
            option_value[2] = kwm.R_2(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[3]
            option_value[3] = kwm.R_3(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[1]
            option_value[4] = kwm.R_4(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[4]
            option_cache = findmax(option_value)[2]
            agent_output[t,:] = option_set[option_cache]
            st = states_sp_cache[option_cache]
        else
            option_set = [[1,0,0,0],[0,1,0,0],[0,0,0,1]]
            states_indx_cache = find_feasible_indx(st,t,kwm,V_domain,V_domain_card)
            states_sp_cache = [(V_domain[t+1][k]) for k in states_indx_cache]
            states_value_cache2 = [(V_value[t+1][k]) for k in states_indx_cache]
            option_value = Vector{Float64}(undef,3)
            option_value[1] = kwm.R_1(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[1]
            option_value[2] = kwm.R_2(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[2]
            option_value[3] = kwm.R_4(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[3]
            option_cache = findmax(option_value)[2]
            agent_output[t,:] = option_set[option_cache]
            st = states_sp_cache[option_cache]
        end
    end
    return agent_output
end

function pmap_agent_choice(eps_vec::Vector{Array{Float64,2}})
   np = nprocs()  # determine the number of processes available
   n = length(eps_vec)
   results = Vector{Array{Int64,2}}(undef,n)
   i = 1
   # function to produce the next work item from the queue.
   # in this case it's just an index.
   nextidx() = (idx=i; i+=1; idx)
   @sync begin
       for p=1:np
           if p != myid() || np == 1
               @async begin
                   while true
                       idx = nextidx()
                       if idx > n
                           break
                       end
                       results[idx] = remotecall_fetch(agent_choice,p,eps_vec[idx])
                   end
               end
           end
       end
   end
   results
end


#--- cutoff point

#====================
=== working area ====
====================#

#--- creat basic variables
kwm = KW94_primitives(N_draws=1000)
@time V_domain = StateSpace(kwm)
V_domain_card = [size(V_domain[i])[1] for i=1:kwm.T]
V_domain_card[1] = 1
mont_carlo_eps = rand(kwm.dist_eps,kwm.N_draws)
mont_carlo_eps = mont_carlo_eps'
mont_carlo_eps = copy(mont_carlo_eps)

#--- Update value function
# create v value dict
V_value =Dict()
@time processor_init()
@time update_v_value()

#--- simulate individuals
#=
@everywhere function agent_choice(individual_eps::Array{Float64,2})
    global kwm;
    global V_domain;
    global V_domain_card;
    global V_value;
    agent_output = Array{Int64,2}(undef,kwm.T,4)
    s0 = kwm.Init_state
    st = kwm.Init_state
    for t= 1:(kwm.T-1)
        if IS_Schooling(st,kwm)
            option_set = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            states_indx_cache = find_feasible_indx(st,t,kwm,V_domain,V_domain_card)
            states_sp_cache = [(V_domain[t+1][k]) for k in states_indx_cache]
            states_value_cache2 = [(V_value[t+1][k]) for k in states_indx_cache]
            option_value = Vector{Float64}(undef,4)
            option_value[1] = kwm.R_1(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[2]
            option_value[2] = kwm.R_2(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[3]
            option_value[3] = kwm.R_3(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[1]
            option_value[4] = kwm.R_4(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[4]
            option_cache = findmax(option_value)[2]
            agent_output[t,:] = option_set[option_cache]
            st = states_sp_cache[option_cache]
        else
            option_set = [[1,0,0,0],[0,1,0,0],[0,0,0,1]]
            states_indx_cache = find_feasible_indx(st,t,kwm,V_domain,V_domain_card)
            states_sp_cache = [(V_domain[t+1][k]) for k in states_indx_cache]
            states_value_cache2 = [(V_value[t+1][k]) for k in states_indx_cache]
            option_value = Vector{Float64}(undef,3)
            option_value[1] = kwm.R_1(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[1]
            option_value[2] = kwm.R_2(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[2]
            option_value[3] = kwm.R_4(st,t,individual_eps[:,t]) + 0.95*states_value_cache2[3]
            option_cache = findmax(option_value)[2]
            agent_output[t,:] = option_set[option_cache]
            st = states_sp_cache[option_cache]
        end
    end
    return agent_output
end

function pmap_agent_choice(eps_vec::Vector{Array{Float64,2}})
   np = nprocs()  # determine the number of processes available
   n = length(eps_vec)
   results = Vector{Array{Int64,2}}(undef,n)
   i = 1
   # function to produce the next work item from the queue.
   # in this case it's just an index.
   nextidx() = (idx=i; i+=1; idx)
   @sync begin
       for p=1:np
           if p != myid() || np == 1
               @async begin
                   while true
                       idx = nextidx()
                       if idx > n
                           break
                       end
                       results[idx] = remotecall_fetch(agent_choice,p,eps_vec[idx])
                   end
               end
           end
       end
   end
   results
end

individual_eps = rand(kwm.dist_eps,40)
@time agent_choice(individual_eps)
=#


individual_eps_vec = Vector{Array{Float64,2}}(undef,kwm.N)
for i = 1:kwm.N
    individual_eps_vec[i] = rand(kwm.dist_eps,kwm.T)
end
individual_options_vec = pmap_agent_choice(individual_eps_vec)

test =zeros(kwm.T,4)
for  i = 1:kwm.N
    global test
    test += individual_options_vec[i]
end
test
