#Julia 1.0.2
function feasibleSet(st::Vector{Int64},up::Array)
    if sum([st[1] st[2] st[3]]) < 50
        if st[1] < 20
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


up = Vector{Vector{Int64}}(undef,4)
up[1] = [1,0,0,0]
up[2] = [0,1,0,0]
up[3] = [0,0,1,0]
up[4] = [0,0,0,0]

st = [20,20,10,1]


feasibleSet(st,up)
feasibleSet1(st,up)


function StateSpace(st::Vector{Int64},T::Int64)
    Domain_set = Dict{Int64,Vector}()
    Domain_set[1] = st
    D = Vector{Vector{Int64}}(undef,0)
    Domain_set[2] = vcat(feasibleSet(st,up),D)
    for t = 2:T-1
        D = Vector{Vector{Int64}}(undef,0)
        for i in Domain_set[t]
            D = vcat(feasibleSet(i,up),D)
        end
        Domain_set[t+1] = unique(D)
    end
    return Domain_set
end

st = [10,0,0,0]
@time Domain_set = StateSpace(st,41)

Domain_set[41]
