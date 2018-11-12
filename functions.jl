function R1(s,x1,x2,ϵ1)
    r1 = p.α10 + p.α11 * s + p.α12 * x1 - p.α13 * x1^2 + p.α14 *x2 - p.α15 *x2^2 .+ ϵ1
end

function R2(s,x1,x2,ϵ2)
    r2 = p.α20 + p.α21 * s + p.α22 * x2 - p.α23 * x2^2 + p.α24 *x1 - p.α25 *x1^2 .+ ϵ2
end

function R3(s,slag,ϵ3)
    r3 = p.β0 - p.β1 * (s >= 13) - p.β2 * (1 - slag) .+ ϵ3
end

function R4(ϵ4)
    r4 = p.γ0 .+ ϵ4
end
