module Model
export update_cap!, update_hyp!, compute_v!, compute_ϕ!, compute_sidewalls!, build_cache, build_cache_cap, build_cache_hyp, MODE, nᵤ
using SparseArrays, StaticArrays, LinearAlgebra, UnPack, Reexport, FLoops
using UnPack

const nᵤ = 7
const MODE = :full # type of augumented formulations
# const MODE = :simple
# const MODE = nothing

include("./Helpers.jl")
include("./Grids.jl")
include("./Ops.jl")
include("./Hyperbolic.jl")
include("./NonConservative.jl")


@reexport using .Helpers
@reexport using .Grids
@reexport using .Ops
@reexport using .Hyperbolic
@reexport using .NonConservative

@static if MODE == :full
    function compute_v!(vx, vy, h, κ, Δx, Δy, n₁, n₂, i, j)# definition of w= (√2κ/√h)(√(1+||∇h||^2)+1)^(-1/2) ∇h
        vx[i, j] = √κ * √(2.0 / (1.0 + √(1.0 + (@dx(h))^2 + (@dy(h))^2))) * (@dx(h)) / √max(h[i, j], 0.0)
        vy[i, j] = √κ * √(2.0 / (1.0 + √(1.0 + (@dx(h))^2 + (@dy(h))^2))) * (@dy(h)) / √max(h[i, j], 0.0)
        return
    end
elseif MODE == :simple
    function compute_v!(vx, vy, h, κ, Δx, Δy, n₁, n₂, i, j)
        vx[i, j] = @. √κ * (@dx(h)) / √max(h[i, j], 0.0)
        vy[i, j] = @. √κ * (@dy(h)) / √max(h[i, j], 0.0)
        return
    end
else
    function compute_v!(vx, vy, h, κ, Δx, Δy, n₁, n₂, i, j)
        return
    end
end

function compute_v!(vx, vy, h, κ, Δx, Δy, n₁, n₂; executor=ThreadedEx())#Evaluating W
    @floop executor for I in CartesianIndices((n₁, n₂))
        compute_v!(vx, vy, h, κ, Δx, Δy, n₁, n₂, Tuple(I)...)
    end
end

function compute_ϕ!(h, ux, uy, ϕx, ϕy, τx, τy, ls, i, j) # definition of ϕ= ((u ⊗ u) / 3h^2) - 1 / 12h^2 * ((u ⊗ u) - h^2 * (τe ⊗ τe) / 4)
    u = @SVector [ux[i, j], uy[i, j]]
    τ = @SVector [0.0, 0.0]
#    ϕ = (u ⊗ u) / 3h[i, j]^2 - 1 / 12h[i, j]^2 * ((u ⊗ u) - h[i, j]^2 * (τ ⊗ τ) / 4)
    #ϕ = (τ ⊗ τ) 
    ϕ = (u ⊗ u)/(5*(3*ls+h[i, j])^2)
    #ϕ1 = @SVector [ϕx[i, j], ϕy[i, j]]
    #ϕx[i, j] = √(5*ϕ[1, 1])
    #ϕy[i, j] = √(5*ϕ[2, 2])
    return
end


function compute_ϕ!(h, ux, uy, ϕx, ϕy, τx, τy, ls; executor=ThreadedEx()) #Evaluating ϕ
    @floop executor for I in CartesianIndices(h)
        compute_ϕ!(h, ux, uy, ϕx, ϕy, τx, τy, ls, Tuple(I)...)
    end
end

function compute_sidewalls!(h, ux, uy, vx, vy, ϕx, ϕy, n₁, n₂, i, j) # definition of ϕ= ((u ⊗ u) / 3h^2) - 1 / 12h^2 * ((u ⊗ u) - h^2 * (τe ⊗ τe) / 4)
    uy[i,1]=0
    uy[i,n₂]=0
    ϕy[i,1]=0
    ϕy[i,n₂]=0
    vy[i,1]=0
    vy[i,n₂]=0
    return
end

function compute_sidewalls!(h, ux, uy, vx, vy, ϕx, ϕy, n₁, n₂; executor=ThreadedEx()) #Evaluating ϕ
    @floop executor for I in CartesianIndices(h)
        compute_sidewalls!(h, ux, uy, vx, vy, ϕx, ϕy, n₁, n₂, Tuple(I)...)
    end
end


function build_cache_hyp(T, n₁, n₂)#initialising arrays
    x = T()
    @preallocate U, Fx, Fy = similar(x, (n₁, n₂, nᵤ))
    @preallocate dUhypx, dUhypy = similar(x, (n₁ * n₂ * nᵤ))

    @preallocate Ue₋, Ue₊, Uw₋, Uw₊, Us₋, Us₊, Un₋, Un₊ = similar(x, (n₁, n₂, nᵤ))
    @preallocate ce₋, ce₊, cw₋, cw₊, cs₋, cs₊, cn₋, cn₊ = similar(x, (n₁, n₂))
    @preallocate de₋, de₊, dw₋, dw₊, ds₋, ds₊, dn₋, dn₊ = similar(x, (n₁, n₂))
    @preallocate ae₋, ae₊, aw₋, aw₊, as₋, as₊, an₋, an₊ = similar(x, (n₁, n₂))
    @preallocate Fe₋, Fe₊, Fw₋, Fw₊, Fs₋, Fs₊, Fn₋, Fn₊ = similar(x, (n₁, n₂, nᵤ))
    @preallocate fe, fw, fs, fn = similar(x, (n₁, n₂, nᵤ))

    return @ntuple U Fx Fy dUhypx dUhypy Ue₋ Ue₊ Uw₋ Uw₊ Us₋ Us₊ Un₋ Un₊ ce₋ ce₊ cw₋ cw₊ cs₋ cs₊ cn₋ cn₊ de₋ de₊ dw₋ dw₊ ds₋ ds₊ dn₋ dn₊ ae₋ ae₊ aw₋ aw₊ as₋ as₊ an₋ an₊ Fe₋ Fe₊ Fw₋ Fw₊ Fs₋ Fs₊ Fn₋ Fn₊ fe fw fs fn
end

function build_cache_cap(T, n₁, n₂) #Initializing variables
    x = T()
    @preallocate h, hux, huy, ux, uy, vx, vy, ϕx, ϕy = similar(x, (n₁, n₂))
    @preallocate fxx, fxy, fyy, gv, fvx, fvy, convxx, convxy, convyx, convyy, gx, gy, Pid = similar(x, (n₁, n₂))
    return @ntuple h hux huy ux uy vx vy ϕx ϕy fxx fxy fyy gv fvx fvy gx gy Pid convxx convxy convyx convyy
end

build_cache(T, n₁, n₂) = (cap=build_cache_cap(T, n₁, n₂), hyp=build_cache_hyp(T, n₁, n₂))


end
