module Hyperbolic
export update_hyp!

using UnPack, FLoops
using FoldsCUDA, CUDA
using ..Helpers
using ..Grids
using ..Model: MODE, nᵤ

@inline minmod(x, y) = 0.5 * (sign(x) + sign(y)) * min(abs(x), abs(y)) #discritisation for second order

@bc U function update_bounds_x!(Uw₋, Uw₊, Ue₋, Ue₊, U, n₁, n₂, Δx, Δy, i, j, k; order=1) #discretisation scheme along x
    if order == 2
        ∇Ui = minmod((U[i, j, k] - U[i-1, j, k]) / Δx, (U[i+1, j, k] - U[i, j, k]) / Δx)
        ∇Ue = minmod((U[i+1, j, k] - U[i, j, k]) / Δx, (U[i+2, j, k] - U[i+1, j, k]) / Δx)
        ∇Uw = minmod((U[i-1, j, k] - U[i-2, j, k]) / Δx, (U[i, j, k] - U[i-1, j, k]) / Δx)
    else
        ∇Ui = 0.0
        ∇Ue = 0.0
        ∇Uw = 0.0
    end
    Ue₋[i, j, k] = U[i, j, k] + Δx / 2 * ∇Ui #(value of U at x=i)
    Ue₊[i, j, k] = U[i+1, j, k] - Δx / 2 * ∇Ue #(value of U at x=i+1)

    Uw₋[i, j, k] = U[i-1, j, k] + Δx / 2 * ∇Uw #(value of U at x=i-1)
    Uw₊[i, j, k] = U[i, j, k] - Δx / 2 * ∇Ui #(value of U at x=i)
    return
end

@bc U function update_bounds_y!(Us₋, Us₊, Un₋, Un₊, U, n₁, n₂, Δx, Δy, i, j, k; order=1) #discretisation scheme along y
    if order == 2
        ∇Ui = minmod((U[i, j, k] - U[i, j-1, k]) / Δy, (U[i, j+1, k] - U[i, j, k]) / Δy)
        ∇Un = minmod((U[i, j+1, k] - U[i, j, k]) / Δy, (U[i, j+2, k] - U[i, j+1, k]) / Δy)
        ∇Us = minmod((U[i, j-1, k] - U[i, j-2, k]) / Δy, (U[i, j, k] - U[i, j-1, k]) / Δy)
    else
        ∇Ui = 0.0
        ∇Un = 0.0
        ∇Us = 0.0
    end

    Un₋[i, j, k] = U[i, j, k] + Δy / 2 * ∇Ui #(value of U at y=j)
    Un₊[i, j, k] = U[i, j+1, k] - Δy / 2 * ∇Un #(value of U at y=j+1)

    Us₋[i, j, k] = U[i, j-1, k] + Δy / 2 * ∇Us #(value of U at y=j-1)
    Us₊[i, j, k] = U[i, j, k] - Δy / 2 * ∇Ui #(value of U at y=j)
    return
end

function compute_caF_x!(c, d, a, F, U, i, j)
    c[i, j] = U[i, j, 2] / U[i, j, 1] #ux
    a[i, j] = √(3/5) *(max(U[i, j, 9] , 0)) # √(3/5)*max(hϕx,0)
    for k in 1:nᵤ
        F[i, j, k] = c[i, j] * U[i, j, k] #ux*h, ux*(h ux), ux*(h uy), ux*(h vx).. ux*(h ψ1y)
    end
    F[i, j, 2] += U[i, j, 1] * U[i, j, 9]* U[i, j, 9]/5 # ux*(h ux) + h^3*ϕx/5
    F[i, j, 3] += U[i, j, 1] * U[i, j, 9]* U[i, j, 10]/5# ux*(h uy) + h^3*ϕx/5
    return
end

function compute_caF_oldx!(c, d, a, F, U, i, j)
    c[i, j] = U[i, j, 2] / U[i, j, 1] #ux
    a[i, j] = √(3U[i, j, 1]) * √(max(U[i, j, 6], 0)) # √(3h)*√max(hϕxx,0)
    for k in 1:nᵤ
        F[i, j, k] = c[i, j] * U[i, j, k] #ux*h, ux*(h ux), ux*(h uy), ux*(h vx).. ux*(h ψ1y)
    end
   F[i, j, 2] += U[i, j, 1]^2 * U[i, j, 6] # ux*(h ux) + h^3*ϕx
   F[i, j, 3] += U[i, j, 1]^2 * U[i, j, 7] # ux*(h uy) + h^3*ϕxy
    return
end


function compute_caF_y!(c, d, a, F, U, i, j)
    c[i, j] = U[i, j, 3] / U[i, j, 1] #uy
    a[i, j] = √(3/5) *(max(U[i, j, 10] , 0)) # (3/5)*√max(hϕy,0)
    for k in 1:nᵤ
        F[i, j, k] = c[i, j] * U[i, j, k] #uy*h, uy*(h ux), uy*(h uy), uy*(h vx).. uy*(h ψ1y)
    end
    F[i, j, 2] += U[i, j, 1] * U[i, j, 9]* U[i, j, 10]/5 # uy*(h ux) + h^3*ϕxϕy/5
    F[i, j, 3] += U[i, j, 1] * U[i, j, 10]* U[i, j, 10]/5 # uy*(h uy) + h^3*ϕx*ϕy/5
    return
end

function compute_caF_oldy!(c, d, a, F, U, i, j)
    c[i, j] = U[i, j, 3] / U[i, j, 1] #uy
    a[i, j] = √(3U[i, j, 1]) * √(max(U[i, j, 8], 0)) # √(3h)*√max(hϕyy,0)
    for k in 1:nᵤ
        F[i, j, k] = c[i, j] * U[i, j, k] #uy*h, uy*(h ux), uy*(h uy), uy*(h vx).. uy*(h ψ1y)
    end
    F[i, j, 2] += U[i, j, 1]^2 * U[i, j, 7] # uy*(h ux) + h^3*ϕxy
    F[i, j, 3] += U[i, j, 1]^2 * U[i, j, 8] # uy*(h uy) + h^3*ϕyy
    return
end

@inline ps(cₗ, cᵣ, aₗ, aᵣ) = max(abs(cₗ) + aₗ, abs(cᵣ) + aᵣ)#ps=max((|ux|+√(3h)*√max(hϕx,0))(i+1),(|ux|+√(3h)*√max(hϕx,0))(i+1))


@inline function compute_boundaries_flux!(f, U₊, U₋, c₊, c₋, a₊, a₋, F₊, F₋, i, j, k)
    f[i, j, k] = 0.5 * ((F₊[i, j, k] + F₋[i, j, k]) - ps(c₊[i, j], c₋[i, j], a₊[i, j], a₋[i, j]) * (U₊[i, j, k] - U₋[i, j, k]))  #Flux= 0.5*((F(i+1)-F(i+1))-(U(i+1)-U(i+1))*ps)
    return
end

@inline function compute_flux_balance!(F, f1, f2, δ, i, j, k) #Updating F, f(i)-f(i+1)/Δx
    F[i, j, k] = (f2[i, j, k] - f1[i, j, k]) / δ
    return
end

function update_hyp_x!(dUvec, U, p, t; gridinfo, cache_hyp, executor=ThreadedEx())
    @unpack Fx = cache_hyp
    @unpack Ue₋, Ue₊, Uw₋, Uw₊ = cache_hyp # storing some temporary storage?
    @unpack ce₋, ce₊, cw₋, cw₊ = cache_hyp
    @unpack de₋, de₊, dw₋, dw₊ = cache_hyp
    @unpack ae₋, ae₊, aw₋, aw₊ = cache_hyp
    @unpack Fe₋, Fe₊, Fw₋, Fw₊ = cache_hyp
    @unpack fe, fw = cache_hyp
    @unpack Δx, Δy, n₁, n₂ = gridinfo #obtaining mesh size and number of grids

    @floop executor for I in CartesianIndices((n₁, n₂, nᵤ)) 
        i, j, k = Tuple(I)
        update_bounds_x!(Uw₋, Uw₊, Ue₋, Ue₊, U, n₁, n₂, Δx, Δy, i, j, k) #Retrieve values for Uw₋=U(i-1), Uw₊=U(i), Ue₋=U(i), Ue₊=U(i+1)
    end

    @floop executor for I in CartesianIndices((n₁, n₂))
        i, j = Tuple(I)
        compute_caF_x!(cw₋, dw₋, aw₋, Fw₋, Uw₋, i, j)#To determine F(i-1)
        compute_caF_x!(cw₊, dw₊, aw₊, Fw₊, Uw₊, i, j)#To determine F(i)
        compute_caF_x!(ce₋, de₋, ae₋, Fe₋, Ue₋, i, j)#To determine F(i)
        compute_caF_x!(ce₊, de₊, ae₊, Fe₊, Ue₊, i, j)#To determine F(i+1)
        for k in 1:nᵤ
            compute_boundaries_flux!(fw, Uw₊, Uw₋, cw₊, cw₋, aw₊, aw₋, Fw₊, Fw₋, i, j, k)# compute flux, f between i,i-1 cell
            compute_boundaries_flux!(fe, Ue₊, Ue₋, ce₊, ce₋, ae₊, ae₋, Fe₊, Fe₋, i, j, k)# compute flux, f between i+1,i cell
            compute_flux_balance!(Fx, fe, fw, Δx, i, j, k)# Updating F in x direction
            vectorize_U!(dUvec, Fx, n₁, n₂, i, j, k)# Updating values of U
        end
    end
    return dUvec
end

function update_hyp_y!(dUvec, U, p, t; gridinfo, cache_hyp, executor=ThreadedEx())
    @unpack Fy = cache_hyp
    @unpack Un₋, Un₊, Us₋, Us₊ = cache_hyp # storing some temporary storage?
    @unpack cn₋, cn₊, cs₋, cs₊ = cache_hyp
    @unpack dn₋, dn₊, ds₋, ds₊ = cache_hyp
    @unpack an₋, an₊, as₋, as₊ = cache_hyp
    @unpack Fn₋, Fn₊, Fs₋, Fs₊ = cache_hyp
    @unpack fn, fs = cache_hyp
    @unpack Δx, Δy, n₁, n₂ = gridinfo #obtaining mesh size and number of grids

    @floop executor for I in CartesianIndices((n₁, n₂, nᵤ))
        i, j, k = Tuple(I)
        update_bounds_y!(Us₋, Us₊, Un₋, Un₊, U, n₁, n₂, Δx, Δy, i, j, k) #Retrieve values for Us₋=U(j-1), Us₊=U(j), Un₋=U(j), Un₊=U(j+1),
    end

    @floop executor for I in CartesianIndices((n₁, n₂))
        i, j = Tuple(I)
        compute_caF_y!(cs₋, ds₋, as₋, Fs₋, Us₋, i, j)#To determine F(j-1)
        compute_caF_y!(cs₊, ds₊, as₊, Fs₊, Us₊, i, j)#To determine F(j)
        compute_caF_y!(cn₋, dn₋, an₋, Fn₋, Un₋, i, j)#To determine F(j)
        compute_caF_y!(cn₊, dn₊, an₊, Fn₊, Un₊, i, j)#To determine F(j+1)
        for k in 1:nᵤ
            compute_boundaries_flux!(fs, Us₊, Us₋, cs₊, cs₋, as₊, as₋, Fs₊, Fs₋, i, j, k)# compute flux, f between j,j-1 cell
            compute_boundaries_flux!(fn, Un₊, Un₋, cn₊, cn₋, an₊, an₋, Fn₊, Fn₋, i, j, k)# compute flux, f between j+1,j cell
            compute_flux_balance!(Fy, fn, fs, Δy, i, j, k)# Updating F in x direction
            vectorize_U!(dUvec, Fy, n₁, n₂, i, j, k) # Updating values of U
        end
    end
    return dUvec
end

function update_hyp!(dUvec, Uvec, p, t; gridinfo, caches, executor=:auto)
    typed_caches = caches[typeof(Uvec)]
    if executor == :auto
        executor = Uvec isa CuArray ? CUDAEx() : ThreadedEx()
    end
    cache_hyp = typed_caches.hyp
    @unpack dUhypx, dUhypy, U = cache_hyp
    @unpack Δx, Δy, n₁, n₂ = gridinfo

    matricize_Uvec!(U, Uvec, n₁, n₂; executor)#MAking Uvec into U matrix(i,j,k)

    update_hyp_x!(dUhypx, U, p, t; gridinfo, cache_hyp, executor) #Finding updated values of U in x
    update_hyp_y!(dUhypy, U, p, t; gridinfo, cache_hyp, executor) #Finding updated values of U in y
    @. dUvec = dUhypx + dUhypy #summing updated values of U in x and y.
    #@show Uvec
end 

end
