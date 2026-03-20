module NonConservative
export update_cap!
using StaticArrays, UnPack, FLoops
using FoldsCUDA, CUDA
using ..Ops
using ..Grids
using ..Model: MODE, nᵤ

function unpack_hu!(hux, huy, Uvec, n₁, n₂, i, j)#function for reducing the dimension of matirx from Uvec
    hux[i, j] = Uvec[gridded_to_flat(2, i, j; nᵤ, n₁, n₂)]
    huy[i, j] = Uvec[gridded_to_flat(3, i, j; nᵤ, n₁, n₂)]
    return
end


function unpack_hu!(hux, huy, Uvec, n₁, n₂; executor=ThreadedEx())
    @floop executor for I in CartesianIndices((n₁, n₂))
        unpack_hu!(hux, huy, Uvec, n₁, n₂, Tuple(I)...) #finding hux and huy
    end
    return hux, huy
end

function compute_skew_cap_coeffs!(
    fxx, fxy, fyy, gx, gy, gv, fvx, fvy, Pid,
    h, vx, vy, κ, θₐ, θᵣ, hₛ,
    hux, huy, ux, uy, ϕx, ϕy,
    Δx, Δy, n₁, n₂, i, j
)
    @static if MODE == :full #definition of f1 tensors(f=fxx,fxy,fyy) and f2 vector: g=(gx,gy) 
        fxx[i, j] = √κ * √h[i, j] * 1 / √(1 + h[i, j] / 4κ * (vx[i, j]^2 + vy[i, j]^2)) * (1 - 1 / (1 + h[i, j] / 2κ * (vx[i, j]^2 + vy[i, j]^2)) * h[i, j] / 4κ * (vx[i, j]^2))
        fxy[i, j] = √κ * √h[i, j] * 1 / √(1 + h[i, j] * 1 / 4κ * (vx[i, j]^2 + vy[i, j]^2)) * (-1 / (1 + h[i, j] / 2κ * (vx[i, j]^2 + vy[i, j]^2)) * h[i, j] / 4κ * (vx[i, j] * vy[i, j]))
        fyy[i, j] = √κ * √h[i, j] * 1 / √(1 + h[i, j] * 1 / 4κ * (vx[i, j]^2 + vy[i, j]^2)) * (1 - 1 / (1 + h[i, j] / 2κ * (vx[i, j]^2 + vy[i, j]^2)) * h[i, j] / 4κ * (vy[i, j]^2))
        gx[i, j] = h[i, j] * vx[i, j] / 2 * (1 + h[i, j] / 2κ * (vx[i, j]^2 + vy[i, j]^2))^(-1)
        gy[i, j] = h[i, j] * vy[i, j] / 2 * (1 + h[i, j] / 2κ * (vx[i, j]^2 + vy[i, j]^2))^(-1)
    elseif MODE == :simple
        fxx[i, j] = fyy[i, j] = √κ * √h[i, j]
        fxy[i, j] = 0.0
        gx[i, j] = h[i, j] * vx[i, j] / 2
        gy[i, j] = h[i, j] * vy[i, j] / 2
    else
        fxx[i, j] = fyy[i, j] = 0.0
        fxy[i, j] = 0.0
        gx[i, j] = 0.0
        gy[i, j] = 0.0
    end
    
    v = @SVector [vx[i, j], vy[i, j]]
    g = @SVector [gx[i, j], gy[i, j]]
    f = @SMatrix(
        [
            fxx[i, j] fxy[i, j]
            fxy[i, j] fyy[i, j]
        ]
    )
    u = @SVector [ux[i, j], uy[i, j]]
    ϕ1 = @SVector [ϕx[i, j], ϕy[i, j]]
   
    gv[i, j] = g' * v #f2.W scalar
    fv = f * v #f1.W vector

    dej = (hₛ / h[i, j])^4 - (hₛ / h[i, j])^3# for n=4,m=3
    ε = 1.e-3
    θₛ = 0.5 * (θₐ + θᵣ) + 0.5 * (θᵣ - θₐ) * tanh((@div(hux, huy)) / ε) #Calculation of angle

    fvx[i, j] = fv[1]
    fvy[i, j] = fv[2]
    Pid[i, j] = (6 / hₛ) * κ * (1 - cos(θₛ)) * dej# for n=4,m=3
    return
end

function compute_skew_cap_coeffs!(
    fxx, fxy, fyy, gx, gy, gv, fvx, fvy, Pid,
    h, vx, vy, κ, θₐ, θᵣ, hₛ,
    hux, huy, ux, uy, ϕx, ϕy,
    Δx, Δy, n₁, n₂; executor=ThreadedEx()
)
    @floop executor for I in CartesianIndices(h)# Evaluates PId, f1.W and f2.W for each grids
        i, j = Tuple(I)
        compute_skew_cap_coeffs!(
            fxx, fxy, fyy, gx, gy, gv, fvx, fvy, Pid,
            h, vx, vy, κ, θₐ, θᵣ, hₛ,
            hux, huy, ux, uy, ϕx, ϕy,
            Δx, Δy, n₁, n₂, i, j
        )
    end
end



function skew_cap_kernel!(
    dU, h, ux, uy, vx, vy, ϕx, ϕy,
    gx, gy, fxx, fxy, fyy, gv, fvx, fvy, Pid, ls, α,
    Re, β,
    Δx, Δy, n₁, n₂, i, j
)


    g = @SVector [gx[i, j], gy[i, j]]# Vector for f2
    f = @SMatrix [fxx[i, j] fxy[i, j]
        fxy[i, j] fyy[i, j]]         # Matrix for f1

    ϕ1 = @SVector [ϕx[i, j], ϕy[i, j]]# Vector for ψ   
    u = @SVector [ux[i, j], uy[i, j]]# Vector for u
    v = @SVector [vx[i, j], vy[i, j]]# Vector for W
    dh= @SVector [(@dx(h)), (@dy(h))]# gradient of h
    gvect=@SVector [h[i,j]*(1-cot(α*pi/180)*(@dx(h))), 0.0]
    #gvect=@SVector [h[i,j], 0.0]

  convdiv= @SVector [ϕx[i, j]*(@dx(ux))+ϕy[i, j]*(@dy(ux)), ϕx[i, j]*(@dx(uy))+ϕy[i, j]*(@dy(uy))] 

   dhuk = (-(@∇(gv)) + (@divh∇t(fvx, fvy))  +gvect/Re)-(u-h[i,j]*ϕ1)/(ls*Re)# momentum balance # divh∇ is now divh∇t
   dhu2=(@divh∇(ux,uy))+(@divh∇t(ux,uy)) + 2 * ((@div(ux,uy))*(@∇(h))+ h[i,j]*(@∇div(ux,uy))) 
   + (h[i,j]*(@∇(h))*(@div(ϕx, ϕy))+((h[i,j]^2)/2)*(@∇div(ϕx, ϕy)))
   - ((1/2)*((@∇(h))*(@∇(h))'+ h[i,j]*(@∇∇(h)))*ϕ1) 
   - ((ϕ1/2)*(((@dx(h))*(@dx(h))) + ((@dy(h))*(@dy(h))) + h[i,j]*(@∇2(h))))
   - ((1/2)*h[i,j]*(@∇(ϕx, ϕy))*(@∇(h)))- ((1/2)*h[i,j]*(@div(ϕx, ϕy))*(@∇(h)))
   
    dhun = -(@∇(gv)) + (@divh∇t(fvx, fvy)) + h[i, j] * (@∇(Pid)) +gvect/Re - (1 / Re) * ( (3*u) / (h[i, j]+3*ls))  #momentum balance 
    dhv = -g * (@div(ux, uy)) - f * (@divh∇t(ux, uy))
    dhϕ1 = (-h[i, j]*convdiv+ h[i, j]*(@div(ux,uy))*ϕ1 + (1. /7.)*((h[i, j])^2) * (@div(ϕx, ϕy)) * ϕ1 + (2. /7.)*((h[i, j])^2)*(@∇(ϕx, ϕy))*ϕ1+ (4. /7.)*(h[i, j])*((ϕ1'*@∇(h)))*ϕ1)+(5/(ls*Re*h[i,j]))*(u-(3*ls+h[i,j])*ϕ1)
    dhϕ11= - (3/8) * ((@∇∇(h))*ϕ1) + (7/8)*(@∇2(h)) * ϕ1 
    - (1/8)*((@∇(ϕx, ϕy))' * (@∇(h)) ) + (17/8)*(@div(ϕx, ϕy))*(@∇(h)) 
    - (1/(2*h[i,j])) * (@∇(h))' * (@∇(h)) * ϕ1 - (1/(2*h[i,j])) *  (ϕ1' * (@∇(h))) * (@∇(h))
    + (@divh∇(ϕx, ϕy))+ (@divh∇t(ϕx, ϕy))+(5/h[i,j])*(@div(ux,uy))*(@∇(h))+(5 /2)*(@∇2(ux,uy)) 
    +(5/(2*h[i,j]))*((@∇(ux,uy))*(@∇(h))+((@∇(ux,uy))'*(@∇(h))))# Equation for ψ1 

    # dU represents the non-conservative part of the equations
    dU[gridded_to_flat(1, i, j; nᵤ, n₁, n₂)] = 0.0 
    dU[gridded_to_flat(2, i, j; nᵤ, n₁, n₂)] = dhuk[1]+dhu2[1]/Re
    dU[gridded_to_flat(3, i, j; nᵤ, n₁, n₂)] = dhuk[2]+dhu2[2]/Re
    dU[gridded_to_flat(4, i, j; nᵤ, n₁, n₂)] = dhv[1]
    dU[gridded_to_flat(5, i, j; nᵤ, n₁, n₂)] = dhv[2]
    dU[gridded_to_flat(6, i, j; nᵤ, n₁, n₂)] = dhϕ1[1]+dhϕ11[1]/Re
    dU[gridded_to_flat(7, i, j; nᵤ, n₁, n₂)] = dhϕ1[2]+dhϕ11[2]/Re
    return
end


function skew_cap_kernel!(
    dU, h, ux, uy, vx, vy, ϕx, ϕy,
    gx, gy, fxx, fxy, fyy, gv, fvx, fvy, Pid, ls, α,
    Re, β,
    Δx, Δy, n₁, n₂; executor=ThreadedEx(),
)
    @floop executor for I in CartesianIndices(h)# Evaluates the non-conservative terms for each grids
        i, j = Tuple(I)
        skew_cap_kernel!(
            dU, h, ux, uy, vx, vy, ϕx, ϕy,
            gx, gy, fxx, fxy, fyy, gv, fvx, fvy, Pid, ls, α,
            Re, β,
            Δx, Δy, n₁, n₂, i, j
        )
    end
end

function update_cap!(dUvec, Uvec, p, t; gridinfo, caches, executor=:auto)
    typed_caches = caches[typeof(Uvec)]
    cache_cap = typed_caches.cap
    @unpack h, hux, huy, ux, uy, vx, vy, ϕx, ϕy = cache_cap #storing temporary variables?
    @unpack fxx, fxy, fyy, gv, fvx, fvy, gx, gy, Pid = cache_cap
    @unpack Δx, Δy, n₁, n₂ = gridinfo # imports the grid size and its number
    @unpack κ, Re, β, θₐ, θᵣ, hₛ, ls, α = p # imports the inputs of the problem

    if executor == :auto
        executor = Uvec isa CuArray ? CUDAEx() : ThreadedEx()
    end

    unpack_Uvec!(h, ux, uy, vx, vy, ϕx, ϕy, Uvec, n₁, n₂; executor)# Removing h from each variable
    unpack_hu!(hux, huy, Uvec, n₁, n₂; executor)# For obtaining the values of hux and huy

    compute_skew_cap_coeffs!(# Evaluates PId, f1.W and f2.W for each grids
        fxx, fxy, fyy, gx, gy, gv, fvx, fvy, Pid, h,
        vx, vy, κ, θₐ, θᵣ, hₛ, hux, huy, ux, uy, ϕx, ϕy, 
        Δx, Δy, n₁, n₂;
        executor
    )

    skew_cap_kernel!(# Evaluates the non-conservative terms
        dUvec, h, ux, uy, vx, vy, ϕx, ϕy,
        gx, gy, fxx, fxy, fyy, gv, fvx, fvy, Pid, ls, α,
        Re, β,
        Δx, Δy, n₁, n₂;
        executor
    )

    return dUvec #returns the non-conservative terms
end
end
