module Ops
export @dx, @dy, @dxx, @dxy, @dyy, @∇, @∇t, @∇2, @∇∇, @div, @div∇, @divh∇, @divh∇t, @∇div, @∇hdiv, ⊗

using ..Helpers
using StaticArrays

@inline @bc m dx(m, Δx, Δy, n₁, n₂, i, j) = (m[i+1, j] - m[i-1, j]) / 2Δx
@inline @bc m dy(m, Δx, Δy, n₁, n₂, i, j) = (m[i, j+1] - m[i, j-1]) / 2Δy
macro dx(m)
    esc(:(Ops.dx($m, Δx, Δy, n₁, n₂, i, j)))
end
macro dy(m)
    esc(:(Ops.dy($m, Δx, Δy, n₁, n₂, i, j)))
end

@inline @bc m dxx(m, Δx, Δy, n₁, n₂, i, j) = (m[i+1, j]-2*m[i, j] + m[i-1, j]) / (Δx)^2
@inline @bc m dxy(m, Δx, Δy, n₁, n₂, i, j) = (m[i+1, j+1]-m[i+1, j-1]-m[i-1, j+1] + m[i-1, j-1]) / 4*(Δx)*(Δy)
@inline @bc m dyy(m, Δx, Δy, n₁, n₂, i, j) = (m[i, j+1] -2*m[i, j]+ m[i, j-1]) / (Δy)^2
macro dxx(m)
    esc(:(Ops.dxx($m, Δx, Δy, n₁, n₂, i, j)))
end
macro dxy(m)
    esc(:(Ops.dxy($m, Δx, Δy, n₁, n₂, i, j)))
end
macro dyy(m)
    esc(:(Ops.dyy($m, Δx, Δy, n₁, n₂, i, j)))
end


@inline ∇(m, Δx, Δy, n₁, n₂, i, j) = @SVector [(@dx(m)), (@dy(m))]
@inline ∇(mx, my, Δx, Δy, n₁, n₂, i, j) = @SMatrix [(@dx(mx)) (@dx(my))
    (@dy(mx)) (@dy(my))]
macro ∇(m)
    esc(:(Ops.∇($m, Δx, Δy, n₁, n₂, i, j)))
end
macro ∇(mx, my)
    esc(:(Ops.∇($mx, $my, Δx, Δy, n₁, n₂, i, j)))
end


@inline ∇t(mx, my, Δx, Δy, n₁, n₂, i, j) = @SMatrix [(@dx(mx)) (@dy(mx))
    (@dx(my)) (@dy(my))]
macro ∇t(mx, my)
    esc(:(Ops.∇t($mx, $my, Δx, Δy, n₁, n₂, i, j)))
end


@inline ∇2(m, Δx, Δy, n₁, n₂, i, j) = ((@dxx(m)) + (@dyy(m)))
@inline ∇2(mx, my, Δx, Δy, n₁, n₂, i, j) = @SVector [((@dxx(mx))+(@dyy(mx))), (@dxx(mx))+(@dyy(my))]
macro ∇2(m)
    esc(:(Ops.∇2($m, Δx, Δy, n₁, n₂, i, j)))
end
macro ∇2(mx, my)
    esc(:(Ops.∇2($mx, $my, Δx, Δy, n₁, n₂, i, j)))
end

@inline ∇∇(m, Δx, Δy, n₁, n₂, i, j) = @SMatrix [(@dxx(m)) (@dxy(m))
(@dxy(m)) (@dyy(m))]
macro ∇∇(m)
    esc(:(Ops.∇∇($m, Δx, Δy, n₁, n₂, i, j)))
end



@inline @bc (mx, mx) div(mx, my, Δx, Δy, n₁, n₂, i, j) = (@dx(mx)) + (@dy(my))
@inline @bc (mxx, mxy, myy) div(mxx, mxy, myy, Δx, Δy, n₁, n₂, i, j) = @SVector[(@dx(mxx)) + (@dy(mxy)), (@dx(mxy)) + (@dy(myy))]
macro div(mx, my)
    esc(:(Ops.div($mx, $my, Δx, Δy, n₁, n₂, i, j)))
end
macro div(mxx, mxy, myy)
    esc(:(Ops.div($mxx, $mxy, $myy, Δx, Δy, n₁, n₂, i, j)))
end

@inline @bc (h, mx, my) function divh∇(h, mx, my, Δx, Δy, n₁, n₂, i, j)
    ∂x⁰_h∂x_mx = (1 / 2 * (h[i+1, j] + h[i, j]) * (mx[i+1, j] - mx[i, j]) - 1 / 2 * (h[i-1, j] + h[i, j]) * (mx[i, j] - mx[i-1, j])) / Δx^2
    ∂y⁰⁰_h∂x⁰⁰_my = (h[i, j+1] * (my[i+1, j+1] - my[i-1, j+1]) - h[i, j-1] * (my[i+1, j-1] - my[i-1, j-1])) / (2Δx * 2Δy)
    ∂x⁰⁰_h∂y⁰⁰_mx = (h[i+1, j] * (mx[i+1, j+1] - mx[i+1, j-1]) - h[i-1, j] * (mx[i-1, j+1] - mx[i-1, j-1])) / (2Δx * 2Δy)
    ∂y⁰_h∂y_my = (1 / 2 * (h[i, j+1] + h[i, j]) * (my[i, j+1] - my[i, j]) - 1 / 2 * (h[i, j-1] + h[i, j]) * (my[i, j] - my[i, j-1])) / Δy^2
    return @SVector [∂x⁰_h∂x_mx + ∂y⁰⁰_h∂x⁰⁰_my, ∂x⁰⁰_h∂y⁰⁰_mx + ∂y⁰_h∂y_my]
end

macro divh∇(mx, my)
    esc(:(Ops.divh∇(h, $mx, $my, Δx, Δy, n₁, n₂, i, j)))
end

@inline @bc (h, mx, my) function divh∇t(h, mx, my, Δx, Δy, n₁, n₂, i, j)
    ∂x⁰_h∂x_mx = (1 / 2 * (h[i+1, j] + h[i, j]) * (mx[i+1, j] - mx[i, j]) - 1 / 2 * (h[i-1, j] + h[i, j]) * (mx[i, j] - mx[i-1, j])) / Δx^2
    ∂y⁰⁰_h∂y⁰⁰_mx = (1 / 2 * (h[i, j+1] + h[i, j]) * (mx[i, j+1] - mx[i, j]) - 1 / 2 * (h[i, j-1] + h[i, j]) * (mx[i, j] - mx[i, j-1])) / Δy^2
    ∂x⁰⁰_h∂x⁰⁰_my = (1 / 2 * (h[i+1, j] + h[i, j]) * (my[i+1, j] - my[i, j]) - 1 / 2 * (h[i-1, j] + h[i, j]) * (my[i, j] - my[i-1, j])) / Δx^2
    ∂y⁰_h∂y_my = (1 / 2 * (h[i, j+1] + h[i, j]) * (my[i, j+1] - my[i, j]) - 1 / 2 * (h[i, j-1] + h[i, j]) * (my[i, j] - my[i, j-1])) / Δy^2
    return @SVector [∂x⁰_h∂x_mx + ∂y⁰⁰_h∂y⁰⁰_mx, ∂x⁰⁰_h∂x⁰⁰_my + ∂y⁰_h∂y_my]
end

macro divh∇t(mx, my)
    esc(:(Ops.divh∇t(h, $mx, $my, Δx, Δy, n₁, n₂, i, j)))
end

@inline ∇div(mx, my, Δx, Δy, n₁, n₂, i, j) = @SVector [(@dxx(mx))+(@dxy(my)), (@dxy(mx))+(@dyy(my))]
macro ∇div(mx, my)
    esc(:(Ops.∇div($mx, $my, Δx, Δy, n₁, n₂, i, j)))
end


@inline @bc (h, mx, my) function ∇hdiv(h, mx, my, Δx, Δy, n₁, n₂, i, j)
    ∂x⁰_h∂x_mx = (1 / 2 * (h[i+1, j] + h[i, j]) * (mx[i+1, j] - mx[i, j]) - 1 / 2 * (h[i-1, j] + h[i, j]) * (mx[i, j] - mx[i-1, j])) / Δx^2
    ∂y⁰⁰_h∂y⁰⁰_mx = (1 / 2 * (h[i, j+1] + h[i, j]) * (mx[i, j+1] - mx[i, j]) - 1 / 2 * (h[i, j-1] + h[i, j]) * (mx[i, j] - mx[i, j-1])) / Δy^2
    ∂x⁰⁰_h∂x⁰⁰_my = (1 / 2 * (h[i+1, j] + h[i, j]) * (my[i+1, j] - my[i, j]) - 1 / 2 * (h[i-1, j] + h[i, j]) * (my[i, j] - my[i-1, j])) / Δx^2
    ∂y⁰_h∂y_my = (1 / 2 * (h[i, j+1] + h[i, j]) * (my[i, j+1] - my[i, j]) - 1 / 2 * (h[i, j-1] + h[i, j]) * (my[i, j] - my[i, j-1])) / Δy^2
    return @SVector [∂x⁰_h∂x_mx + ∂y⁰⁰_h∂y⁰⁰_mx, ∂x⁰⁰_h∂x⁰⁰_my + ∂y⁰_h∂y_my]
end

macro ∇hdiv(mx, my)
    esc(:(Ops.∇hdiv(h, $mx, $my, Δx, Δy, n₁, n₂, i, j)))
end

⊗(a::AbstractVector, b::AbstractVector) = a * b'
end