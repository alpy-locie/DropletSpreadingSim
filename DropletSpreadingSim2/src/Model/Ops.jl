module Ops
export @dx, @dy, @dxx, @dxy, @dyy, @∇, @∇t, @∇2, @∇∇, @div, @div∇, @divh∇, @divh∇t, @∇div, @∇hdiv, ⊗

using ..Helpers
using StaticArrays

@inline @bc m dx(m, Δx, Δy, n₁, n₂, i, j) = (m[i+1, j] - m[i-1, j]) / 2Δx
@inline @bc m dy(m, Δx, Δy, n₁, n₂, i, j) = (m[i, j+1] - m[i, j-1]) / 2Δy
macro dx(m)#d/dx
    esc(:(Ops.dx($m, Δx, Δy, n₁, n₂, i, j)))
end
macro dy(m)#d/dy
    esc(:(Ops.dy($m, Δx, Δy, n₁, n₂, i, j)))
end

@inline @bc m dxx(m, Δx, Δy, n₁, n₂, i, j) = (m[i+1, j]-2*m[i, j] + m[i-1, j]) / (Δx)^2
@inline @bc m dxy(m, Δx, Δy, n₁, n₂, i, j) = (m[i+1, j+1]-m[i+1, j-1]-m[i-1, j+1] + m[i-1, j-1]) / (4*(Δx)*(Δy))
@inline @bc m dyy(m, Δx, Δy, n₁, n₂, i, j) = (m[i, j+1] -2*m[i, j]+ m[i, j-1]) / (Δy)^2
macro dxx(m)#d^2/dx^2
    esc(:(Ops.dxx($m, Δx, Δy, n₁, n₂, i, j)))
end
macro dxy(m) #d^2/dxdy
    esc(:(Ops.dxy($m, Δx, Δy, n₁, n₂, i, j)))
end
macro dyy(m) #d^2/dy^2
    esc(:(Ops.dyy($m, Δx, Δy, n₁, n₂, i, j)))
end


@inline ∇(m, Δx, Δy, n₁, n₂, i, j) = @SVector [(@dx(m)), (@dy(m))] #[d/dx ex, d/dy ey]
@inline ∇t(mx, my, Δx, Δy, n₁, n₂, i, j) = @SMatrix [(@dx(mx)) (@dx(my)); #[d/dx mx ex ex, d/dx my ex ey; d/dy mx ey ex, d/dx my ey ey]
    (@dy(mx)) (@dy(my))]
macro ∇(m)
    esc(:(Ops.∇($m, Δx, Δy, n₁, n₂, i, j)))
end
macro ∇t(mx, my)
    esc(:(Ops.∇t($mx, $my, Δx, Δy, n₁, n₂, i, j)))
end


@inline ∇(mx, my, Δx, Δy, n₁, n₂, i, j) = @SMatrix [(@dx(mx)) (@dy(mx));  #[d/dx mx ex ex, d/dy mx ey ex; d/dx my ex ey, d/dx my ey ey]
    (@dx(my)) (@dy(my))]
macro ∇(mx, my)
    esc(:(Ops.∇($mx, $my, Δx, Δy, n₁, n₂, i, j)))
end


@inline ∇2(m, Δx, Δy, n₁, n₂, i, j) = ((@dxx(m)) + (@dyy(m))) # d^2/dx^2 m  + d^2/dy^2 m
@inline ∇2(mx, my, Δx, Δy, n₁, n₂, i, j) = @SVector [((@dxx(mx))+(@dyy(mx))), (@dxx(mx))+(@dyy(my))]#[(d^2/dx^2 mx  + d^2/dy^2 mx) ex,  (d^2/dx^2 my  + d^2/dy^2 my) ey]
macro ∇2(m)
    esc(:(Ops.∇2($m, Δx, Δy, n₁, n₂, i, j)))
end
macro ∇2(mx, my)
    esc(:(Ops.∇2($mx, $my, Δx, Δy, n₁, n₂, i, j)))
end

@inline ∇∇(m, Δx, Δy, n₁, n₂, i, j) = @SMatrix [(@dxx(m)) (@dxy(m));
(@dxy(m)) (@dyy(m))]  #[((d^2/dx^2)m ex ex, (d^2/dxdy)m ex ey,  ((d^2/dydx)m  ey ex+ (d^2/dy^2 m) ey ey]
macro ∇∇(m)
    esc(:(Ops.∇∇($m, Δx, Δy, n₁, n₂, i, j)))
end



@inline @bc (mx, mx) div(mx, my, Δx, Δy, n₁, n₂, i, j) = (@dx(mx)) + (@dy(my)) #[∇. [mx,my]]
@inline @bc (mxx, mxy, myy) div(mxx, mxy, myy, Δx, Δy, n₁, n₂, i, j) = @SVector[(@dx(mxx)) + (@dy(mxy)), (@dx(mxy)) + (@dy(myy))]
macro div(mx, my)
    esc(:(Ops.div($mx, $my, Δx, Δy, n₁, n₂, i, j)))
end
macro div(mxx, mxy, myy)
    esc(:(Ops.div($mxx, $mxy, $myy, Δx, Δy, n₁, n₂, i, j)))
end

@inline @bc (h, mx, my) function divh∇t(h, mx, my, Δx, Δy, n₁, n₂, i, j) #∇. h Tranposse{∇[mx,my]}
    ∂x⁰_h∂x_mx = (1 / 2 * (h[i+1, j] + h[i, j]) * (mx[i+1, j] - mx[i, j]) - 1 / 2 * (h[i-1, j] + h[i, j]) * (mx[i, j] - mx[i-1, j])) / Δx^2
    ∂y⁰⁰_h∂x⁰⁰_my = (h[i, j+1] * (my[i+1, j+1] - my[i-1, j+1]) - h[i, j-1] * (my[i+1, j-1] - my[i-1, j-1])) / (2Δx * 2Δy)
    ∂x⁰⁰_h∂y⁰⁰_mx = (h[i+1, j] * (mx[i+1, j+1] - mx[i+1, j-1]) - h[i-1, j] * (mx[i-1, j+1] - mx[i-1, j-1])) / (2Δx * 2Δy)
    ∂y⁰_h∂y_my = (1 / 2 * (h[i, j+1] + h[i, j]) * (my[i, j+1] - my[i, j]) - 1 / 2 * (h[i, j-1] + h[i, j]) * (my[i, j] - my[i, j-1])) / Δy^2
    return @SVector [∂x⁰_h∂x_mx + ∂y⁰⁰_h∂x⁰⁰_my, ∂x⁰⁰_h∂y⁰⁰_mx + ∂y⁰_h∂y_my]
end

macro divh∇t(mx, my)
    esc(:(Ops.divh∇t(h, $mx, $my, Δx, Δy, n₁, n₂, i, j)))
end

@inline @bc (h, mx, my) function divh∇(h, mx, my, Δx, Δy, n₁, n₂, i, j)#∇. h [∇[mx,my]]
    ∂x⁰_h∂x_mx = (1 / 2 * (h[i+1, j] + h[i, j]) * (mx[i+1, j] - mx[i, j]) - 1 / 2 * (h[i-1, j] + h[i, j]) * (mx[i, j] - mx[i-1, j])) / Δx^2
    ∂y⁰⁰_h∂y⁰⁰_mx = (1 / 2 * (h[i, j+1] + h[i, j]) * (mx[i, j+1] - mx[i, j]) - 1 / 2 * (h[i, j-1] + h[i, j]) * (mx[i, j] - mx[i, j-1])) / Δy^2
    ∂x⁰⁰_h∂x⁰⁰_my = (1 / 2 * (h[i+1, j] + h[i, j]) * (my[i+1, j] - my[i, j]) - 1 / 2 * (h[i-1, j] + h[i, j]) * (my[i, j] - my[i-1, j])) / Δx^2
    ∂y⁰_h∂y_my = (1 / 2 * (h[i, j+1] + h[i, j]) * (my[i, j+1] - my[i, j]) - 1 / 2 * (h[i, j-1] + h[i, j]) * (my[i, j] - my[i, j-1])) / Δy^2
    return @SVector [∂x⁰_h∂x_mx + ∂y⁰⁰_h∂y⁰⁰_mx, ∂x⁰⁰_h∂x⁰⁰_my + ∂y⁰_h∂y_my]
end

macro divh∇(mx, my)
    esc(:(Ops.divh∇(h, $mx, $my, Δx, Δy, n₁, n₂, i, j)))
end

@inline ∇div(mx, my, Δx, Δy, n₁, n₂, i, j) = @SVector [(@dxx(mx))+(@dxy(my)), (@dxy(mx))+(@dyy(my))] #∇(∇.[mx,my])
macro ∇div(mx, my)
    esc(:(Ops.∇div($mx, $my, Δx, Δy, n₁, n₂, i, j)))
end


@inline @bc (h, mx, my) function ∇hdiv(h, mx, my, Δx, Δy, n₁, n₂, i, j)#∇(h∇.[mx,my])
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