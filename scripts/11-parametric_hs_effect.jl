# %%
using DropletSpreadingSim2

using DifferentialEquations, Sundials, Logging, DrWatson
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger(stderr))

# %%
function do_simulate(p; filename)
    @unpack h₀, ls, σ, ρ, μ, τ, θτ, L, hₛ, hₛ_ratio, θₛ, dθₛ, hₛ, aspect_ratio, tmax, α, f,
    mass, ndrops, hdrop_std, two_dim, reproject = p
    θₐ = deg2rad(θₛ + dθₛ)
    θᵣ = deg2rad(θₛ - dθₛ)

    # %%
    experiment = DropletSpreadingExperiment(; tmax, h₀, ls, σ, ρ, μ, τ, θτ, L, hₛ_ratio, hₛ, θₐ, θᵣ, α, f,
        aspect_ratio, mass, ndrops, hdrop_std, two_dim)

    # %%
    prob = ODEProblem(experiment, (0.0, p[:tmax]))
    # cfl_limiter = build_cfl_limiter(experiment; safety_factor=p[:cfl_safety_factor])
    callbacks = Any[]
    if ~isnothing(filename)
        save_cb = build_save_callback(
            filename, prob, experiment;
            saveat=get(p, :save_timestep, nothing), attrib=p
        )
        push!(callbacks, save_cb)
    end
    if reproject
        reproject_cb = build_reprojection_callback(experiment; thresh=1e-3)
        push!(callbacks, reproject_cb)
    end

    @info "launch sim" p
    @time sol = solve(
        prob,
        #AutoTsit5(Rosenbrock23());
        #Tsit5();
        SSPRK432();
        callback=CallbackSet(callbacks...),
        progress=true,
        progress_steps=1,
        save_everystep=false,
        saveat=get(p, :keep_timestep, []),
        dt=1e-3,
    )

    return sol, experiment
end
#SSPRK432();
# %%
parameters = Dict(
    :tmax => 1000,
    :hₛ_ratio => [20,40,60,80],
    #:hₛ => 0.334112,
    #:hₛ => 1.27726,
    :ndrops => 1,
    :hₛ => 1,
    :hdrop_std => 0.2,
    :h₀ => 0.001,
    :ls => 0.001,
    #:ls => [2e-2,1e-2,5e-3,2e-3,1e-3,5e-4,2e-4],
    #:μ => 0.00313015,
    #:μ => 0.00669445,
    :μ => 0.0014,
    #:σ => 0.075,
    #:σ => 0.0484,
    #:σ => 0.067,
    :σ => 0.074,
    :θₛ => -90,
    #:α => 6.4,
    :α => 90,
    :f => 0,
    :dθₛ => 0,
    :save_timestep => 10,
    :θτ => 0.0,
    #:mass => 12.95,
    :mass => 0.0182089,
    #:mass => [0.0182089,0.0387211,0.055809],
    :aspect_ratio => 5/2,
    :ρ => 998,
    #:ρ => 1098.3,
    #:ρ => 1067.7,
    :τ => 0.0,
    :L => 4,
    :two_dim => true,
    :cfl_safety_factor => 0.9,
    :reproject => true,
)
parameters = dict_list(parameters)

# %%
for p ∈ parameters
    #params = (hₛ="0.05", hₛ_ratio="4")
    out_dir = "data/fallingfilms/3-D/Florineexperiments_29_disjoiningpressure"
    filename = savename(p, "nc", accesses=[:hₛ_ratio])
    #filename = savename(params, "nc")
    if ~isnothing(filename) && isfile(joinpath(out_dir, "$(basename(filename)).done"))
        @info "skipping" filename
        continue
    end
    # remove filename if it exists
    if ~isnothing(filename) && isfile(filename)
        rm(filename)
    end

    sol, experiment = do_simulate(p; filename=joinpath(out_dir, filename))
    if ~isnothing(filename)
        touch(joinpath(out_dir, "$(basename(filename)).done"))
    end
end
