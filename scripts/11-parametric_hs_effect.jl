# %%
using DropletSpreadingSim2

using DifferentialEquations, Sundials, Logging, DrWatson
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger(stderr))

# %%
function do_simulate(p; filename)
    @unpack h₀, ls, σ, ρ, μ, τ, θτ, L, hₛ, hₛ_ratio, θₛ, dθₛ, hₛ, aspect_ratio, tmax,
    mass, ndrops, hdrop_std, two_dim, reproject = p
    θₐ = deg2rad(θₛ + dθₛ)
    θᵣ = deg2rad(θₛ - dθₛ)

    # %%
    experiment = DropletSpreadingExperiment(; h₀, ls, σ, ρ, μ, τ, θτ, L, hₛ_ratio, hₛ, θₐ, θᵣ,
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
    :tmax => 1200,
    :hₛ_ratio => 1.0,
    :hₛ => [2e-1,17e-2,15e-2,12e-2,1e-1,8e-2,7e-2,5e-2,3e-2,2e-2],
    :ndrops => 1,
    :hdrop_std => 0.2,
    :h₀ => 0.001,
    :ls => 0.002,
    :μ => 0.01,
    #:μ => 0.01,
    #:σ => 0.075,
    :σ => 0.020,
    #:θₛ => 30,
    :θₛ => 50,
    :dθₛ => 0,
    :save_timestep => 10,
    :θτ => 0.0,
    :mass => 12.95,
    :aspect_ratio => 4,
    #:ρ => 1000.0,
    :ρ => 964,
    :τ => 0.0,
    :L => 6,
    :two_dim => false,
    :cfl_safety_factor => 0.9,
    :reproject => true,
)
parameters = dict_list(parameters)

# %%
for p ∈ parameters
    out_dir = "data/outputs/2-D/test"
    filename = savename(p, "nc", accesses=[:hₛ])
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
