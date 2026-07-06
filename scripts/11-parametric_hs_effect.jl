# %%
using DropletSpreadingSim2

using DifferentialEquations, Sundials, Logging, DrWatson
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger(stderr))

# %%
function do_simulate(p; filename)
    @unpack h₀, ls, σ, ρ, μ, α, τ, θτ, L, hₛ, hₛ_ratio, θₛ, dθₛ, hₛ, aspect_ratio, tmax,
    mass, ndrops, hdrop_std, two_dim, reproject = p
    θₐ = deg2rad(θₛ + dθₛ)
    θᵣ = deg2rad(θₛ - dθₛ)
    #tmax = 1000
    @show(tmax)
    # %%
    experiment = DropletSpreadingExperiment(; h₀, ls, σ, ρ, μ, α, τ, θτ, L, hₛ_ratio, hₛ, θₐ, θᵣ, 
        aspect_ratio, mass, ndrops, hdrop_std, two_dim)

    # %%*(sin(α*pi/180))
    prob = ODEProblem(experiment, (0.0, p[:tmax]*(sin(α*pi/180))))
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
	#Rosenbrock23(autodiff=false);
        #AutoTsit5(Rosenbrock23());
        #Tsit5();
        #Euler();
        SSPRK432();
	#RadauIIA3();
	#KenCarp4();
	#Rodas5P(autodiff=false);
        callback=CallbackSet(callbacks...),
        progress=true,
        progress_steps=1,
        save_everystep=false,
        saveat=get(p, :keep_timestep, []),
        dt=1e-4,
    )

    return sol, experiment
end
#SSPRK432();
# %%
parameters = Dict(
    :tmax => 500,
    :hₛ_ratio => 4,
    :hₛ => [0.02],
    #:hₛ => 5e-2,
    :ndrops => 1,
    :hdrop_std => 0.2,
    :h₀ => 0.001,
    #:ls => [2e-1,1e-1,5e-2,2e-2,1e-2],
    :ls => 0.002,
    #:ls => [2e-2,1e-2,5e-3,2e-3,1e-3,5e-4,2e-4],
    #:μ => 0.001,
    :μ => 0.01,
    :σ => 0.02,
    #:σ => 0.072,
    #:θₛ => 11,
    :θₛ => 48,
    #:θₛ => 52.45,
    #:θₛ => [15,30,45,60,75],
    #:dθₛ => [0,10],
    :dθₛ => [0,2.5,5,10],
    :save_timestep => 10,
    :θτ => 0.0,
    :α => [5,15],
    :mass => 6,
    #:mass => 27.17,
    :aspect_ratio => 5,
    #:ρ => 1000.0,
    #:ρ => 998,
    :ρ => 936,
    :τ => 0.0,
    :L => 5,
    :two_dim => false,
    :cfl_safety_factor => 0.9,
    :reproject => false,
)
parameters = dict_list(parameters)

# %%
for p ∈ parameters
    #params = (hₛ="0.05", hₛ_ratio="4")
    #params = (α=0,)
    out_dir = "data/outputs/2-D/dynamicontactangle"
    filename = savename(p, "nc", accesses=[:hₛ, :dθₛ, :α])
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
