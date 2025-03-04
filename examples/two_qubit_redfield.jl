# =============================================================================
# Simulating 2-qubit model under Redfield (non-Markovian) dynamics
# =============================================================================
using OpenQuantumTools

# -----------------------------------------------------------------------------
# Construct the evolution object defined by Redfield
# -----------------------------------------------------------------------------
# choose model parameters s.t. λ1<0
ω1 = 0.25
ω2 = 0.5

γ1 = 1.0
γ2 = 4.0
α = 3.0
κ = 1.0

λ1 = (γ1 + γ2) / 4 - sqrt((γ1^2 + γ2^2 + 8 * κ^2) / 8)
λ2 = (γ1 + γ2) / 4 + sqrt((γ1^2 + γ2^2 + 8 * κ^2) / 8)

# initial state
ψ0 = sparse([sqrt(1/3), sqrt(1/3), sqrt(1/3), 0])

# jump operators
L1 = adjoint(
[0.0 0.0 0.0 0.0
 1.0 0.0 0.0 0.0
 0.0 0.0 0.0 0.0
 0.0 0.0 1.0 0.0])

L2 = adjoint(
[0.0 0.0 0.0 0.0
 0.0 0.0 0.0 0.0
 1.0 0.0 0.0 0.0
 0.0 1.0 0.0 0.0])

### Define Hamiltonian
norm1 = sqrt(
    1 + (γ2 - γ1 + sqrt(2) * sqrt(γ1^2 + γ2^2 + 8*κ^2))^2
    / ((γ1 + γ2) ^ 2 + 16 * κ^2))

norm2 = sqrt(
    1 + (γ2 - γ1 - sqrt(2) * sqrt(γ1^2 + γ2^2 + 8*κ^2))^2
    / ((γ1 + γ2) ^ 2 + 16 * κ^2))

U = [
    (γ1-γ2-sqrt(2)*sqrt(γ1^2 + γ2^2 + 8*κ^2))/(γ1 + γ2 + 4im*κ)/norm1 (γ1-γ2+sqrt(2)*sqrt(γ1^2 + γ2^2 + 8*κ^2))/(γ1 + γ2 + 4im*κ)/norm2
    1/norm1 1/norm2
]

# Write σ₊/σ₋ in terms of L using the explicit definition of U
Udag = adjoint(U)
σ₋1 = Udag[1, 1] * L1 + Udag[2, 1] * L2
σ₋2 = Udag[1, 2] * L1 + Udag[2, 2] * L2
σ₊1 = adjoint(σ₋1)
σ₊2 = adjoint(σ₋2)

Hmat = ((ω1 + α) * σ₊1 * σ₋1
     + (ω2 + α + κ) * σ₊2 * σ₋2
     + (α + κ / 2 - im * (γ1 - γ2) / 8) * σ₊2 * σ₋1
     + (α + κ / 2 - im * (γ2 - γ1) / 8) * σ₊1 * σ₋2)
Hmat = sparse(Hmat)
H = SparseHamiltonian([(s) -> 1], [Hmat], unit=:ħ)

lind = [Lindblad(λ1, sparse(L1)), Lindblad(λ2, sparse(L2))]
iset = InteractionSet(lind...)

evo = Annealing(H, ψ0, interactions=iset, annealing_parameter=(tf, t) -> t)

# -----------------------------------------------------------------------------
# Simulate the evoultion
# -----------------------------------------------------------------------------
using Base.Threads
using .OpenQMC
using .OpenQMC.ThreadUtils

const Nw = Int(1e6)
const dt = 0.005
const initiator_limit = Int(1e3)
# currently only support Euler and AdamsBashforth2 methods
const alg = :AdamsBashforth2

tf = 2.5
t_qmc_list = [0.0, tf]

const callback_interval = 0.05
struct Measurement
    # diagonal population
    state_00_pop::Float64
    state_11_pop::Float64
    state_22_pop::Float64
    # off-diagonal population
    state_01_pop::ComplexF64
    state_02_pop::ComplexF64
    state_12_pop::ComplexF64
    # diagonal walker population
    Nw_diag_re::Int
    Nw_diag_im::Int 
end

const measurement_list = Measurement[]
const t_list = []

function measure_population(ρ::QMCState, t::Real, measurements::AbstractVector{T}) where T 
    Nw_diag = ρ.nw_diag_re[] + im*ρ.nw_diag_im[]

    state_00 = ρ(1).state
    state_00_pop = real((state_00.re[] + im*state_00.im[]) / Nw_diag)
    state_11 = ρ(6).state
    state_11_pop = real((state_11.re[] + im*state_11.im[]) / Nw_diag)
    state_22 = ρ(11).state
    state_22_pop = real((state_22.re[] + im*state_22.im[]) / Nw_diag)
    
    state_01 = ρ(2).state
    state_01_pop = (state_01.re[] + im*state_01.im[])  / Nw_diag
    state_02 = ρ(3).state
    state_02_pop = (state_02.re[] + im*state_02.im[])  / Nw_diag
    state_12 = ρ(7).state
    state_12_pop = (state_12.re[] + im*state_12.im[])  / Nw_diag

    measurement = Measurement(state_00_pop, state_11_pop, state_22_pop, state_01_pop, state_02_pop, state_12_pop, real(Nw_diag), imag(Nw_diag))
    push!(measurements, measurement)
    push!(t_list, t)

    return nothing
end

ρ = ThreadUtils.solve_qmc_threaded(evo, t_qmc_list, dt=dt, Nw=Nw, initiator_limit=initiator_limit, alg=alg,
                                    callback=measure_population, callback_interval=callback_interval, measurements=measurement_list
                                )
