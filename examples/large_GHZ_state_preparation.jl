# =============================================================================
# Construct a large GHZ state preparation circuit and simulate it using OpenQMC
# =============================================================================
import OpenQuantumTools: spσz, spσx, spσi, σ₋, q_translate_state
import Kronecker: ⊗
import LinearAlgebra: Diagonal, I
import SparseArrays: sparse

# -----------------------------------------------------------------------------
# Define the pulse function
# -----------------------------------------------------------------------------
function constant_pulse(
    t_start_time::Real, Ω_val::To, duration::Real
) where To<:Union{Float64, ComplexF64}
    t_end_time = t_start_time + duration

    return function(t)
        return t_start_time <= t < t_end_time ? Ω_val : 0
    end
end

# -----------------------------------------------------------------------------
# Define the GHZ state preparation circuit
# -----------------------------------------------------------------------------
function construct_ghz_circuit(
    # number of qubits
    nqubit::Int, 
    # qubit frequency and crosstalk strength
    J::Float64,
    # calibration and gate time
    calibration_time::Real, gate_time::Real,
    # T1 and T2 noise
    T1::Float64, T2::Float64
)

    # construct the lazy representation for the Hamiltonian
    H_mat = CompositeKroneckerProduct(
        [J*2π for _ in 1:nqubit-1],
        [lazy_single_clause([spσz, spσz], [i, i+1], nqubit) for i in 1:nqubit-1]
    )

    # CNOT gate (10 ns calibration and 50 ns gate time)
    amp = 1 / (calibration_time*8)
    Ω_val = 1 / (gate_time*8)
    total_gate_time = calibration_time + gate_time
    tf = total_gate_time * (nqubit-1) # total circuit time

    # construct Hamiltonians (i.e., crosstalk and gate Hamiltonians)
    calibration_time_list = [n*total_gate_time for n in 0:nqubit-2]
    gate_time_list = [n*total_gate_time + calibration_time for n in 0:nqubit-2]
    func_list = Function[]
    gate_list = []
    # crosstalk Hamiltonian
    push!(func_list, (s) -> 1)
    push!(gate_list, H_mat)
    # CNOT gate Hamiltonian
    for i in 1:nqubit-1
        push!(func_list, (s) -> constant_pulse(calibration_time_list[i], -amp, calibration_time)(s))
        push!(gate_list, 
            CompositeKroneckerProduct(
                [2π, 2π, -2π],
                [lazy_single_clause([spσz], [i], nqubit), lazy_single_clause([spσx], [i+1], nqubit), spσi⊗nqubit]
            )
        )
        push!(func_list, (s) -> constant_pulse(gate_time_list[i], Ω_val, gate_time)(s))
        push!(gate_list, CompositeKroneckerProduct([2π], [lazy_single_clause([spσz, spσx], [i, i+1], nqubit)]))
    end

    # add T1 noise
    γ1 = 1/T1
    spσ₋ = sparse(σ₋)
    lind_t1 = [lazy_single_clause([spσ₋], [i], nqubit) for i in 1:nqubit]
    # add T2 noise
    γ2 = 1/T2
    lind_t2 = [lazy_single_clause([spσz], [i], nqubit) for i in 1:nqubit]
    iset = vcat(lind_t1, lind_t2)
    γ = vcat([(s) -> γ1 for _ in 1:nqubit], [(s) -> γ2 for _ in 1:nqubit])

    ψ0 = sparse(q_translate_state("(0"*"0"^(nqubit-1)*")+(1"*"0"^(nqubit-1)*")", normal=true))

    evo = LazyEvolution(ψ0, gate_list, func_list, iset, γ)
    t_qmc_list = sort(vcat(calibration_time_list, gate_time_list, [Float64(tf)]))

    return evo, t_qmc_list
end

# 50 kHz ZZ coupling, 10 ns for single-qubit gate calibration, 50 ns for CNOT gate
# 50 μs T1 and 30 μs T2 time
const nqubit = 5
const J = 5e-5
const calibration_time = 10.0
const gate_time = 50.0
const T1 = 5e4
const T2 = 3e4
const evo, t_qmc_list = construct_ghz_circuit(nqubit, J, calibration_time, gate_time, T1, T2)

using Base.Threads
using .OpenQMC
using .OpenQMC.ThreadUtils

const Nw = Int(1e6)
const dt = 0.05
const initiator_limit = Int(1e2)
# currently only support Euler and AdamsBashforth2 methods
const alg = :AdamsBashforth2

@time ρ = ThreadUtils.solve_qmc_threaded(evo, t_qmc_list, 
                                        dt=dt, Nw=Nw, initiator_limit=initiator_limit, alg=alg
                                    )
