# =============================================================================
# Use dynamical decoupling to suppress crosstalk noise in free evolution of qubits
# =============================================================================
using OpenQuantumTools

# -----------------------------------------------------------------------------
# Define the pulse function
# -----------------------------------------------------------------------------
function constant_pulse(
    t_start_list::Vector{Float64}, duration::Float64;
    # π/2 puluse by default
    Ω_val::Float64=1/(4*duration)
)
    isempty(t_start_list) && throw(ArgumentError("t_start_list must not be empty"))
    duration <= 0 && throw(ArgumentError("duration must be positive"))

    t_end_list = t_start_list .+ duration
    t_pulse_list = sort(vcat(t_start_list, t_end_list))
    ampl = zeros(Float64, length(t_pulse_list))
    @views ampl[1:2:end] .= Ω_val

    return function(t::Real)
        idx = searchsortedlast(t_pulse_list, t)
        return idx == 0 ? 0.0 : ampl[idx]
    end
end

function construct_dd_circuit(
    # number of qubits
    nqubit::Int, 
    # crosstalk strength (here we move to the rotating frame and use RWA so ωq = 0)
    J::Float64,
    # dd gate number and time
    dd_num::Int, dd_time::Real,
    # total simulation time
    tf::Float64,
    # T1 and T2 noise
    T1::Float64, T2::Float64
)
    # crosstalks are XX-couplings in the Pauli-X basis
    H_mat = sum([single_clause([spσx, spσx], [i, i+1], J/2, nqubit) for i in 1:nqubit-1])
    
    ### Construct simulation settings ###
    # free evolution Hamiltonian
    H = SparseHamiltonian([(s) -> 1.0], [H_mat], unit=:h)

    # add T1 noise
    γ1 = 1/T1
    # relaxation is now σ₊
    spσ₊ = sparse(σ₊)
    lind_t1 = [Lindblad(γ1, single_clause([spσ₊], [i], 1, nqubit)) for i in 1:nqubit]
    # dephasing is now σx
    γ2 = 1/T2
    lind_t2 = [Lindblad(γ2, single_clause([spσx], [i], 1, nqubit)) for i in 1:nqubit]

    iset = InteractionSet(vcat(lind_t1, lind_t2)...)

    # staggered dynamical decoupling (X sequences are Z sequences in Pauli-X basis)
    dd_ind_1 = collect(1:2:nqubit)
    dd_sequence_1 = single_clause([spσz for _ in dd_ind_1], [i for i in dd_ind_1], 1, nqubit)
    dd_ind_2 = setdiff(1:nqubit, dd_ind_1)
    dd_sequence_2 = single_clause([spσz for _ in dd_ind_2], [i for i in dd_ind_2], 1, nqubit)
    t_dd_time_1 = range(0, tf, length=dd_num+1)

    τ = ((tf / dd_num) - 2*dd_time) / 2 # free evolution time between two pulses
    t_dd_time_2 = t_dd_time_1[1:end-1] .+ (τ+dd_time)
    pulse_func_1 = constant_pulse(collect(t_dd_time_1), dd_time)
    pulse_func_2 = constant_pulse(collect(t_dd_time_2), dd_time)

    H_dd = SparseHamiltonian([(s) -> 1.0, (s) -> pulse_func_1(s), (s) -> pulse_func_2(s)], [H_mat, dd_sequence_1, dd_sequence_2], unit=:h)
    ϕ0 = sparse(q_translate_state("0"^nqubit))

    evo    = Annealing(H   , ϕ0, interactions=iset, annealing_parameter=(tf, t) -> t)
    evo_dd = Annealing(H_dd, ϕ0, interactions=iset, annealing_parameter=(tf, t) -> t)

    t_dd_time = vcat(t_dd_time_1, t_dd_time_2)
    t_qmc_list = sort([t_dd_time; t_dd_time .+ dd_time])[1:end-1]

    return evo, evo_dd, t_qmc_list
end

# 600 kHz ZZ coupling, 10 ns for single-qubit gate
# 100 μs T1 and 50 μs T2 time
const nqubit = 4
const J = 6*1e-4
const dd_num = 200
const dd_time = 10.0
const tf = 5e3
const T1 = 1e5
const T2 = 5e4

const evo_free, evo_dd, t_qmc_list = construct_dd_circuit(nqubit, J, dd_num, dd_time, tf, T1, T2)

using Base.Threads
using .OpenQMC
using .OpenQMC.ThreadUtils

const Nw = Int(1e6)
const dt = 0.25
const initiator_limit = Int(1e3)
# currently only support Euler and AdamsBashforth2 methods
const alg = :AdamsBashforth2

# Make measurements every 10 ns
struct Measurement
    Nw_diag_re::Int64
    Nw_diag_im::Int64
    fidelity::Float64
end
function measure_fidelity(ρ::QMCState, measurements::Vector{Measurement})
    state = ρ(1).state
    Nw = count_diagonal(ρ)
    fidelity = abs((state.re[] + im*state.im[]) / real(Nw))^2
    push!(measurements, Measurement(Int(real(Nw)), Int(imag(Nw)), fidelity))
end
const measurements_free = Measurement[]
const measurements_dd = Measurement[]
const callback_interval = 10.0

@time ρ_free = ThreadUtils.solve_qmc_threaded(evo_free, t_qmc_list[end], 
                                        dt=dt, Nw=Nw, initiator_limit=initiator_limit, alg=alg, 
                                        callback=measure_fidelity, callback_interval=callback_interval, measurements=measurements_free
                                    )

@time ρ_dd = ThreadUtils.solve_qmc_threaded(evo_dd, t_qmc_list, 
                                        dt=dt, Nw=Nw, initiator_limit=initiator_limit, alg=alg, 
                                        callback=measure_fidelity, callback_interval=callback_interval, measurements=measurements_dd
                                    )

