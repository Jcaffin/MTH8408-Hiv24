
include("functions.jl")


using LinearAlgebra
using SolverCore, SolverBenchmark
using ADNLPModels, NLPModels
using OptimizationProblems, OptimizationProblems.ADNLPProblems
using JSOSolvers




########################## CRÉATION DU DICTIONNAIRE DE SOLVEURS ##########################
solvers = solvers = Dict(
  :lbfgs1 => model -> lbfgs(model, mem=1),
  :lbfgs5 => model -> lbfgs(model, mem=5),
  :lbfgs20 => model -> lbfgs(model, mem=20),
)


########################## SÉLECTION DES PROBLÈMES POUR LES TEST ##########################
n = 20
ad_problems = (eval(Meta.parse(problem))(;n) for problem ∈ OptimizationProblems.meta[!, :name])


######################### PASSAGE DES PROBLÈMES DANS LES SOLVEURS #########################
stats = bmark_solvers(
  solvers, ad_problems,
  skipif=prob -> (!unconstrained(prob) || get_nvar(prob) > 100 || get_nvar(prob) < 5),
)

########################## MISE EN FORME DU TABLEAU DES RÉSULTATS #########################
cols = [:id, :name, :nvar, :objective, :dual_feas, :neval_obj, :neval_grad, :neval_hess, :iter, :elapsed_time, :status]
header = Dict(
  :nvar => "n",
  :objective => "f(x)",
  :dual_feas => "‖∇f(x)‖",
  :neval_obj => "# f",
  :neval_grad => "# ∇f",
  :neval_hess => "# ∇²f",
  :elapsed_time => "t",
)

for solver ∈ keys(solvers)
  pretty_stats(stats[solver][!, cols], hdr_override=header, tf=tf_markdown)
end


########################### CRÉATION DES PROFILS DE PERFORMANCE ##########################

first_order(df) = df.status .== :first_order
unbounded(df) = df.status .== :unbounded
solved(df) = first_order(df) .| unbounded(df)

costnames = ["time", "obj + grad + hess"]
costs = [
  df -> .!solved(df) .* Inf .+ df.elapsed_time,
  df -> .!solved(df) .* Inf .+ df.neval_obj .+ df.neval_grad .+ df.neval_hess,
]

using Plots
gr()

profile_solvers(stats, costs, costnames)