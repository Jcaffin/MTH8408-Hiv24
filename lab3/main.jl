using LinearAlgebra
using SolverCore, SolverBenchmark
using ADNLPModels, NLPModels
using OptimizationProblems, OptimizationProblems.ADNLPProblems
using JSOSolvers

include("functions.jl")


# fH(x) = (x[2]+x[1].^2-11).^2+(x[1]+x[2].^2-7).^2
# x0H = [10., 20.]
# himmelblau = ADNLPModel(fH, x0H)

# problem2 = ADNLPModel(x->-x[1]^2, ones(3))

# roz(x) = 100 *  (x[2] - x[1]^2)^2 + (x[1] - 1.0)^2
# rosenbrock = ADNLPModel(roz, [-1.2, 1.0])

# f(x) = x[1]^2 * (2*x[1] - 3) - 6*x[1]*x[2] * (x[1] - x[2] - 1)
# pb_du_cours = ADNLPModel(f, [-1.001, -1.001]) #ou [1.5, .5] ou [.5, .5]


using LinearAlgebra
using SolverCore, SolverBenchmark
using ADNLPModels, NLPModels
using OptimizationProblems, OptimizationProblems.ADNLPProblems
using JSOSolvers


nlp = OptimizationProblems.ADNLPProblems.AMPGO02()
#problems = OptimizationProblems.meta[!, :name]
#limited_bfgs(nlp)
armijo_Newton_cg(nlp)