using ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems
include("fcts_lab.jl")

problems_names = setdiff(names(OptimizationProblems.ADNLPProblems))
ad_problems = (eval(Meta.parse(problem))() for problem ∈ problems_names)
pb_sc = filter(nlp -> test_var_eq(nlp), collect(ad_problems))