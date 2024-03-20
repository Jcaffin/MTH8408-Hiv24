using YahooFinance, JuMP, DataFrames, Ipopt, MathOptInterface
include("functions.jl")

debut ="2019-01-13"
fin="2019-01-20"
PF=["NFLX","IBM"]

n = length(PF)

m,Q = assets_cov(PF)



model = Model(Ipopt.Optimizer)
@variable(model, x[1:n] >= 0)

@objective(model, Min, x' * Q * x)

@constraint(model, sum(x) <= 1000)
@constraint(model, m' * x >= 0.5)

optimize!(model)
solution_summary(model)