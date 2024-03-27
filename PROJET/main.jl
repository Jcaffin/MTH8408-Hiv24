using YahooFinance, JuMP, DataFrames, Ipopt, MathOptInterface
include("functions.jl")


re = 0.05 # rendement esperé sur une période de temps
debut ="2019-01-13"
fin="2019-01-20"
PF=["NFLX","IBM"]
n = length(PF)
m,Σ = assets_cov(PF,debut,fin)


model = Model(Ipopt.Optimizer)
@variable(model, ω[1:n] >= 0)
@constraint(model, sum(ω) <= 1)
@constraint(model, m' * ω >= re)
@objective(model, Min, ω' * Σ * ω)

optimize!(model)
solution_summary(model)

value.(ω) 