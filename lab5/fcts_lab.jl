using LinearAlgebra, ADNLPModels, Logging, NLPModels, NLPModelsIpopt, Printf, SolverCore, Test, NLPModelsIpopt #,Krylov

function dsol(A, b, ϵ; solver :: Function = lsqr)
    (d, stats) = solver(A, b, atol = ϵ)
    return d
end


function quad_penalty_adnlp(nlp :: ADNLPModel, ρ :: Real)
    f = x -> NLPModels.obj(nlp,x)
    c = x -> NLPModels.cons(nlp,x)
    f_quad = x -> f(x) + (ρ/2) * norm(c(x))^2
    nlp_quad = ADNLPModel(f_quad, nlp.meta.x0)
   return nlp_quad
end

function KKT_eq_constraint(nlp :: AbstractNLPModel, x, λ; ϵ=10^(-10))
    grad_f = grad(nlp,x)
    jac_c = jac(nlp,x)
    test1 = isapprox(grad_f, jac_c'*λ, atol=ϵ)
    test2 = isapprox(cons(nlp,x), zeros(length(λ)), atol=ϵ)
    return test1 && test2
 end

 function quad_penalty(nlp      :: AbstractNLPModel,
    x        :: AbstractVector; 
    ϵ        :: AbstractFloat = 1e-3,
    η        :: AbstractFloat = 1e6, 
    σ        :: AbstractFloat = 2.0,
    max_eval :: Int = 1_000, 
    max_time :: AbstractFloat = 60.,
    max_iter :: Int = typemax(Int64)
    )
##### Initialiser cx et gx au point x;
cx = cons(nlp,x)
gx = grad(nlp,x)
######################################################
normcx = normcx_old = norm(cx)

ρ = 1.

iter = 0    

el_time = 0.0
tired   = neval_cons(nlp) > max_eval || el_time > max_time
status  = :unknown

start_time = time()
too_small  = false
normdual   = norm(gx) #exceptionnellement on ne va pas vérifier toute l'optimalité au début.
optimal    = max(normcx, normdual) ≤ ϵ

nlp_quad   = quad_penalty_adnlp(nlp, ρ)

@info log_header([:iter, :nf, :primal, :status, :nd, :Δ],
[Int, Int, Float64, String, Float64, Float64],
hdr_override=Dict(:nf => "#F", :primal => "‖F(x)‖", :nd => "‖d‖"))

while !(optimal || tired || too_small)

#Appeler Ipopt pour résoudre le problème pénalisé en partant du point x0 = x.
#utiliser l'option print_level = 0 pour enlever les affichages d'ipopt.
stats = ipopt(nlp; x0 = x, print_level = 0)
################################################

if stats.status == :first_order
###### Mettre à jour cx avec la solution renvoyé par Ipopt
x = stats.solution
cx = cons(nlp,x)
##########################################################
normcx_old = normcx
normcx = norm(cx)
end

if normcx_old > 0.95 * normcx
ρ *= σ
end

@info log_row(Any[iter, neval_cons(nlp), normcx, stats.status])

nlp_quad   = quad_penalty_adnlp(nlp, ρ)

el_time      = time() - start_time
iter   += 1
many_evals   = neval_cons(nlp) > max_eval
iter_limit   = iter > max_iter
tired        = many_evals || el_time > max_time || iter_limit || ρ ≥ η
##### Utiliser la réalisabilité dual renvoyé par Ipopt pour `normdual`
normdual     = stats.dual_feas
###################################################################
optimal      = max(normcx, normdual) ≤ ϵ
end

status = if optimal 
:first_order
elseif tired
if neval_cons(nlp) > max_eval
:max_eval
elseif el_time > max_time
:max_time
elseif iter > max_iter
:max_iter
else
:unknown
end
elseif too_small
:stalled
else
:unknown
end

return GenericExecutionStats(nlp, status = status, solution = x,
               objective = obj(nlp, x),
               primal_feas = normcx,
               dual_feas = normdual,
               multipliers = stats.multipliers,
               iter = iter, 
               elapsed_time = el_time,
               solver_specific = Dict(:penalty => ρ))
end


function test_var_eq(nlp)
    L = nlp.meta.lvar
    U = nlp.meta.uvar
    n = length(L)
    not_too_big = n < 5000
    with_constraints = false
    eq_constraints = true
    for k = 1:length(L)
        if L[k] > -Inf && U[k] < Inf
            with_constraints = true
            if L[k] != U[k]
                eq_constraints = false
            end
        end
    end
    return with_constraints && eq_constraints && not_too_big
end