using LinearAlgebra, NLPModels, Printf
using ADNLPModels
using SolverCore
using LinearOperators
using NLPModels

function armijo(xk, dk, fk, gk, slope, nlp :: AbstractNLPModel; τ1 = 1.0e-4, t_update = 1.5)
    t = 1.0
    fk_new = obj(nlp, xk + dk) # t = 1.0
    while fk_new > fk + τ1 * t * slope
      t /= t_update
      fk_new = obj(nlp, xk + t * dk)
    end
    return t, fk_new
  end

  function limited_bfgs(nlp      :: AbstractNLPModel;
    x        :: AbstractVector = nlp.meta.x0,
    atol     :: Real = √eps(eltype(x)), 
    rtol     :: Real = √eps(eltype(x)),
    max_eval :: Int = -1,
    max_time :: Float64 = 30.0,
    f_min    :: Float64 = -1.0e16,
    verbose  :: Bool = true,
    mem      :: Int = 5)
    start_time = time()
    elapsed_time = 0.0

    T = eltype(x)
    n = nlp.meta.nvar

    xt = zeros(T, n)
    ∇ft = zeros(T, n)

    f = obj(nlp, x)
    ∇f = grad(nlp, x)
    #################################################
    H = InverseLBFGSOperator(T,n)
    #################################################

    ∇fNorm = norm(∇f) #nrm2(n, ∇f)
    ϵ = atol + rtol * ∇fNorm
    iter = 0

    @info log_header([:iter, :f, :dual, :slope, :bk], [Int, T, T, T, T],
    hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"∇fᵀd"))

    optimal = ∇fNorm ≤ ϵ
    unbdd = f ≤ f_min
    tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
    stalled = false
    status = :unknown

    while !(optimal || tired || stalled || unbdd)

        #################################################
        d = -H*∇f
        #################################################
        slope = dot(d, ∇f)
        if slope ≥ 0
        @error "not a descent direction" slope
        status = :not_desc
        stalled = true
        continue
        end

        # Perform improved Armijo linesearch.
        t, ft = armijo(x, d, f, ∇f, slope, nlp)

        @info log_row(Any[iter, f, ∇fNorm, slope, t])

        # Update L-BFGS approximation.
        xt = x + t * d
        ∇ft = grad(nlp, xt) # grad!(nlp, xt, ∇ft)
        #################################################
        push!(H,xt-x,∇ft-∇f)
        #################################################

        # Move on.
        x = xt
        f = ft
        ∇f = ∇ft

        ∇fNorm = norm(∇f) #nrm2(n, ∇f)
        iter = iter + 1

        optimal = ∇fNorm ≤ ϵ
        unbdd = f ≤ f_min
        elapsed_time = time() - start_time
        tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
    end
    @info log_row(Any[iter, f, ∇fNorm])

    if optimal
        status = :first_order
    elseif tired
        if neval_obj(nlp) > max_eval ≥ 0
            status = :max_eval
        elseif elapsed_time > max_time
            status = :max_time
        end
    elseif unbdd
        status = :unbounded
    end

    return GenericExecutionStats(
            nlp,
            status=status,
            solution=x,
            objective=f,
            dual_feas=∇fNorm,
            iter=iter,
            elapsed_time=elapsed_time,
            )
end


function cg_optim(H, ∇f)
    #setup the tolerance:
    n∇f = norm(∇f)
#####################################
    ϵk = min(0.5, √(n∇f))*n∇f
####################################
    n = length(∇f)
    z = zeros(n)
    r = ∇f
    d = -r
    
    j = 0
    while norm(r) ≥ ϵk && j < 3 * n
###############################################
        if dot(d, H * d) ≤ 0
            if j == 0
                return -∇f
            else
                return z 
            end
        end
##############################################
        α = dot(r,r)/dot(d,H*d)
##############################################        
        z += α * d
        nrr2 = dot(r, r)
        r += α * H * d
##############################################
        β  = dot(r, r)/nrr2
##############################################
        d  = -r + β * d
        j += 1
    end
    return z
end

function armijo_Newton_cg(nlp      :: AbstractNLPModel;
    x        :: AbstractVector = nlp.meta.x0,
    atol     :: Real = √eps(eltype(x)), 
    rtol     :: Real = √eps(eltype(x)),
    max_eval :: Int = -1,
    max_time :: Float64 = 30.0,
    f_min    :: Float64 = -1.0e16)
start_time = time()
elapsed_time = 0.0

T = eltype(x)
n = nlp.meta.nvar

f = obj(nlp, x)
∇f = grad(nlp, x)
#################################################
H =  hess_op(nlp,x)
#################################################

∇fNorm = norm(∇f) #nrm2(n, ∇f)
ϵ = atol + rtol * ∇fNorm
iter = 0

@info log_header([:iter, :f, :dual, :slope, :bk], [Int, T, T, T, T],
hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"∇fᵀd"))

optimal = ∇fNorm ≤ ϵ
unbdd = f ≤ f_min
tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
stalled = false
status = :unknown

while !(optimal || tired || stalled || unbdd)

d = cg_optim(H, ∇f)

slope = dot(d, ∇f)
if slope ≥ 0
@error "not a descent direction" slope
status = :not_desc
stalled = true
continue
end

# Perform improved Armijo linesearch.
t, f = armijo(x, d, f, ∇f, slope, nlp)

@info log_row(Any[iter, f, ∇fNorm, slope, t])

# Update L-BFGS approximation.
x += t * d
∇f = grad(nlp, x)
#################################################
H = hess_op(nlp,x)
#################################################

∇fNorm = norm(∇f) #nrm2(n, ∇f)
iter = iter + 1

optimal = ∇fNorm ≤ ϵ
unbdd = f ≤ f_min
elapsed_time = time() - start_time
tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
end
@info log_row(Any[iter, f, ∇fNorm])

if optimal
status = :first_order
elseif tired
if neval_obj(nlp) > max_eval ≥ 0
status = :max_eval
elseif elapsed_time > max_time
status = :max_time
end
elseif unbdd
status = :unbounded
end

return GenericExecutionStats(nlp, status = status, solution=x, objective=f, dual_feas=∇fNorm,
         iter=iter, elapsed_time=elapsed_time)
end

