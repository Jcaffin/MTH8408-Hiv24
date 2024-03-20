using YahooFinance

debut ="2019-01-13"
fin="2019-01-20"

PF=["NFLX","IBM"]


function assets_cov(PF,date_debut,date_fin)
    n=length(PF)
    vals=[]
    for asset in PF
        x =get_symbols(asset, date_debut, date_fin)
        val=values(x["Close"])
        push!(vals,val)
    end
    jours=length(vals[1])

    ############# Calcul de la matrice des rendements #############
    R=zeros(jours-1,n)
    for j =1:n                      # j correspond à l'asset
        for i = 1:jours-1           # i correspond au jour
            R[i,j] = (vals[j][i+1]-vals[j][i])/vals[j][i] # évolution (%) de la valeur de l'asset entre i et i+1
        end
    end

    m = moyenne(R)
    Q = covariance(R)  # Matrice de covariance
    return m,Q
end

function moyenne(R)
    p,a = size(R)
    m = []
    for j = 1:a     # a correspond au nombre d'actifs

        mj = 0
        for i = 1:p # p correspond au nombre de période de temps
            mj += R[i,j] 
        end
        mj = mj/p

        push!(m,mj)
    end
    return m
end

function covariance(R)
    p,a = size(R)
    m = moyenne(R)

    ############# remplissage de la matrice de covariance #############
    Q = zeros(a,a)
    for x = 1:a
        for y = 1:a
            cov_xy = 0
            for i = 1 : p
                cov_xy += (R[i,x] - m[x]) * (R[i,y] - m[y])
            end
            cov_xy = cov_xy/(p-1)
            Q[x,y] = cov_xy
        end
    end

    return Q
end



