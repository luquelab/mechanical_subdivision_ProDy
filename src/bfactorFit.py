import numpy as np
import numba as nb

#@nb.njit()
def fastFlucts(evals, evecs, n_modes):
    n_atoms = evecs.shape[0]
    flucts = np.zeros(n_atoms)
    for i in range(n_modes):
        flucts += 1/evals[i]*evecs[:,i]**2

    return flucts

@nb.njit()
def prsFlucts(evals, evecs, n_modes):
    n_atoms = evecs.shape[0]
    flucts = np.zeros(n_atoms)
    for i in range(n_modes):
        for j in range(n_atoms):
            vec = 1/evals[i] * evecs[3*j:3*j+3, i]
            flucts[j] += np.sum(np.abs(np.outer(vec, vec)))
    return flucts

#@nb.njit()
def checkIcoFlucts(flucts):
    F = np.reshape(flucts, (60,-1))
    devs = np.ptp(F, axis=0)
    d = np.max(devs)
    #print('Maximum deviation from icosahedral: ', np.max(devs))
    return d




#@nb.njit(nb.float64(nb.float64[:], nb.float64[:,:]))
def springFit(bfactors, sqFlucts):
    import statsmodels.api as sm
    from settings import intercept

    if intercept:
        sqFlucts = sm.add_constant(sqFlucts)
    M = sm.RLM(bfactors, sqFlucts, M=sm.robust.norms.HuberT())
    results = M.fit()
    #print(results.summary())
    a = results.params[-1]
    stderr = results.bse * np.sqrt(bfactors.shape[0])
    pv = results.pvalues
    ci = results.conf_int(alpha=0.1)

    if intercept:
        b = results.params[0]
    else:
        b = 0

    return a, b, stderr, ci, pv



def fluctFit(evals, evecs, bfactors, forceIco=False, icotol=0.002):
    from settings import fitmodes
    coeffs = []
    ks = []
    flucts = []
    ssrs = []
    pvs = []
    devs = []
    intercepts = []
    nms = []
    from settings import n_modes
    minModes = 12
    if fitmodes:
        for n_modes in range(len(evals)):
            if n_modes < minModes:
                continue
            c, k, intercept, fluct, ssr, stderr, ci, pv, dev = costFunc(evals, evecs, bfactors, n_modes)
            coeffs.append(c)
            ks.append(k)
            flucts.append(fluct)
            ssrs.append(ssr)
            pvs.append(pv)
            devs.append(dev)
            intercepts.append(intercept)
            nms.append(n_modes+1)

        plotModes = True
        if plotModes:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2,1)
            ax[0].plot(nms, coeffs)
            #ax[0].vlines(nModes, np.min(coeffs), np.max(coeffs))
            ax[0].set_xlabel('Number Of Modes')
            ax[0].set_ylabel('CC')
            fig.suptitle('Accuracy vs number of low frequency modes')

            ax[1].plot(nms, devs)
            #ax[1].vlines(nModes, np.min(devs), np.max(devs))
            ax[1].set_xlabel('Number Of Modes')
            ax[1].set_ylabel('Icosahedral deviation')
            plt.show()

        if forceIco:
            icoI = np.nonzero(np.array(devs) < icotol)
            print('Ico indices: ', np.array(nms)[icoI])
            i = np.argmax(np.array(coeffs)[icoI])
            nModes = np.array(nms)[icoI][i]
            coeff = np.array(coeffs)[icoI][i]
            kbest = np.array(ks)[icoI][i]
            fluct = np.array(flucts)[icoI][i]
            ssr = np.array(ssrs)[icoI][i]
            intercept = np.array(intercepts)[icoI][i]
        else:
            i = np.argmax(coeffs)
            nModes = nms[i]
            coeff = coeffs[i]
            kbest = ks[i]
            fluct = flucts[i]
            ssr = ssrs[i]
            intercept = intercepts[i]


        print('Intercept: ', intercept)
        err = standardError(bfactors, fluct, kbest)

        return nModes, coeff, kbest, fluct, err, ssr
    else:
        n_modes = evals.shape[0]
        c, k, intercept, fluct, ssr, stderr, ci, pv, dev = costFunc(evals, evecs, bfactors, n_modes)
        err = standardError(bfactors, fluct, k)
        return n_modes, c, k, fluct, err, ssr


# @nb.njit()
def costFunc(evals, evecs, bfactors, n_modes):
    sqFlucts = fastFlucts(evals,evecs,n_modes)
    #sqFlucts = prsFlucts(evals, evecs, n_modes)
    if sqFlucts.shape[0] != bfactors.shape[0]:
        sqFlucts = np.reshape(sqFlucts, (-1, 3)).sum(axis=-1)

    dev = checkIcoFlucts(sqFlucts)
    k, intercept, stderr, ci, pv = springFit(bfactors, sqFlucts[:,np.newaxis])
    scaledFlucts = k*sqFlucts + intercept
    c = np.corrcoef(bfactors,scaledFlucts)[1,0]
    ssr = np.sum((bfactors - scaledFlucts)**2)
    return c, k, intercept, scaledFlucts, ssr, stderr, ci, pv, dev

def standardError(bfactors, sqFlucts, k):
    n = bfactors.shape[0]
    y = sqFlucts*k
    err = np.sqrt(1/(n-2) * np.sum((bfactors-y)**2)/(np.sum((bfactors - np.mean(bfactors))**2)))
    return err

def confidenceInterval(bfactors, stderr):
    from scipy.stats import t
    alpha = 1-0.95
    df = bfactors.shape[0]
    score = abs(t.ppf(alpha / 2, df-2))
    #score = 1.65
    return score*stderr#*np.sqrt(df)

    #from sklearn.linear_model import HuberRegressor
    #a, _, _, _ = np.linalg.lstsq(sqFlucts, bfactors)
    #a = springFit2(bfactors, sqFlucts)
    #huber = HuberRegressor(fit_intercept=False, tol=0, alpha=0.0).fit(sqFlucts, bfactors)
    #a = huber.coef_
    #b = huber.intercept_
    # outliers = huber.outliers_
    # residuals = np.sum((bfactors[~outliers] - a*sqFlucts[~outliers]) ** 2)
    # r2 = 1 - residuals / (np.sum((bfactors - bfactors.mean()) ** 2))
    # print(r2)
    #r2 = huber.score(a*sqFlucts, bfactors)