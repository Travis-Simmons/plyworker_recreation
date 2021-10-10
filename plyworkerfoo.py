
import emcee

# (A) ydata = read plyworker pointcloud


def lnprior(p):
    return 0

def lnlike(p, xdata, ydata):
	# (1) run command line with fx = [0] etc.
    # (2) ymodel = read in new point cloud
    lnlikelihood = (-0.5 * ((ymodel - ydata))**2).sum()
    return lnlikelihood

def lnprob(p):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p)


ndim, nwalkers, nsteps = 4, 14, 10000

"""
parameters

fx
fy
cx
cy
"""
initial_pos_fx = [np.random.uniform(0,1000) for i in range(nwalkers)]
initial_pos_fy = [np.random.uniform(0,1000) for i in range(nwalkers)]
initial_pos_cx = [np.random.uniform(0,1000) for i in range(nwalkers)]
initial_pos_cy = [np.random.uniform(0,1000) for i in range(nwalkers)]

pos = [initial_pos_fx, initial_pos_fy, initial_pos_cx, initial_pos_cy]

#xdata = np.asarray(commandline_pointcloud.points)
#ydata = np.asarray(plyworker_pointcloud.points)

#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xdata, ydata))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=())
sampler.run_mcmc(pos, nsteps)

