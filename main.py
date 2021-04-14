from mpdaf.obj import Cube
from fitutils import Fitting
import numpy as np

# load datacubes
datafile = '/Users/mandychen/CUBS/J0454-6116/eso/qsub_HRSDI_eso_smoothed_sig1.5.fits'
cube = Cube(datafile)

# set up the fitting
wavemin1, wavemax1 = 6620, 6700
wavemin2, wavemax2 = 8830, 8980
wavebounds = [wavemin1, wavemax1, wavemin2, wavemax2]

p0 = [0.7861, 75, 40, 1, 10, 1, 0, 1, 0]
plim = ((0, 0, 0, 0.35, 0, -np.inf, -np.inf, -np.inf, -np.inf), 
       (np.inf, np.inf, np.inf, 1.5, np.inf, np.inf, np.inf, np.inf, np.inf))
# p0 = [0.7861, 0.7864, 75, 35, 40, 1, 10, 40, 1, 10, 1, 0, 1, 0]
# plim = ((0, 0, 0, 0, 0, 0.35, 0, 0, 0.35, 0, -np.inf, -np.inf, -np.inf, -np.inf), 
#        (np.inf, np.inf, np.inf, np.inf, np.inf, 1.5, np.inf, np.inf, 1.5, np.inf, np.inf, np.inf, np.inf, np.inf))
# xlim, ylim = [120, 210], [120, 230]
xlim, ylim = [169, 171], [172, 174]


fitting = Fitting(cube = cube, wavebounds = wavebounds, p0 = p0, plim = plim,
					xlim = xlim, ylim = ylim)
fitting.set_func(nfunc = 1)
fitting.dofit()
fitting.savefitting('/Users/mandychen/CUBS/J0454-6116/eso/testpopt.fits',
					'/Users/mandychen/CUBS/J0454-6116/eso/testperr.fits')

# do the fitting



# save restuls