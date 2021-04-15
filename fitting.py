from mpdaf.obj import Cube
from fitutils import Fitting, FitFunc, get_lsf
import numpy as np

# load datacubes
datafile = '/Users/mandychen/CUBS/J0454-6116/eso/qsub_HRSDI_eso_smoothed_sig1.5.fits'
cube = Cube(datafile)

# set up the fitting
wavemin1, wavemax1 = 6620, 6700
wavemin2, wavemax2 = 8830, 8980
wavebounds = [wavemin1, wavemax1, wavemin2, wavemax2]

# p0 = [0.7861, 75, 40, 1, 10, 1, 0, 1, 0]
# plim = ((0, 0, 0, 0.35, 0, -np.inf, -np.inf, -np.inf, -np.inf), 
#        (np.inf, np.inf, np.inf, 1.5, np.inf, np.inf, np.inf, np.inf, np.inf))
p0 = [0.7861, 75, 40, 1, 10, 0.7861, 75, 40, 1, 10, 1, 0, 1, 0]
plim = ((0, 0, 0, 0.35, 0, 0, 0, 0, 0.35, 0, -np.inf, -np.inf, -np.inf, -np.inf), 
       (np.inf, np.inf, np.inf, 1.5, np.inf, np.inf, np.inf, np.inf, 1.5, np.inf, np.inf, np.inf, np.inf, np.inf))
xlim, ylim = [120, 210], [120, 230]
# xlim, ylim = [169, 171], [172, 174]

line = [3727.092, 3729.875, 4960.295, 5008.240, 6564.632]
z0 = 0.7861
line_z = np.asarray(line)*(1+z0)
mask_ok = (line_z>4700) & (line_z<9000)
line_ok = line_z[mask_ok]
lsf = np.zeros(len(line))
lsf[mask_ok] = get_lsf(line_ok)
fitfunc = FitFunc(line, lsf, wavebounds)


# do the fitting
fitting = Fitting(cube = cube, wavebounds = wavebounds, func = fitfunc.gauss_o2_o3_2comp_w_cont, p0 = p0, plim = plim,
					xlim = xlim, ylim = ylim)
fitting.dofit()
fitting.savefitting('/Users/mandychen/CUBS/J0454-6116/eso/poptmap_o2_o3_2comp_smoothed_sig1.5_1.fits',
					'/Users/mandychen/CUBS/J0454-6116/eso/perrmap_o2_o3_2comp_smoothed_sig1.5_1.fits')


# save restuls