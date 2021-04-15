from astropy.io import fits
from resultutils import FitResult
from fitutils import get_lsf, concolve_lsf
import numpy as np


datapath = '/Users/mandychen/CUBS/J0454-6116/eso/'

popt = fits.getdata(datapath + 'poptmap_o2_o3_smoothed_sig1.5.fits')
perr = fits.getdata(datapath + 'perrmap_o2_o3_smoothed_sig1.5.fits')
_,hdr = fits.getdata(datapath + 'nb_eso_OII_smoothed_sig1.5.fits', header=True)
flagmap = fits.getdata(datapath + 'flagmap_3sig_OII.fits')

pixcenter = [164, 179]
rmin = 2

z0 = 0.7861
line = [3727.092, 3729.875, 4960.295, 5008.240, 6564.632]
line_z = np.asarray(line)*(1+z0)
mask_ok = (line_z>4700) & (line_z<9000)
line_ok = line_z[mask_ok]
lsf = np.zeros(len(line))
lsf[mask_ok] = get_lsf(line_ok)

result = FitResult(popt=popt, perr=perr, line = line, lsf = lsf, hdr = hdr,
				   flagmap = flagmap, pixcenter = pixcenter, rmin = rmin)
# result.get_vmap(z0=z0, vfile = datapath+'vmap_test.fits',
# 				dvfile = datapath+'dvmap_test.fits', comp='comp1')
result.get_fluxmap(ffile =datapath+'OII_1_comp1_flux_test.fits',
				dffile = datapath+'dOII_1_comp1_flux_test.fits', comp='comp1_o2_1')
result.get_fluxmap(ffile = datapath+'OII_2_comp1_flux_test.fits',
				dffile = datapath+'dOII_2_comp1_flux_test.fits', comp='comp1_o2_2')
result.get_fluxmap(ffile = datapath+'OIII_2_comp1_flux_test.fits',
				dffile = datapath+'dOIII_2_comp1_flux_test.fits', comp='comp1_o3_2')
d1,h = fits.getdata(datapath+'OII_1_comp1_flux_test.fits', header=True)
d2 = fits.getdata(datapath+'OII_2_comp1_flux_test.fits')
fits.writeto(datapath + 'OII_comp1_flux_test.fits',d1+d2, h,overwrite=True)

d1,h = fits.getdata(datapath+'dOII_1_comp1_flux_test.fits', header=True)
d2 = fits.getdata(datapath+'dOII_2_comp1_flux_test.fits')
fits.writeto(datapath + 'dOII_comp1_flux_test.fits',np.sqrt(d1**2+d2**2), h,overwrite=True)