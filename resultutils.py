from astropy.io import fits
import numpy as np 
from fitutils import concolve_lsf

class FitResult():
	def __init__(self, popt, perr, line, lsf, hdr = None,
				 flagmap = None, pixcenter = None, rmin = None):
		self.popt = popt
		self.perr = perr
		self.line = line
		self.lsf = lsf
		self.flagmap = flagmap
		self.pixcenter = pixcenter
		self.rmin = rmin
		self.hdr = hdr

	def get_rmask(self):
		nx, ny = self.popt.shape[2], self.popt.shape[1]
		xc, yc = self.pixcenter[0], self.pixcenter[1] # center of QSO
		x, y = np.arange(0,nx), np.arange(0,ny)
		xx, yy = np.meshgrid(x, y)
		r2 = (xx-xc)**2+(yy-yc)**2
		r = np.sqrt(r2)
		mask = (r <= self.rmin)
		return mask

	def get_gaussian_area(self, zi, sigi, ni, linei, scalei):
		zmap = self.popt[zi, :, :]
		sigmap = self.popt[sigi,:,:]
		dsigmap = sigmap/concolve_lsf(sigmap, np.full_like(sigmap,self.lsf[linei]))*self.perr[sigi,:,:]
		sig_ang = concolve_lsf(sigmap, np.full_like(sigmap,self.lsf[linei]))/2.998e5*self.line[linei]*(1+zmap)
		dsig_ang = dsigmap/2.998e5*self.line[linei]*(1+zmap)

		if scalei == 0: 
			nmap, dnmap = self.popt[ni,:,:], self.perr[ni,:,:]
		elif scalei == 1: 
			nmap, dnmap = self.popt[ni,:,:]*1/3, self.perr[ni,:,:]*1/3
		elif scalei == 2: 
			nmap = self.popt[ni,:,:]*self.popt[(ni-1),:,:]
			dnmap = np.sqrt((self.perr[ni,:,:]/self.popt[ni,:,:])**2+(self.perr[(ni-1),:,:]/self.popt[(ni-1),:,:])**2)*nmap
		else: print('Invalid scaling of flux, check your input')

		fluxmap = np.sqrt(2*np.pi)*sig_ang*nmap
		dfluxmap = np.sqrt((dnmap/nmap)**2 + (dsigmap/sigmap)**2)*fluxmap
		return fluxmap, dfluxmap

	def get_vmap(self, z0, vfile, dvfile, comp = 'comp1'):
		if comp == 'comp1': compi = 0
		if comp == 'comp2': compi = 5
		i = compi
		vmap = (self.popt[i,:,:] - z0)/(1 + z0)*2.998e5
		dvmap = self.perr[i,:,:]/(1 + z0)*2.998e5

		if self.flagmap is not None:
			vmap[self.flagmap == 0] = np.nan
			dvmap[self.flagmap == 0] = np.nan
		if self.rmin is not None:
			mask = self.get_rmask()
			vmap[mask] = np.nan
			dvmap[mask] = np.nan
			
		if self.hdr == None:
			fits.writeto(vfile, vmap, overwrite=True)
			fits.writeto(dvfile, dvmap, overwrite=True)
		else:
			fits.writeto(vfile, vmap, self.hdr, overwrite=True)
			fits.writeto(dvfile, dvmap, self.hdr, overwrite=True)

	def get_fluxmap(self, ffile, dffile, comp = 'comp1_o2_1'):
		# z1, sig1, n1, n21, n3, z2, sig2, n5, n65, n7, a1, b1, a2, b2
		# 0.   1.    2.   3.  4.  5.  6.    7.   8.  9. 10. 11.  12. 13
		if comp == 'comp1_o2_1': 
			zi, sigi, ni, linei, scalei = 0, 1, 2, 0, 0
		elif comp == 'comp1_o2_2': 
			zi, sigi, ni, linei, scalei = 0, 1, 3, 1, 2
		elif comp == 'comp1_o3_1': 
			zi, sigi, ni, linei, scalei = 0, 1, 4, 2, 1
		elif comp == 'comp1_o3_2': 
			zi, sigi, ni, linei, scalei = 0, 1, 4, 3, 0
		elif comp == 'comp2_o2_1': 
			zi, sigi, ni, linei, scalei = 5, 6, 7, 0, 0
		elif comp == 'comp2_o2_2': 
			zi, sigi, ni, linei, scalei = 5, 6, 8, 1, 2
		elif comp == 'comp2_o3_1': 
			zi, sigi, ni, linei, scalei = 5, 6, 9, 2, 1
		elif comp == 'comp2_o3_2': 
			zi, sigi, ni, linei, scalei = 5, 6, 9, 3, 0
		else: print('Invalid component name')

		fluxmap, dfluxmap = self.get_gaussian_area(zi, sigi, ni, linei, scalei)
		if self.hdr == None:
			fits.writeto(ffile, fluxmap, overwrite=True)
			fits.writeto(dfffile, dfluxmap, overwrite=True)
		else:
			fits.writeto(ffile, fluxmap, self.hdr, overwrite=True)
			fits.writeto(dffile, dfluxmap, self.hdr, overwrite=True)
		