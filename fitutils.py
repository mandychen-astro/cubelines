import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from astropy.io import fits
import time


def get_lsf(wave):
    l0, r0 = np.loadtxt('/Users/mandychen/CUBS/muse_lsf.dat',unpack=True)
    r = interp1d(l0, r0)(wave)
    lsf = 2.998e5/r
    return lsf

def concolve_lsf(sig, lsf):
    a = 2*np.sqrt(2*np.log(2))
    return np.sqrt(sig**2+(lsf/a)**2)

def wave2vel(line,z,wave):
    vel = (wave-line*(1+z))/(line*(1+z))*2.998e5
    return vel

def gauss(x, mu, sig, n):
    return n*np.exp(-(x-mu)**2/(2*sig**2))

class FitFunc():
	def __init__(self, line, lsf, wavebounds):
		self.line = line
		self.lsf = lsf
		self.wavemin1, self.wavemax1 = wavebounds[0], wavebounds[1]
		self.wavemin2, self.wavemax2 = wavebounds[2], wavebounds[3]

	def gauss_o2_o3_w_cont(self, x, z, sig, n1, n21, n3, a1, b1, a2, b2):
		n2 = n1*n21
		n4 = n3*3.
		mask1 = (x>self.wavemin1) & (x<self.wavemax1)
		mask2 = (x>self.wavemin2) & (x<self.wavemax2)
		x1, x2 = x[mask1], x[mask2]
		cont1 = a1 + x1*b1
		cont2 = a2 + x2*b2
		cont = np.zeros(len(x))
		cont[mask1] = cont1
		cont[mask2] = cont2
		g1 = gauss(x, self.line[0]*(1+z), concolve_lsf(sig, self.lsf[0])/2.998e5*self.line[0]*(1+z), n1)
		g2 = gauss(x, self.line[1]*(1+z), concolve_lsf(sig, self.lsf[1])/2.998e5*self.line[1]*(1+z), n2)
		g3 = gauss(x, self.line[2]*(1+z), concolve_lsf(sig, self.lsf[2])/2.998e5*self.line[2]*(1+z), n3)
		g4 = gauss(x, self.line[3]*(1+z), concolve_lsf(sig, self.lsf[3])/2.998e5*self.line[3]*(1+z), n4)
		return g1+g2+g3+g4+cont

	def gauss_o2_o3_2comp_w_cont(self, x, z1, sig1, n1, n21, n3, z2, sig2, n5, n65, n7, a1, b1, a2, b2):
		n2 = n1*n21
		n4 = n3*3.
		n6 = n5*n65
		n8 = n7*3.
		mask1 = (x>self.wavemin1) & (x<self.wavemax1)
		mask2 = (x>self.wavemin2) & (x<self.wavemax2)
		x1, x2 = x[mask1], x[mask2]
		cont1 = a1 + x1*b1
		cont2 = a2 + x2*b2
		cont = np.zeros(len(x))
		cont[mask1] = cont1
		cont[mask2] = cont2
		g1 = gauss(x, self.line[0]*(1+z1), concolve_lsf(sig1, self.lsf[0])/2.998e5*self.line[0]*(1+z1), n1)
		g2 = gauss(x, self.line[1]*(1+z1), concolve_lsf(sig1, self.lsf[1])/2.998e5*self.line[1]*(1+z1), n2)
		g3 = gauss(x, self.line[2]*(1+z1), concolve_lsf(sig1, self.lsf[2])/2.998e5*self.line[2]*(1+z1), n3)
		g4 = gauss(x, self.line[3]*(1+z1), concolve_lsf(sig1, self.lsf[3])/2.998e5*self.line[3]*(1+z1), n4)
		g5 = gauss(x, self.line[0]*(1+z2), concolve_lsf(sig2, self.lsf[0])/2.998e5*self.line[0]*(1+z2), n5)
		g6 = gauss(x, self.line[1]*(1+z2), concolve_lsf(sig2, self.lsf[1])/2.998e5*self.line[1]*(1+z2), n6)
		g7 = gauss(x, self.line[2]*(1+z2), concolve_lsf(sig2, self.lsf[2])/2.998e5*self.line[2]*(1+z2), n7)
		g8 = gauss(x, self.line[3]*(1+z2), concolve_lsf(sig2, self.lsf[3])/2.998e5*self.line[3]*(1+z2), n8)

		return g1+g2+g3+g4+g5+g6+g7+g8+cont

class Fitting():
	def __init__(self, cube, wavebounds, func, p0, plim, xlim, ylim):
		self.wave = cube.wave.coord()
		self.data = cube.data.data
		self.err = np.sqrt(cube.var.data)
		self.dh = cube.data_header
		self.wavebounds = wavebounds
		self.p0 = p0
		self.plim = plim
		self.xlim = xlim
		self.ylim = ylim
		self.func = func

	def get_wavemask(self):
		wavemin1, wavemax1 = self.wavebounds[0], self.wavebounds[1]
		wavemin2, wavemax2 = self.wavebounds[2], self.wavebounds[3]
		mask1 = (self.wave>wavemin1) & (self.wave<wavemax1)
		mask2 = (self.wave>wavemin2) & (self.wave<wavemax2)
		return mask1, mask2

	def dofit(self):
		nx, ny = self.data.shape[2], self.data.shape[1]		
		mask1, mask2 = self.get_wavemask()
		wavefit = np.concatenate((self.wave[mask1], self.wave[mask2]))
		dof = len(wavefit) - len(self.p0)
		self.poptmap = np.zeros(shape=(len(self.p0), ny, nx))
		self.perrmap = np.zeros(shape=(len(self.p0), ny, nx))
		self.chi2map = np.zeros(shape=(ny, nx))
		self.poptmap[:,:,:], self.perrmap[:,:,:], self.chi2map[:,:] = np.nan, np.nan, np.nan

		t0 = time.time()
		for y in range(self.ylim[0], self.ylim[1]):
			for x in range(self.xlim[0], self.xlim[1]):
				dataspec = self.data[:, y, x]
				errspec = self.err[:, y, x]

				fluxfit = np.concatenate((dataspec[mask1], dataspec[mask2]))
				errfit = np.concatenate((errspec[mask1], errspec[mask2]))
				try:
					popt, pcov = curve_fit(self.func, wavefit, fluxfit, 
											sigma=errfit, p0=self.p0, bounds=self.plim)
					perr = np.sqrt(np.diag(pcov))
					chi2 = np.sum((self.func(wavefit, *popt) - fluxfit)**2/errfit**2)
					chi2_nu = chi2/dof
					self.poptmap[:, y, x] = popt
					self.perrmap[:, y, x] = perr
					self.chi2map[y, x] = chi2_nu
				except:
					self.poptmap[:, y, x] = np.nan
					self.perrmap[:, y, x] = np.nan
					self.chi2map[y, x] = np.nan
				# popt, pcov = curve_fit(self.func, wavefit, fluxfit, 
				# 						sigma=errfit, p0=self.p0, bounds=self.plim)
				# perr = np.sqrt(np.diag(pcov))
				# self.poptmap[:, y, x] = popt
				# self.perrmap[:, y, x] = perr
				# print(popt)
			
		t1 = time.time()
		print('Fitting finished in ', t1-t0, 's; ', (t1-t0)/60., 'mins')

	def savefitting(self, poptfile, perrfile, chi2file):
		fits.writeto(poptfile, self.poptmap, overwrite=True)
		fits.writeto(perrfile, self.perrmap, overwrite=True)
		fits.writeto(chi2file, self.chi2map, overwrite=True)








