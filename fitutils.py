import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from astropy.io import fits
import time
from mpdaf.obj import Cube
import os

def get_lsf(wave):
    l0, r0 = np.loadtxt(os.environ['HOME']+'/CUBS/muse_lsf.dat',unpack=True)
    r = interp1d(l0, r0)(wave)
    lsf = 2.998e5/r
    return lsf

def convolve_lsf(sig, lsf):
    a = 2*np.sqrt(2*np.log(2))
    return np.sqrt(sig**2+(lsf/a)**2)

def wave2vel(line,z,wave):
    vel = (wave-line*(1+z))/(line*(1+z))*2.998e5
    return vel

def gauss(x, mu, sig, n):
    return n*np.exp(-(x-mu)**2/(2*sig**2))

def interpWindow(line, z, dv, ddv):
    # generate interpolation window, dv in km/s
    # for lines in the line list redshifted to z
    # output blue window [wv1, wv2] ([-2dv, -dv])
    # output red window [wv3, wv4] ([+dv, +2dv])
    redline = line*(1+z)
    dlam = dv/2.998e5*redline
    ddlam = (ddv+dv)/2.998e5*redline
    wv1, wv2 = redline-ddlam, redline-dlam
    wv3, wv4 = redline+dlam, redline+ddlam
    return wv1, wv2, wv3, wv4

class FitFunc():
	def __init__(self, line, lsf, wavebounds):
		self.line = line
		self.lsf = lsf
		self.wavemin1, self.wavemax1 = wavebounds[0], wavebounds[1]
		self.wavemin2, self.wavemax2 = wavebounds[2], wavebounds[3]

	def err_prop_term_o2(self, x, z, sig, n1, n21):
		n2 = n1*n21
		g1 = gauss(x, self.line[0]*(1+z), convolve_lsf(sig, self.lsf[0])/2.998e5*self.line[0]*(1+z), n1)
		g2 = gauss(x, self.line[1]*(1+z), convolve_lsf(sig, self.lsf[1])/2.998e5*self.line[1]*(1+z), n2)
		term1 = -(x - self.line[0]*(1+z))/(convolve_lsf(sig, self.lsf[0])/2.998e5*self.line[0]*(1+z))*(g1+g2)
		return term1

	def err_prop_term_o2_o3(self, x, z, sig, n1, n21, n3):
		n2 = n1*n21
		n4 = n3*3.
		g1 = gauss(x, self.line[0]*(1+z), convolve_lsf(sig, self.lsf[0])/2.998e5*self.line[0]*(1+z), n1)
		g2 = gauss(x, self.line[1]*(1+z), convolve_lsf(sig, self.lsf[1])/2.998e5*self.line[1]*(1+z), n2)
		g3 = gauss(x, self.line[2]*(1+z), convolve_lsf(sig, self.lsf[2])/2.998e5*self.line[2]*(1+z), n3)
		g4 = gauss(x, self.line[3]*(1+z), convolve_lsf(sig, self.lsf[3])/2.998e5*self.line[3]*(1+z), n4)
		term1 = -(x - self.line[0]*(1+z))/(convolve_lsf(sig, self.lsf[0])/2.998e5*self.line[0]*(1+z))*(g1+g2)
		term2 = -(x - self.line[2]*(1+z))/(convolve_lsf(sig, self.lsf[2])/2.998e5*self.line[2]*(1+z))*g3
		term3 = -(x - self.line[3]*(1+z))/(convolve_lsf(sig, self.lsf[3])/2.998e5*self.line[3]*(1+z))*g4
		return term1+term2+term3

	def gauss_o2(self, x, z, sig, n1, n21):
		n2 = n1*n21
		g1 = gauss(x, self.line[0]*(1+z), convolve_lsf(sig, self.lsf[0])/2.998e5*self.line[0]*(1+z), n1)
		g2 = gauss(x, self.line[1]*(1+z), convolve_lsf(sig, self.lsf[1])/2.998e5*self.line[1]*(1+z), n2)
		return g1+g2

	def gauss_o2_display(self, x, z, sig, n1, n21):
		n2 = n1*n21
		g1 = gauss(x, self.line[0]*(1+z), convolve_lsf(sig, self.lsf[0])/2.998e5*self.line[0]*(1+z), n1)
		g2 = gauss(x, self.line[1]*(1+z), convolve_lsf(sig, self.lsf[1])/2.998e5*self.line[1]*(1+z), n2)
		return g1+g2, g1, g2

	def gauss_o2_w_cont(self, x, z, sig, n1, n21, a1, b1):
		n2 = n1*n21
		mask1 = (x>self.wavemin1) & (x<self.wavemax1)
		x1 = x[mask1]
		cont1 = a1 + x1*b1
		cont = np.zeros(len(x))
		cont[mask1] = cont1
		g1 = gauss(x, self.line[0]*(1+z), convolve_lsf(sig, self.lsf[0])/2.998e5*self.line[0]*(1+z), n1)
		g2 = gauss(x, self.line[1]*(1+z), convolve_lsf(sig, self.lsf[1])/2.998e5*self.line[1]*(1+z), n2)
		return g1+g2+cont

	def gauss_o2_2comp_w_cont(self, x, z1, sig1, n1, n21, z21, sig2, n5, n65, a1, b1):
		n2 = n1*n21
		n6 = n5*n65
		z2 = z1+z21
		mask1 = (x>self.wavemin1) & (x<self.wavemax1)
		x1 = x[mask1]
		cont1 = a1 + x1*b1
		cont = np.zeros(len(x))
		cont[mask1] = cont1
		g1 = gauss(x, self.line[0]*(1+z1), convolve_lsf(sig1, self.lsf[0])/2.998e5*self.line[0]*(1+z1), n1)
		g2 = gauss(x, self.line[1]*(1+z1), convolve_lsf(sig1, self.lsf[1])/2.998e5*self.line[1]*(1+z1), n2)
		g5 = gauss(x, self.line[0]*(1+z2), convolve_lsf(sig2, self.lsf[0])/2.998e5*self.line[0]*(1+z2), n5)
		g6 = gauss(x, self.line[1]*(1+z2), convolve_lsf(sig2, self.lsf[1])/2.998e5*self.line[1]*(1+z2), n6)

		return g1+g2+g5+g6+cont

	def gauss_o3(self, x, z, sig, n3):
		n4 = n3*3.
		g3 = gauss(x, self.line[2]*(1+z), convolve_lsf(sig, self.lsf[2])/2.998e5*self.line[2]*(1+z), n3)
		g4 = gauss(x, self.line[3]*(1+z), convolve_lsf(sig, self.lsf[3])/2.998e5*self.line[3]*(1+z), n4)
		return g3+g4

	def gauss_o3_red(self, x, z, sig, n4):
		g4 = gauss(x, self.line[3]*(1+z), convolve_lsf(sig, self.lsf[3])/2.998e5*self.line[3]*(1+z), n4)		
		return g4

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
		g1 = gauss(x, self.line[0]*(1+z), convolve_lsf(sig, self.lsf[0])/2.998e5*self.line[0]*(1+z), n1)
		g2 = gauss(x, self.line[1]*(1+z), convolve_lsf(sig, self.lsf[1])/2.998e5*self.line[1]*(1+z), n2)
		g3 = gauss(x, self.line[2]*(1+z), convolve_lsf(sig, self.lsf[2])/2.998e5*self.line[2]*(1+z), n3)
		g4 = gauss(x, self.line[3]*(1+z), convolve_lsf(sig, self.lsf[3])/2.998e5*self.line[3]*(1+z), n4)
		return g1+g2+g3+g4+cont

	def gauss_o2_o3_2comp_w_cont(self, x, z1, sig1, n1, n21, n3, z21, sig2, n5, n65, n7, a1, b1, a2, b2):
		n2 = n1*n21
		n4 = n3*3.
		n6 = n5*n65
		n8 = n7*3.
		z2 = z1+z21
		mask1 = (x>self.wavemin1) & (x<self.wavemax1)
		mask2 = (x>self.wavemin2) & (x<self.wavemax2)
		x1, x2 = x[mask1], x[mask2]
		cont1 = a1 + x1*b1
		cont2 = a2 + x2*b2
		cont = np.zeros(len(x))
		cont[mask1] = cont1
		cont[mask2] = cont2
		g1 = gauss(x, self.line[0]*(1+z1), convolve_lsf(sig1, self.lsf[0])/2.998e5*self.line[0]*(1+z1), n1)
		g2 = gauss(x, self.line[1]*(1+z1), convolve_lsf(sig1, self.lsf[1])/2.998e5*self.line[1]*(1+z1), n2)
		g3 = gauss(x, self.line[2]*(1+z1), convolve_lsf(sig1, self.lsf[2])/2.998e5*self.line[2]*(1+z1), n3)
		g4 = gauss(x, self.line[3]*(1+z1), convolve_lsf(sig1, self.lsf[3])/2.998e5*self.line[3]*(1+z1), n4)
		g5 = gauss(x, self.line[0]*(1+z2), convolve_lsf(sig2, self.lsf[0])/2.998e5*self.line[0]*(1+z2), n5)
		g6 = gauss(x, self.line[1]*(1+z2), convolve_lsf(sig2, self.lsf[1])/2.998e5*self.line[1]*(1+z2), n6)
		g7 = gauss(x, self.line[2]*(1+z2), convolve_lsf(sig2, self.lsf[2])/2.998e5*self.line[2]*(1+z2), n7)
		g8 = gauss(x, self.line[3]*(1+z2), convolve_lsf(sig2, self.lsf[3])/2.998e5*self.line[3]*(1+z2), n8)

		return g1+g2+g3+g4+g5+g6+g7+g8+cont

	def gauss_o2_o3_3comp_w_cont(self, x, z1, sig1, n1, n21, n3, 
										z21, sig2, n5, n65, n7,
										z31, sig3, n9, n109, n11, 
										a1, b1, a2, b2):
		n2 = n1*n21
		n4 = n3*3.
		n6 = n5*n65
		n8 = n7*3.
		n10 = n9*n109
		n12 = n11*3.

		z2 = z1+z21
		z3 = z1+z31

		mask1 = (x>self.wavemin1) & (x<self.wavemax1)
		mask2 = (x>self.wavemin2) & (x<self.wavemax2)
		x1, x2 = x[mask1], x[mask2]
		cont1 = a1 + x1*b1
		cont2 = a2 + x2*b2
		cont = np.zeros(len(x))
		cont[mask1] = cont1
		cont[mask2] = cont2
		g1 = gauss(x, self.line[0]*(1+z1), convolve_lsf(sig1, self.lsf[0])/2.998e5*self.line[0]*(1+z1), n1)
		g2 = gauss(x, self.line[1]*(1+z1), convolve_lsf(sig1, self.lsf[1])/2.998e5*self.line[1]*(1+z1), n2)
		g3 = gauss(x, self.line[2]*(1+z1), convolve_lsf(sig1, self.lsf[2])/2.998e5*self.line[2]*(1+z1), n3)
		g4 = gauss(x, self.line[3]*(1+z1), convolve_lsf(sig1, self.lsf[3])/2.998e5*self.line[3]*(1+z1), n4)
		g5 = gauss(x, self.line[0]*(1+z2), convolve_lsf(sig2, self.lsf[0])/2.998e5*self.line[0]*(1+z2), n5)
		g6 = gauss(x, self.line[1]*(1+z2), convolve_lsf(sig2, self.lsf[1])/2.998e5*self.line[1]*(1+z2), n6)
		g7 = gauss(x, self.line[2]*(1+z2), convolve_lsf(sig2, self.lsf[2])/2.998e5*self.line[2]*(1+z2), n7)
		g8 = gauss(x, self.line[3]*(1+z2), convolve_lsf(sig2, self.lsf[3])/2.998e5*self.line[3]*(1+z2), n8)
		g9 = gauss(x, self.line[0]*(1+z3), convolve_lsf(sig3, self.lsf[0])/2.998e5*self.line[0]*(1+z3), n9)
		g10 = gauss(x, self.line[1]*(1+z3), convolve_lsf(sig3, self.lsf[1])/2.998e5*self.line[1]*(1+z3), n10)
		g11 = gauss(x, self.line[2]*(1+z3), convolve_lsf(sig3, self.lsf[2])/2.998e5*self.line[2]*(1+z3), n11)
		g12 = gauss(x, self.line[3]*(1+z3), convolve_lsf(sig3, self.lsf[3])/2.998e5*self.line[3]*(1+z3), n12)

		return g1+g2+g3+g4+g5+g6+g7+g8+g9+g10+g11+g12+cont

	def gauss_o2_o3(self, x, z, sig, n1, n21, n3):
		n2 = n1*n21
		n4 = n3*3.
		g1 = gauss(x, self.line[0]*(1+z), convolve_lsf(sig, self.lsf[0])/2.998e5*self.line[0]*(1+z), n1)
		g2 = gauss(x, self.line[1]*(1+z), convolve_lsf(sig, self.lsf[1])/2.998e5*self.line[1]*(1+z), n2)
		g3 = gauss(x, self.line[2]*(1+z), convolve_lsf(sig, self.lsf[2])/2.998e5*self.line[2]*(1+z), n3)
		g4 = gauss(x, self.line[3]*(1+z), convolve_lsf(sig, self.lsf[3])/2.998e5*self.line[3]*(1+z), n4)
		return g1+g2+g3+g4

	def gauss_o2_o3_2comp(self, x, z1, sig1, n1, n21, n3, z21, sig2, n5, n65, n7):
		n2 = n1*n21
		n4 = n3*3.
		n6 = n5*n65
		n8 = n7*3.
		z2 = z1+z21
		g1 = gauss(x, self.line[0]*(1+z1), convolve_lsf(sig1, self.lsf[0])/2.998e5*self.line[0]*(1+z1), n1)
		g2 = gauss(x, self.line[1]*(1+z1), convolve_lsf(sig1, self.lsf[1])/2.998e5*self.line[1]*(1+z1), n2)
		g3 = gauss(x, self.line[2]*(1+z1), convolve_lsf(sig1, self.lsf[2])/2.998e5*self.line[2]*(1+z1), n3)
		g4 = gauss(x, self.line[3]*(1+z1), convolve_lsf(sig1, self.lsf[3])/2.998e5*self.line[3]*(1+z1), n4)
		g5 = gauss(x, self.line[0]*(1+z2), convolve_lsf(sig2, self.lsf[0])/2.998e5*self.line[0]*(1+z2), n5)
		g6 = gauss(x, self.line[1]*(1+z2), convolve_lsf(sig2, self.lsf[1])/2.998e5*self.line[1]*(1+z2), n6)
		g7 = gauss(x, self.line[2]*(1+z2), convolve_lsf(sig2, self.lsf[2])/2.998e5*self.line[2]*(1+z2), n7)
		g8 = gauss(x, self.line[3]*(1+z2), convolve_lsf(sig2, self.lsf[3])/2.998e5*self.line[3]*(1+z2), n8)

		return g1+g2+g3+g4+g5+g6+g7+g8

class Fitting():
	def __init__(self, cube, wavebounds, func, p0, plim, xlim, ylim, prev_p0 = None):
		self.wv = cube.wave
		self.wave = cube.wave.coord()
		self.data = cube.data.data
		self.err = np.sqrt(cube.var.data)
		self.wcs = cube.wcs
		self.dh = cube.data_header
		self.wavebounds = wavebounds
		self.p0 = p0
		self.plim = plim
		self.xlim = xlim
		self.ylim = ylim
		self.func = func
		self.nx, self.ny = self.data.shape[2], self.data.shape[1]	
		self.update_p0 = False
		if prev_p0 is not None: 
			self.update_p0 = True
			self.prev_p0 = prev_p0
			print('Previous fitting result is used to guide current fitting')

	def get_wavemask(self):
		wavemin1, wavemax1 = self.wavebounds[0], self.wavebounds[1]
		wavemin2, wavemax2 = self.wavebounds[2], self.wavebounds[3]
		mask1 = (self.wave>wavemin1) & (self.wave<wavemax1)
		mask2 = (self.wave>wavemin2) & (self.wave<wavemax2)
		return mask1, mask2

	def cont_sub(self, line, z, dv, ddv, savefile = None):
		wv1, wv2, wv3, wv4 = interpWindow(line, z, dv, ddv)
		mask_b = (self.wave>=wv1)&(self.wave<=wv2)
		mask_r = (self.wave>=wv3)&(self.wave<=wv4)
		mask_mid = (self.wave>=(wv1-200))&(self.wave<=(wv4+200))
		wavefit = np.concatenate((self.wave[mask_b], self.wave[mask_r]))
		for y in range(0, self.ny):
			for x in range(0, self.nx):
				fluxfit = np.concatenate((self.data[mask_b,y,x], self.data[mask_r,y,x]))
				z = np.polyfit(wavefit, fluxfit, 1)
				p = np.poly1d(z)
				linemodel = p(self.wave[mask_mid])
				self.data[mask_mid, y, x] -=  linemodel
		if savefile is not None:
			cubenew = Cube(data = self.data, var = (self.err)**2, data_header=self.dh, 
				wcs=self.wcs, wave=self.wv)
			cubenew.write(savefile, savemask = 'none')

	def dofit(self, prev_p0 = None):	
		mask1, mask2 = self.get_wavemask()
		wavefit = np.concatenate((self.wave[mask1], self.wave[mask2]))
		self.poptmap = np.zeros(shape=(len(self.p0), self.ny, self.nx))
		self.perrmap = np.zeros(shape=(len(self.p0), self.ny, self.nx))
		self.poptmap[:,:,:], self.perrmap[:,:,:] = np.nan, np.nan

		t0 = time.time()
		for y in range(self.ylim[0], self.ylim[1]):
			for x in range(self.xlim[0], self.xlim[1]):
				dataspec = self.data[:, y, x]
				errspec = self.err[:, y, x]

				fluxfit = np.concatenate((dataspec[mask1], dataspec[mask2]))
				errfit = np.concatenate((errspec[mask1], errspec[mask2]))

				if self.update_p0:
					p00 = self.prev_p0[:, y, x]
					if p00[0] is not np.nan:
						self.p0[0] = np.abs(np.random.normal(p00[0], 0.0002))
						self.p0[1] = np.abs(np.random.normal(p00[1], 30))
						# self.p0[2] = np.abs(np.random.normal(p00[2], 30))
						# self.p0[3] = np.abs(np.random.normal(p00[3], 0.2))
						# self.p0[4] = np.abs(np.random.normal(p00[4], 5))

				try:
					popt, pcov = curve_fit(self.func, wavefit, fluxfit, 
											sigma=errfit, p0=self.p0, bounds=self.plim)
					perr = np.sqrt(np.diag(pcov))
					# if len(self.p0)>5:
					# popt[4] = popt[0] + popt[4]
					# perr[4] = np.sqrt(perr[0]**2 + perr[4]**2)
					# 	print('this happened')
					self.poptmap[:, y, x] = popt
					self.perrmap[:, y, x] = perr
					# print(popt)
				except:
					self.poptmap[:, y, x] = np.nan
					self.perrmap[:, y, x] = np.nan
					# print('Fitting on [%d, %d] failed' % (x,y))

				# popt, pcov = curve_fit(self.func, wavefit, fluxfit, 
				# 						sigma=errfit, p0=self.p0, bounds=self.plim)
				# perr = np.sqrt(np.diag(pcov))
				# self.poptmap[:, y, x] = popt
				# self.perrmap[:, y, x] = perr
				# print(popt)
			
		t1 = time.time()
		print('Fitting finished in ', t1-t0, 's; ', (t1-t0)/60., 'mins')

	def savefitting(self, poptfile, perrfile):
		fits.writeto(poptfile, self.poptmap, overwrite=True)
		fits.writeto(perrfile, self.perrmap, overwrite=True)








