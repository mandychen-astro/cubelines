class Fitting():
	def __init__(self, wavebounds):
	    self.wavebounds = wavebounds 

    def get_wavefit(self, x):
        # return the valid wave values for fitting
        wavefit = []
        nbounds = len(self.wavebounds)
        for i in range(nbounds):
            wavefit += x[(x > wavebounds[i][0]) & (x < wavebounds[i][1])]
        return wavefit

    def dofit(self):
    	