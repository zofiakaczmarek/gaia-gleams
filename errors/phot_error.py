import numpy as np

magGr = 12, 13, 13, 14, 16, 17, 17.5, 18, 18, 18.25, 18.25, 19, 20, 20.25
sigmaGr = 0.00041, 0.00055, 0.00055, 0.001, 0.003, 0.005, 0.0068, 0.0084, 0.0084, 0.01, 0.01, 0.02, 0.042, 0.057
reg = np.polyfit(magGr, np.log10(sigmaGr), 3)
p2 = np.poly1d(reg)

def phot_err_transit(G):
    return 10**p2(G)
    
def phot_err_CCD(G):
    return 3*10**p2(G)