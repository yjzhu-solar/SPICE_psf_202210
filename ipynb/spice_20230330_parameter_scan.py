import numpy as np
import matplotlib.pyplot as plt
from sunraster.instr.spice import read_spice_l2_fits
import h5py
import sunpy 
import sunpy.map
from correct_2d_psf import get_fwd_matrices, correct_spice_raster
from util import bindown, as_dict, get_iris_data, masked_median_filter, get_mask_errs
from fit_spice_lines import get_overall_center, fit_spice_lines as fsl
import astropy
from astropy.visualization import (ImageNormalize, AsinhStretch)
from astropy import constants as const
import juanfit
import importlib
importlib.reload(juanfit)
from juanfit import SpectrumFit2D

from copy import deepcopy
import os

spice_raster = read_spice_l2_fits("../src/solo_L2_spice-n-ras_20230330T104824_V03_18454953.fits")
spice_NeVIII_770_window = spice_raster["Ne VIII 770 - Peak"]

spice_dat = deepcopy(spice_NeVIII_770_window.data[0])
specmin = np.nanmin(spice_NeVIII_770_window.data[0], axis=2)
spice_dat = spice_dat - specmin[:,:,np.newaxis]
spice_dat = spice_dat - np.nanmedian(spice_dat[np.r_[0:8,18:],:,:], axis=0)[np.newaxis,:,:]

spice_hdr = spice_NeVIII_770_window.meta.original_header
# This exponent sets the non-gaussianity of the PSF core, 1 = Gaussian
# It also also changes its width somewhat, which is not ideal...
yl_core_xpo = 1.5

# Rotation angle of the PSF, both core and wings
psf_yl_angle = -15*np.pi/180

# FWHMs of PSF core. First argument is width along y axis before rotation,
# and is in arcseconds. Second is along lambda axis and is in angstrom.
fwhm_core0_yl = np.array([2, 0.95])

# This descriptor for plots should be manually edited to reflect the PSF parameters
gaussian_desc = '2-part Gaussian PSF'

fwhm_wing0_yl = np.array([10.0, 2.5]) # FWHMs of PSF wings in arcseconds and angstroms, respectively
desc_str='; standard wing aspect ratio'

# Fraction of overall PSF amplitude in wings (core weight is 1.0 - wing_weight).
# PSFs have unit peak amplitude, -- PLEASE NOTE: they do not integrate to 1.
wing_weight = 0.2

rebin_facs = [1,4,1]
# rebin_facs = [1,12,3]

spicedat_bindown = bindown(spice_dat[:,120:700,:],np.round(np.array(spice_dat[:,120:700,:].shape)/rebin_facs).astype(np.int32))
spice_dat1 = deepcopy(spicedat_bindown).transpose([2,1,0]).astype(np.float32)

fwhm_core0_yl_facs = np.linspace(0.5, 1.5, 5)
fwhm_wing0_yl_facs = np.linspace(0.5, 1.5, 5)

for fwhm_core0_yl_fac_ in fwhm_core0_yl_facs:
    for fwhm_wing0_yl_fac_ in fwhm_wing0_yl_facs:
        spice_corr_dat, spice_corr_chi2s, metadict = correct_spice_raster(spice_dat1, spice_hdr, fwhm_core0_yl*fwhm_core0_yl_fac_,
                                                                           fwhm_wing0_yl*fwhm_wing0_yl_fac_,psf_yl_angle, wing_weight,
                                                                           yl_core_xpo=yl_core_xpo,super_fac=1, psf_thold_core=0.0005, spice_bin_facs=rebin_facs)

        with h5py.File(os.path.join("../sav/NeVIII_20230330/", 
            "spice_1024_parameter_scan_core_large_bin_{:.2f}_{:.2f}_wing_{:.2f}_{:.2f}.h5".format(*(fwhm_core0_yl*fwhm_core0_yl_fac_),
                                                                                        *(fwhm_wing0_yl*fwhm_wing0_yl_fac_))), "w") as f:
            f.create_dataset("spice_corr_dat", data=spice_corr_dat)
            f.create_dataset("spice_corr_chi2s", data=spice_corr_chi2s)
            for k in metadict:
                f.attrs[k] = metadict[k]



