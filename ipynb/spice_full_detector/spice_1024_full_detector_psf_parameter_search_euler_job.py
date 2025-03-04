import numpy as np
import matplotlib.pyplot as plt
from sunraster.instr.spice import read_spice_l2_fits
import h5py
import sunpy 
import sunpy.map
from sharpesst.correct_2d_psf import get_fwd_matrices, correct_spice_raster
from sharpesst.util import bindown, as_dict, get_iris_data, masked_median_filter, get_mask_errs
from sharpesst.fit_spice_lines import get_overall_center, fit_spice_lines as fsl
import astropy
from astropy.visualization import (ImageNormalize, AsinhStretch)
from astropy import constants as const
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import juanfit
import importlib
importlib.reload(juanfit)
from juanfit import SpectrumFit2D, gaussian, SpectrumFitSingle, SpectrumFitRow
from scipy.signal import find_peaks
from sospice import spice_error
from scipy.optimize import curve_fit
import os
import itertools
from concurrent.futures import ProcessPoolExecutor
import logging

from copy import deepcopy
from glob import glob

# os.environ['OPENBLAS_NUM_THREADS'] = '1'

def gaussian_with_bg(x, wvl, int_total, width, bg):
    return gaussian(x, wvl, int_total, width) + bg

def work(args):
    (spice_dat1, spice_hdr, 
    fwhm_core0_yl, fwhm_wing0_yl, psf_yl_angle, wing_weight, 
    yl_core_xpo, rebin_facs, spice_orig_fit_int,
    spice_orig_fit_wvl, spice_orig_fit_wid,
    spice_la,)= args

    spice_corr_dat, spice_corr_chi2s, metadict = correct_spice_raster(spice_dat1, spice_hdr, fwhm_core0_yl, fwhm_wing0_yl, 
                                                                    psf_yl_angle, wing_weight, super_fac=1, chi2_th = 0.5,
                                                                    psf_thold_core=0.0005, spice_bin_facs = rebin_facs)
    spice_corr_dat = spice_corr_dat.transpose([2,1,0])

    NeVIII_row_20_corr_fit_wvl = np.zeros(25)
    NeVIII_row_20_corr_fit_int = np.zeros(25)
    NeVIII_row_20_corr_fit_wid = np.zeros(25)

    for ii in range(175,200):
        popt, pcov = curve_fit(gaussian_with_bg,
                            spice_la[np.r_[3:17,27:42]],
                            spice_corr_dat[np.r_[3:17,27:42], ii, 0],
                            p0=[spice_la[np.nanargmax(spice_corr_dat[:,ii,0])],
                            np.nanmax(spice_corr_dat[:,ii,0]),
                            0.7, np.nanmean(spice_corr_dat[3:15,ii,0])])

        NeVIII_row_20_corr_fit_int[ii-175] = popt[1]
        NeVIII_row_20_corr_fit_wvl[ii-175] = popt[0]
        NeVIII_row_20_corr_fit_wid[ii-175] = popt[2]
    
    logging.info(f"Processed: {args[2:7]}")
    

    if True: #np.nanstd(NeVIII_row_20_corr_fit_wvl[5:]) < 1*np.nanstd(spice_orig_fit_wvl[5:]):
        fig = plt.figure(figsize=(8,6),layout="constrained")
        axd = fig.subplot_mosaic(
            """
            AD
            BE
            CF
            CG
            """
        )

        axd["A"].imshow(spice_dat1[0,:,:].T, origin="lower",
                norm=ImageNormalize(vmin=np.nanpercentile(spice_dat1[0,:,:], 3),
                                    vmax=np.nanpercentile(spice_dat1[0,:,:], 99.95),
                                    stretch=AsinhStretch(0.5)), aspect=1)
        axd["B"].imshow(spice_corr_dat[:,:,0], origin="lower",
                norm=ImageNormalize(vmin=np.nanpercentile(spice_corr_dat[:,:,0], 3),
                                    vmax=np.nanpercentile(spice_corr_dat[:,:,0], 99.95),
                                    stretch=AsinhStretch(0.5)), aspect=0.5)

        axd["C"].step(spice_la, spice_dat1[0,363//2,:], where="mid")
        axd["C"].step(spice_la, spice_corr_dat[:,363//2,0], where="mid")

        axd["D"].plot(spice_orig_fit_int)
        axd["D"].plot(NeVIII_row_20_corr_fit_int)
        axd["E"].plot(spice_orig_fit_wvl)
        axd["E"].plot(NeVIII_row_20_corr_fit_wvl)
        axd["F"].plot(spice_orig_fit_wid)
        axd["F"].plot(NeVIII_row_20_corr_fit_wid)

        axd["D"].set_ylim(0,5)
        axd["E"].set_ylim(770.0, 770.4)
        axd["F"].set_ylim(0.3,1)

        axd["G"].axis("off")
        axd["G"].text(0.2, 0.7, f"sigma_wvl_orig: {np.nanstd(spice_orig_fit_wvl[5:]):.4f}")
        axd["G"].text(0.2, 0.3, f"sigma_wvl_corr: {np.nanstd(NeVIII_row_20_corr_fit_wvl[5:]):.4f}")

        fig.savefig(os.path.join("/cluster/home/zhuyin/work/spice_psf/figs_full_detector_parameter_search_1024/",
        f"spice_psf_{fwhm_core0_yl[0]:.2f}_{fwhm_core0_yl[1]:.2f}_{fwhm_wing0_yl[0]:.2f}_{fwhm_wing0_yl[1]:.2f}_{psf_yl_angle:.2f}_{wing_weight:.2f}_{yl_core_xpo:.2f}.png"), dpi=300)


def main(test=False):
    raster_files = sorted(glob("/cluster/home/zhuyin/Solar/SPICE_psf_202210/src/full_detector/level2/2022/10/24/*.fits"))

    test_raster_cube = read_spice_l2_fits(raster_files[0])
    sw_data = np.ones((test_raster_cube['Full SW 4:1 Focal Lossy'].data.shape[1],
                   test_raster_cube['Full SW 4:1 Focal Lossy'].data.shape[2],
                   len(raster_files)))*np.nan

    lw_data = np.ones((test_raster_cube['Full LW 4:1 Focal Lossy'].data.shape[1],
                        test_raster_cube['Full LW 4:1 Focal Lossy'].data.shape[2],
                        len(raster_files)))*np.nan

    sw_wcs_list = []
    lw_wcs_list = []
    sw_meta_list = []
    lw_meta_list = []
    sw_wvl_list = []
    lw_wvl_list = []

    for ii, file_ in enumerate(raster_files):
        datacube_ = read_spice_l2_fits(file_)
        sw_data[:,:,ii] = datacube_['Full SW 4:1 Focal Lossy'].data[0,:,:,0]
        lw_data[:,:,ii] = datacube_['Full LW 4:1 Focal Lossy'].data[0,:,:,0]
        sw_wcs_list.append(datacube_['Full SW 4:1 Focal Lossy'].wcs)
        lw_wcs_list.append(datacube_['Full LW 4:1 Focal Lossy'].wcs)
        sw_meta_list.append(datacube_['Full SW 4:1 Focal Lossy'].meta)
        lw_meta_list.append(datacube_['Full LW 4:1 Focal Lossy'].meta)
        sw_wvl_list.append(datacube_['Full SW 4:1 Focal Lossy'].spectral_axis.to_value(u.AA))
        lw_wvl_list.append(datacube_['Full LW 4:1 Focal Lossy'].spectral_axis.to_value(u.AA))
    sw_data = np.flip(sw_data, axis=-1)
    lw_data = np.flip(lw_data, axis=-1)

    test_fit_NeVIII_wvl = sw_wvl_list[0][735:799]

    with fits.open(raster_files[-21]) as hdul:
        spice_hdr_20 = hdul[0].header.copy()
    
    spice_hdr_20_NeVIII = deepcopy(spice_hdr_20)
    spice_hdr_20_NeVIII_new_naxis3 = 799 - 735
    spice_hdr_20_NeVIII_new_crpix3 = (1 + 799 - 735)/2
    spice_hdr_20_NeVIII_new_crval3 =  ((736 + 799)/2 - spice_hdr_20_NeVIII["CRPIX3"])*spice_hdr_20_NeVIII["CDELT3"] + spice_hdr_20_NeVIII["CRVAL3"]
    spice_hdr_20_NeVIII["NAXIS3"] = spice_hdr_20_NeVIII_new_naxis3 
    spice_hdr_20_NeVIII["CRPIX3"] = spice_hdr_20_NeVIII_new_crpix3
    spice_hdr_20_NeVIII["CRVAL3"] = spice_hdr_20_NeVIII_new_crval3
    
    test_NeVIII_data = sw_data[735:799,210:790,:]
    test_NeVIII_data_bg_rm = test_NeVIII_data - np.nanmean(test_NeVIII_data, axis=2)[:,:,np.newaxis]

    rebin_facs = [1,2,1]

    test_NeVIII_data_bin = bindown(test_NeVIII_data_bg_rm,np.round(np.array(test_NeVIII_data_bg_rm.shape)/rebin_facs).astype(np.int32))
    test_NeVIII_data_bin_trans = deepcopy(test_NeVIII_data_bin[:,:,20:22]).transpose([2,1,0]).astype(np.float64)

    NeVIII_row_20_orig_fit_wvl = np.zeros(25)
    NeVIII_row_20_orig_fit_int = np.zeros(25)
    NeVIII_row_20_orig_fit_wid = np.zeros(25)

    for ii in range(175,200):
        popt, pcov = curve_fit(gaussian_with_bg,
                            test_fit_NeVIII_wvl[np.r_[3:17,27:42]],
                            test_NeVIII_data_bin[np.r_[3:17,27:42], ii, 20],
                            p0=[test_fit_NeVIII_wvl[np.nanargmax(test_NeVIII_data_bin[:,ii,20])],
                            np.nanmax(test_NeVIII_data_bin[:,ii,20]),
                            0.7, np.nanmean(test_NeVIII_data_bin[3:15,ii,20])])
        
        NeVIII_row_20_orig_fit_int[ii-175] = popt[1]
        NeVIII_row_20_orig_fit_wvl[ii-175] = popt[0]
        NeVIII_row_20_orig_fit_wid[ii-175] = popt[2]

    

    if test:
        # parameter set 1
        yl_core_xpo = 1.0

        # Rotation angle of the PSF, both core and wings
        psf_yl_angle = -20*np.pi/180

        # FWHMs of PSF core. First argument is width along y axis before rotation,
        # and is in arcseconds. Second is along lambda axis and is in angstrom.
        fwhm_core0_yl = np.array([5.5, 1.15])

        # This descriptor for plots should be manually edited to reflect the PSF parameters
        gaussian_desc = '2-part Gaussian PSF'

        fwhm_wing0_yl = np.array([20.0, 4]) # FWHMs of PSF wings in arcseconds and angstroms, respectively
        desc_str='; standard wing aspect ratio'

        # Fraction of overall PSF amplitude in wings (core weight is 1.0 - wing_weight).
        # PSFs have unit peak amplitude, -- PLEASE NOTE: they do not integrate to 1.
        wing_weight = 0.21

        work((test_NeVIII_data_bin_trans, spice_hdr_20_NeVIII,
        fwhm_core0_yl, fwhm_wing0_yl, psf_yl_angle, wing_weight, 
        yl_core_xpo, rebin_facs,NeVIII_row_20_orig_fit_int,
        NeVIII_row_20_orig_fit_wvl, NeVIII_row_20_orig_fit_wid,
        test_fit_NeVIII_wvl))
    
    else:
        ncpus = 10

        yl_core_xpo_list = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        psf_yl_angle_list = np.deg2rad(np.linspace(-15,-25,5))
        fwhm_core0_yl_0_all = np.linspace(4.5,6.5,5)
        fwhm_core0_yl_1_all = np.linspace(1.05, 1.35,7)
        fwhm_wing0_0_all = np.linspace(10,30,5)
        fwhm_wing0_1_all = np.linspace(2,6,9)
        wing_weight_list = np.linspace(0.11, 0.31, 5)

        arg_array = []

        # use itertools to avoid ugly and inefficient nested loops
        for combination in itertools.product(
                fwhm_core0_yl_0_all, fwhm_core0_yl_1_all, fwhm_wing0_0_all,
                fwhm_wing0_1_all,  psf_yl_angle_list, wing_weight_list, yl_core_xpo_list):
            
            combination = np.array(combination)

            arg_array.append([test_NeVIII_data_bin_trans, spice_hdr_20_NeVIII,
            combination[:2], combination[2:4], *combination[4:], 
            rebin_facs,NeVIII_row_20_orig_fit_int,
            NeVIII_row_20_orig_fit_wvl, NeVIII_row_20_orig_fit_wid,
            test_fit_NeVIII_wvl])
        
        with ProcessPoolExecutor(max_workers=ncpus) as executor:
            rs=executor.map(work, arg_array[:400], chunksize=1000)

        # work(arg_array[10])

    
if __name__ == '__main__':
    main(test=False)
