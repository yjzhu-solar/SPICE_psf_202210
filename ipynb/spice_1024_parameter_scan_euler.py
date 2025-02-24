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
from copy import deepcopy
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def work(args):
    (fwhm_core0_yl_fac_, fwhm_wing0_yl_fac_, spice_dat1, spice_hdr, 
    fwhm_core0_yl, fwhm_wing0_yl, psf_yl_angle, wing_weight, 
    yl_core_xpo, rebin_facs, spice_err_fac,
    spice_la, spice_wlcen0, corr_sdev_guess, spice_fits,
    spice_origin_vel)= args

    # print(args)
    
    spice_corr_dat, spice_corr_chi2s, metadict = correct_spice_raster(spice_dat1, spice_hdr, fwhm_core0_yl*fwhm_core0_yl_fac_,
                                                                        fwhm_wing0_yl*fwhm_wing0_yl_fac_,psf_yl_angle, wing_weight,
                                                                        yl_core_xpo=yl_core_xpo,super_fac=1, psf_thold_core=0.0005, spice_bin_facs=rebin_facs)
    


    spice_corr_mask, spice_corr_err = get_mask_errs(spice_corr_dat, spice_err_fac)

    [det_origin0, det_dims0, det_scale0] = metadict["det_origin0"], metadict["det_dims0"], metadict["det_scale0"]

    spice_corr_fits = fsl(spice_corr_dat, spice_corr_err, spice_la, spice_corr_mask, spice_wlcen0, corr_sdev_guess)

    spice_corr_vel = (spice_corr_fits["centers"].T/spice_wlcen0 - 1)*const.c.to_value("km/s")
    spice_corr_vel = spice_corr_vel - np.nanmedian(spice_corr_vel[:,:], axis=0)[np.newaxis,:]

    fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(10,12),layout="constrained")

    int_norm = ImageNormalize(stretch=AsinhStretch(0.1))

    ax1.imshow(spice_fits["amplitudes"].T,origin="lower",norm=int_norm,aspect=1,
                cmap="sdoaia171")
    ax1.set_title("Original")

    ax2.imshow(spice_corr_fits["amplitudes"].T,origin="lower",norm=int_norm,aspect=1,
                cmap="sdoaia171")
    ax2.set_title("Corrected")

    ax3.imshow(spice_origin_vel,origin="lower",vmin=-40,vmax=40,aspect=1,
                cmap="coolwarm")

    ax4.imshow(spice_corr_vel,origin="lower",vmin=-40,vmax=40,aspect=1,
                cmap="coolwarm")

    # select_column_index = 93
    select_column_index = 63  #cut through the 1024 upflow region
    
    for ax_ in (ax1,ax2,ax3,ax4):
        ax_.axvline(select_column_index, ls='--', lw=0.5, color='grey', alpha=0.5)

    ln1, = ax5.plot((spice_fits["amplitudes"].T)[:,select_column_index], color='black', label='int')

    ax5_vel = ax5.twinx()
    ln2, = ax5_vel.plot(spice_origin_vel[:,select_column_index], color='blue', label='vlos')

    ax5_dint = ax5.twinx()
    ln3, = ax5_dint.plot(np.gradient((spice_fits["amplitudes"].T)[:,select_column_index]), color='red', label='dint')

    ax6.plot((spice_corr_fits["amplitudes"].T)[:,select_column_index], color='black', label='int')

    ax6_vel = ax6.twinx()
    ax6_vel.plot(spice_corr_vel[:,select_column_index], color='blue', label='vlos')

    ax6_dint = ax6.twinx()
    ax6_dint.plot(np.gradient((spice_corr_fits["amplitudes"].T)[:,select_column_index]), color='red', label='dint')

    ax5.legend([ln1, ln2, ln3], [ln1.get_label(), ln2.get_label(), ln3.get_label()],frameon=False)

    for ax_ in (ax5_vel, ax6_vel):
        ax_.set_ylim(-40,40)

    fig.savefig(os.path.join("/cluster/home/zhuyin/Solar/SPICE_psf_202210/sav/NeVIII_parameter_scan_fine_fig/",
                "spice_1024_parameter_scan_core_large_bin_{:.2f}_{:.2f}_wing_{:.2f}_{:.2f}.png".format(*(fwhm_core0_yl*fwhm_core0_yl_fac_),
                                                                                    *(fwhm_wing0_yl*fwhm_wing0_yl_fac_))),
                dpi=300, bbox_inches="tight")

    with h5py.File(os.path.join("/cluster/home/zhuyin/Solar/SPICE_psf_202210/sav/NeVIII_parameter_fine_scan/", 
        "spice_1024_parameter_scan_core_large_bin_{:.2f}_{:.2f}_wing_{:.2f}_{:.2f}.h5".format(*(fwhm_core0_yl*fwhm_core0_yl_fac_),
                                                                                    *(fwhm_wing0_yl*fwhm_wing0_yl_fac_))), "w") as f:
        f.create_dataset("spice_corr_dat", data=spice_corr_dat)
        f.create_dataset("spice_corr_chi2s", data=spice_corr_chi2s)
        for k in metadict:
            f.attrs[k] = metadict[k]

    print(f"Done: {args[:2]}")
    
    return None

def main():
    spice_raster = read_spice_l2_fits("../src/solo_L2_spice-n-ras_20221024T231535_V07_150995398-000.fits")
    
    # spice_raster = read_spice_l2_fits("../src/solo_L2_spice-n-ras_20221023T195035_V03_150995388-000.fits")
    # spice_raster = read_spice_l2_fits("../src/solo_L2_spice-n-ras_20221017T031211_V03_150995346-000.fits")

    spice_NeVIII_770_window = spice_raster["Ne VIII 770 - Peak"]
    # spice_NeVIII_770_window = spice_raster["Ne VIII 770 / Mg VIII 772 (Merged)"]

    spice_dat = deepcopy(spice_NeVIII_770_window.data[0])
    specmin = np.nanmin(spice_NeVIII_770_window.data[0], axis=2)
    # spice_dat = spice_dat - specmin[:,:,np.newaxis]
    spice_dat = spice_dat - np.nanmin(spice_dat[np.r_[12:18,33:40],:,:], axis=0)[np.newaxis,:,:]
    
    spice_hdr = spice_NeVIII_770_window.meta.original_header
    # This exponent sets the non-gaussianity of the PSF core, 1 = Gaussian
    # It also also changes its width somewhat, which is not ideal...
    yl_core_xpo = 1.5

    # Rotation angle of the PSF, both core and wings
    # psf_yl_angle = -35*np.pi/180

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

    # rebin_facs = [1,4,1]
    rebin_facs = np.array([1,2,1]) # less binning

    spicedat_bindown = bindown(spice_dat[:,120:700,:],np.round(np.array(spice_dat[:,120:700,:].shape)/rebin_facs).astype(np.int32))
    # spice_err_fac = np.nanstd(spicedat_bindown[:,5:30,180:192]) # for 1024 4-pixel bin
    spice_err_fac = np.nanstd(spicedat_bindown[:,10:60,180:192]) # for 1024 2-pixel bin
    # spice_err_fac = np.nanstd(spicedat_bindown[:,120:140,0:10]) # for 1023 4-pixel bin
    spice_dat1 = deepcopy(spicedat_bindown).transpose([2,1,0]).astype(np.float32)

    # fwhm_core0_yl_facs = np.linspace(0.5, 1.5, 11)
    # fwhm_wing0_yl_facs = np.linspace(0.5, 1.5, 11)

    fwhm_core0_yl_facs = np.linspace(0.75, 1.25, 3)
    fwhm_wing0_yl_facs = np.linspace(0.75, 1.25, 3)
    
    spice_sdev_guess = 0.1
    corr_sdev_guess = 0.05

    spice_la = spice_NeVIII_770_window.spectral_axis.to_value("Angstrom")
    spice_wlcen0 = spice_la[np.nanargmax(np.nanmean(spice_dat1[:,:,:], axis=(0,1)))]


    spice_mask, spice_err = get_mask_errs(spice_dat1.astype(np.float64), spice_err_fac)
    spice_fits = fsl(spice_dat1, spice_err, spice_la, spice_mask, spice_wlcen0, spice_sdev_guess)

    spice_origin_vel = (spice_fits["centers"].T/spice_wlcen0 - 1)*const.c.to_value("km/s")
    spice_origin_vel = spice_origin_vel - np.nanmedian(spice_origin_vel[:,:], axis=0)[np.newaxis,:]

    ncpus = 9
    arg_array = []
    for fwhm_core0_yl_fac_ in fwhm_core0_yl_facs:
        for fwhm_wing0_yl_fac_ in fwhm_wing0_yl_facs:
            arg_array.append((fwhm_core0_yl_fac_,fwhm_wing0_yl_fac_,
            spice_dat1, spice_hdr, 
            fwhm_core0_yl, fwhm_wing0_yl, psf_yl_angle, wing_weight, 
            yl_core_xpo, rebin_facs, spice_err_fac,
            spice_la, spice_wlcen0, corr_sdev_guess, spice_fits,
            spice_origin_vel))

    # work(arg_array[0])

    with ProcessPoolExecutor(max_workers=ncpus) as executor:
       rs=executor.map(work, arg_array[:])

    print("Done")

# def accumulate_sum():
#     sumv = 0
#     for i in v:
#         sumv += i
#     return sumv

# def main2(): 
#     import sys
#     import time
#     n = 50_000_000
#     vec = np.random.randint(0,1000,n)
#     # The script requires an input argument which is the number of processes to execute the program
#     num_processes = int(sys.argv[1])
#     n_per_process = int(n/num_processes) 
#     vec_per_process = [vec[i*n_per_process:(i+1)*n_per_process] for i in range(1)]
#     v = vec_per_process[0]
   
#     # start the stop watch
#     start = time.time()

#     with ProcessPoolExecutor(max_workers=num_processes) as executor:
#         results=executor.map(accumulate_sum)

#     # end the stop watch
#     end = time.time()

#     print("The accumulated sum is {:3.2e}".format(sum(results)))
#     print("Elasped time: {:3.2f}s".format(end-start))

if __name__ == '__main__':
    main()


# for fwhm_core0_yl_fac_ in fwhm_core0_yl_facs:
#     for fwhm_wing0_yl_fac_ in fwhm_wing0_yl_facs:
#         spice_corr_dat, spice_corr_chi2s, metadict = correct_spice_raster(spice_dat1, spice_hdr, fwhm_core0_yl*fwhm_core0_yl_fac_,
#                                                                            fwhm_wing0_yl*fwhm_wing0_yl_fac_,psf_yl_angle, wing_weight,
#                                                                            yl_core_xpo=yl_core_xpo,super_fac=1, psf_thold_core=0.0005, spice_bin_facs=rebin_facs)


#         spice_corr_mask, spice_corr_err = get_mask_errs(spice_corr_dat, spice_err_fac)

#         [det_origin0, det_dims0, det_scale0] = metadict["det_origin0"], metadict["det_dims0"], metadict["det_scale0"]

#         spice_corr_fits = fsl(spice_corr_dat, spice_corr_err, spice_la, spice_corr_mask, spice_wlcen0, corr_sdev_guess)

#         spice_corr_vel = (spice_corr_fits["centers"].T/spice_wlcen0 - 1)*const.c.to_value("km/s")
#         spice_corr_vel = spice_corr_vel - np.nanmedian(spice_corr_vel[:,:], axis=0)[np.newaxis,:]

#         fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,8),layout="constrained")

#         int_norm = ImageNormalize(stretch=AsinhStretch(0.1))

#         ax1.imshow(spice_fits["amplitudes"].T,origin="lower",norm=int_norm,aspect=1,
#                     cmap="sdoaia171")
#         ax1.set_title("Original")

#         ax2.imshow(spice_corr_fits["amplitudes"].T,origin="lower",norm=int_norm,aspect=1,
#                     cmap="sdoaia171")
#         ax2.set_title("Corrected")

#         ax3.imshow(spice_origin_vel,origin="lower",vmin=-40,vmax=40,aspect=1,
#                     cmap="coolwarm")

#         ax4.imshow(spice_corr_vel,origin="lower",vmin=-40,vmax=40,aspect=1,
#                     cmap="coolwarm")

#         fig.savefig("../sav/NeVIII_parameter_scan_fig/spice_1024_parameter_scan_core_large_bin_{:.2f}_{:.2f}_wing_{:.2f}_{:.2f}.png",
#                     dpi=300, bbox_inches="tight")



#         with h5py.File(os.path.join("../sav/NeVIII_parameter_scan/", 
#             "spice_1024_parameter_scan_core_large_bin_{:.2f}_{:.2f}_wing_{:.2f}_{:.2f}.h5".format(*(fwhm_core0_yl*fwhm_core0_yl_fac_),
#                                                                                         *(fwhm_wing0_yl*fwhm_wing0_yl_fac_))), "w") as f:
#             f.create_dataset("spice_corr_dat", data=spice_corr_dat)
#             f.create_dataset("spice_corr_chi2s", data=spice_corr_chi2s)
#             for k in metadict:
#                 f.attrs[k] = metadict[k]
