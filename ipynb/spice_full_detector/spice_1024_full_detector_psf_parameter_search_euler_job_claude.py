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
import time
from pathlib import Path
import gc
from copy import deepcopy
from glob import glob

# Explicitly set OpenBLAS threads to 1 to prevent oversubscription
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Setup logging to monitor performance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spice_psf_optimization.log"),
        logging.StreamHandler()
    ]
)

# Define main directories as Path objects for better path handling
OUTPUT_DIR = Path("/cluster/home/zhuyin/work/spice_psf/figs_full_detector_parameter_search_1024")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

def gaussian_with_bg(x, wvl, int_total, width, bg):
    return gaussian(x, wvl, int_total, width) + bg

def work(args):
    # Extract parameter set ID early for logging
    fwhm_core0_yl, fwhm_wing0_yl, psf_yl_angle, wing_weight, yl_core_xpo = args[2:7]
    params_id = f"{fwhm_core0_yl[0]:.2f}_{fwhm_core0_yl[1]:.2f}_{fwhm_wing0_yl[0]:.2f}_{fwhm_wing0_yl[1]:.2f}_{psf_yl_angle:.2f}_{wing_weight:.2f}_{yl_core_xpo:.2f}"
    
    # Check if output already exists to avoid redundant computation
    output_path = OUTPUT_DIR / f"spice_psf_{params_id}.png"
    if output_path.exists():
        logging.info(f"Skipping existing result: {params_id}")
        return None
    
    # Start timing for performance analysis
    start_time = time.time()
    
    (spice_dat1, spice_hdr, 
    fwhm_core0_yl, fwhm_wing0_yl, psf_yl_angle, wing_weight, 
    yl_core_xpo, rebin_facs, spice_orig_fit_int,
    spice_orig_fit_wvl, spice_orig_fit_wid,
    spice_la,) = args

    try:
        # Time the PSF correction step
        psf_start = time.time()
        spice_corr_dat, spice_corr_chi2s, metadict = correct_spice_raster(spice_dat1, spice_hdr, fwhm_core0_yl, fwhm_wing0_yl, 
                                                                        psf_yl_angle, wing_weight, super_fac=1, chi2_th = 0.5,
                                                                        psf_thold_core=0.0005, spice_bin_facs = rebin_facs)
        psf_time = time.time() - psf_start
        logging.debug(f"PSF correction for {params_id} took {psf_time:.2f}s")
        
        spice_corr_dat = spice_corr_dat.transpose([2,1,0])

        # Pre-allocate arrays for fit results
        NeVIII_row_20_corr_fit_wvl = np.zeros(25)
        NeVIII_row_20_corr_fit_int = np.zeros(25)
        NeVIII_row_20_corr_fit_wid = np.zeros(25)

        # Prepare indices once for fitting to avoid repeated calculations
        fit_indices = np.r_[3:17,27:42]
        
        # Time the fitting step
        fit_start = time.time()
        for ii in range(175,200):
            try:
                # Precompute values used multiple times
                spec_data = spice_corr_dat[fit_indices, ii, 0]
                spec_x = spice_la[fit_indices]
                
                # Initial parameter estimates
                max_idx = np.nanargmax(spice_corr_dat[:,ii,0])
                p0 = [
                    spice_la[max_idx],
                    np.nanmax(spice_corr_dat[:,ii,0]),
                    0.7, 
                    np.nanmean(spice_corr_dat[3:15,ii,0])
                ]
                
                popt, pcov = curve_fit(gaussian_with_bg,
                                    spec_x,
                                    spec_data,
                                    p0=p0)

                NeVIII_row_20_corr_fit_int[ii-175] = popt[1]
                NeVIII_row_20_corr_fit_wvl[ii-175] = popt[0]
                NeVIII_row_20_corr_fit_wid[ii-175] = popt[2]
            except Exception as e:
                logging.warning(f"Fitting failed for row {ii}, params {params_id}: {str(e)}")
                # Set to NaN on failure
                NeVIII_row_20_corr_fit_int[ii-175] = np.nan
                NeVIII_row_20_corr_fit_wvl[ii-175] = np.nan
                NeVIII_row_20_corr_fit_wid[ii-175] = np.nan
                
        fit_time = time.time() - fit_start
        logging.debug(f"Fitting for {params_id} took {fit_time:.2f}s")
        
        # Calculate performance metrics
        orig_std = np.nanstd(spice_orig_fit_wvl[5:])
        corr_std = np.nanstd(NeVIII_row_20_corr_fit_wvl[5:])
        std_ratio = corr_std / orig_std if orig_std > 0 else np.nan
        
        if std_ratio < 1.0:  # Only save results that show improvement
            # Time the plotting step
            plot_start = time.time()
            
            fig = plt.figure(figsize=(8,6), layout="constrained")
            axd = fig.subplot_mosaic(
                """
                AD
                BE
                CF
                CG
                """
            )

            # Use normalization objects once rather than calculating percentiles multiple times
            orig_norm = ImageNormalize(
                vmin=np.nanpercentile(spice_dat1[0,:,:], 3),
                vmax=np.nanpercentile(spice_dat1[0,:,:], 99.95),
                stretch=AsinhStretch(0.5)
            )
            corr_norm = ImageNormalize(
                vmin=np.nanpercentile(spice_corr_dat[:,:,0], 3),
                vmax=np.nanpercentile(spice_corr_dat[:,:,0], 99.95),
                stretch=AsinhStretch(0.5)
            )
            
            axd["A"].imshow(spice_dat1[0,:,:].T, origin="lower", norm=orig_norm, aspect=1)
            axd["B"].imshow(spice_corr_dat[:,:,0], origin="lower", norm=corr_norm, aspect=0.5)

            axd["C"].step(spice_la, spice_dat1[0,363//2,:], where="mid", label="Original")
            axd["C"].step(spice_la, spice_corr_dat[:,363//2,0], where="mid", label="Corrected")
            axd["C"].legend(loc='upper right', fontsize='small')

            axd["D"].plot(spice_orig_fit_int, label="Original")
            axd["D"].plot(NeVIII_row_20_corr_fit_int, label="Corrected")
            axd["D"].legend(loc='upper right', fontsize='small')
            
            axd["E"].plot(spice_orig_fit_wvl, label="Original")
            axd["E"].plot(NeVIII_row_20_corr_fit_wvl, label="Corrected")
            axd["E"].legend(loc='upper right', fontsize='small')
            
            axd["F"].plot(spice_orig_fit_wid, label="Original")
            axd["F"].plot(NeVIII_row_20_corr_fit_wid, label="Corrected")
            axd["F"].legend(loc='upper right', fontsize='small')

            axd["D"].set_ylim(0,5)
            axd["E"].set_ylim(770.0, 770.4)
            axd["F"].set_ylim(0.3,1)

            axd["G"].axis("off")
            axd["G"].text(0.2, 0.7, f"sigma_wvl_orig: {orig_std:.4f}")
            axd["G"].text(0.2, 0.3, f"sigma_wvl_corr: {corr_std:.4f}")
            axd["G"].text(0.2, 0.0, f"Ratio: {std_ratio:.4f}")

            # Save with lower DPI to speed up file writing
            fig.savefig(output_path, dpi=150)
            plt.close(fig)  # Explicitly close figure to free memory
            
            plot_time = time.time() - plot_start
            logging.debug(f"Plotting for {params_id} took {plot_time:.2f}s")
            
            # Save the performance metrics to a CSV file for later analysis
            with open(OUTPUT_DIR / "parameter_results.csv", "a") as f:
                f.write(f"{params_id},{orig_std:.6f},{corr_std:.6f},{std_ratio:.6f},{psf_time:.2f},{fit_time:.2f}\n")
        
        total_time = time.time() - start_time
        logging.info(f"Processed {params_id} in {total_time:.2f}s | Ratio: {std_ratio:.4f}")
        
        # Return performance metrics for sorting results
        return params_id, std_ratio, orig_std, corr_std
        
    except Exception as e:
        logging.error(f"Error processing {params_id}: {str(e)}")
        return None
    finally:
        # Explicitly call garbage collection to free memory
        gc.collect()


def main(test=False):
    start_time = time.time()
    logging.info("Starting SPICE PSF optimization")
    
    # Create results CSV header if it doesn't exist
    results_file = OUTPUT_DIR / "parameter_results.csv"
    if not results_file.exists():
        with open(results_file, "w") as f:
            f.write("params_id,orig_std,corr_std,ratio,psf_time,fit_time\n")
    
    # Load data only once
    logging.info("Loading SPICE raster files")
    raster_files = sorted(glob("/cluster/home/zhuyin/Solar/SPICE_psf_202210/src/full_detector/level2/2022/10/24/*.fits"))

    data_load_start = time.time()
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

    # Process files in batches to avoid memory issues
    batch_size = 5
    num_batches = (len(raster_files) + batch_size - 1) // batch_size
    
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(raster_files))
        
        logging.info(f"Processing file batch {batch+1}/{num_batches} (files {start_idx+1}-{end_idx})")
        
        for ii in range(start_idx, end_idx):
            try:
                file_ = raster_files[ii]
                datacube_ = read_spice_l2_fits(file_)
                sw_data[:,:,ii] = datacube_['Full SW 4:1 Focal Lossy'].data[0,:,:,0]
                lw_data[:,:,ii] = datacube_['Full LW 4:1 Focal Lossy'].data[0,:,:,0]
                sw_wcs_list.append(datacube_['Full SW 4:1 Focal Lossy'].wcs)
                lw_wcs_list.append(datacube_['Full LW 4:1 Focal Lossy'].wcs)
                sw_meta_list.append(datacube_['Full SW 4:1 Focal Lossy'].meta)
                lw_meta_list.append(datacube_['Full LW 4:1 Focal Lossy'].meta)
                sw_wvl_list.append(datacube_['Full SW 4:1 Focal Lossy'].spectral_axis.to_value(u.AA))
                lw_wvl_list.append(datacube_['Full LW 4:1 Focal Lossy'].spectral_axis.to_value(u.AA))
            except Exception as e:
                logging.error(f"Error loading file {file_}: {str(e)}")
        
        # Force garbage collection after each batch
        gc.collect()
    
    sw_data = np.flip(sw_data, axis=-1)
    lw_data = np.flip(lw_data, axis=-1)
    logging.info(f"Data loading completed in {time.time() - data_load_start:.2f}s")

    # Extract wavelength info just once
    test_fit_NeVIII_wvl = sw_wvl_list[0][735:799]

    # Load header once
    with fits.open(raster_files[-21]) as hdul:
        spice_hdr_20 = hdul[0].header.copy()
    
    # Prepare the header just once
    spice_hdr_20_NeVIII = deepcopy(spice_hdr_20)
    spice_hdr_20_NeVIII_new_naxis3 = 799 - 735
    spice_hdr_20_NeVIII_new_crpix3 = (1 + 799 - 735)/2
    spice_hdr_20_NeVIII_new_crval3 =  ((736 + 799)/2 - spice_hdr_20_NeVIII["CRPIX3"])*spice_hdr_20_NeVIII["CDELT3"] + spice_hdr_20_NeVIII["CRVAL3"]
    spice_hdr_20_NeVIII["NAXIS3"] = spice_hdr_20_NeVIII_new_naxis3 
    spice_hdr_20_NeVIII["CRPIX3"] = spice_hdr_20_NeVIII_new_crpix3
    spice_hdr_20_NeVIII["CRVAL3"] = spice_hdr_20_NeVIII_new_crval3
    
    # Extract and prepare the data just once
    process_start_time = time.time()
    test_NeVIII_data = sw_data[735:799,210:790,:]
    test_NeVIII_data_bg_rm = test_NeVIII_data - np.nanmean(test_NeVIII_data, axis=2)[:,:,np.newaxis]

    rebin_facs = [1,2,1]

    test_NeVIII_data_bin = bindown(test_NeVIII_data_bg_rm,np.round(np.array(test_NeVIII_data_bg_rm.shape)/rebin_facs).astype(np.int32))
    test_NeVIII_data_bin_trans = deepcopy(test_NeVIII_data_bin[:,:,20:22]).transpose([2,1,0]).astype(np.float64)
    logging.info(f"Data preparation completed in {time.time() - process_start_time:.2f}s")

    # Calculate original fits just once
    fit_start_time = time.time()
    NeVIII_row_20_orig_fit_wvl = np.zeros(25)
    NeVIII_row_20_orig_fit_int = np.zeros(25)
    NeVIII_row_20_orig_fit_wid = np.zeros(25)

    # Prepare indices for fitting once
    fit_indices = np.r_[3:17,27:42]
    
    for ii in range(175,200):
        try:
            # Precompute max value and position
            row_data = test_NeVIII_data_bin[:,ii,20]
            max_idx = np.nanargmax(row_data)
            max_val = np.nanmax(row_data)
            mean_bg = np.nanmean(row_data[3:15])
            
            popt, pcov = curve_fit(gaussian_with_bg,
                                test_fit_NeVIII_wvl[fit_indices],
                                row_data[fit_indices],
                                p0=[test_fit_NeVIII_wvl[max_idx],
                                    max_val,
                                    0.7, mean_bg])
            
            NeVIII_row_20_orig_fit_int[ii-175] = popt[1]
            NeVIII_row_20_orig_fit_wvl[ii-175] = popt[0]
            NeVIII_row_20_orig_fit_wid[ii-175] = popt[2]
        except Exception as e:
            logging.warning(f"Original fitting failed for row {ii}: {str(e)}")
            # Set to NaN on failure
            NeVIII_row_20_orig_fit_int[ii-175] = np.nan
            NeVIII_row_20_orig_fit_wvl[ii-175] = np.nan
            NeVIII_row_20_orig_fit_wid[ii-175] = np.nan
    
    logging.info(f"Original fit calculations completed in {time.time() - fit_start_time:.2f}s")

    # Clear memory of large arrays we don't need anymore
    del sw_data, lw_data, test_NeVIII_data, test_NeVIII_data_bg_rm
    gc.collect()

    if test:
        # Just test one parameter set for verification
        logging.info("Running in test mode with single parameter set")
        yl_core_xpo = 1.0
        psf_yl_angle = -20*np.pi/180
        fwhm_core0_yl = np.array([5.5, 1.15])
        fwhm_wing0_yl = np.array([20.0, 4]) 
        wing_weight = 0.21

        work((test_NeVIII_data_bin_trans, spice_hdr_20_NeVIII,
        fwhm_core0_yl, fwhm_wing0_yl, psf_yl_angle, wing_weight, 
        yl_core_xpo, rebin_facs, NeVIII_row_20_orig_fit_int,
        NeVIII_row_20_orig_fit_wvl, NeVIII_row_20_orig_fit_wid,
        test_fit_NeVIII_wvl))
    
    else:
        # Determine optimal number of CPUs based on system
        available_cpus = os.cpu_count()
        ncpus = min(10, max(1, available_cpus - 2))  # Leave some CPUs for system tasks
        logging.info(f"Using {ncpus} CPUs out of {available_cpus} available")

        # Generate parameter combinations
        logging.info("Generating parameter combinations")
        yl_core_xpo_list = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        psf_yl_angle_list = np.deg2rad(np.linspace(-15,-25,5))
        fwhm_core0_yl_0_all = np.linspace(4.5,6.5,5)
        fwhm_core0_yl_1_all = np.linspace(1.05, 1.35,7)
        fwhm_wing0_0_all = np.linspace(10,30,5)
        fwhm_wing0_1_all = np.linspace(2,6,9)
        wing_weight_list = np.linspace(0.11, 0.31, 5)

        # Use more efficient parameter combination approach
        combinations = []
        for core0 in fwhm_core0_yl_0_all:
            for core1 in fwhm_core0_yl_1_all:
                for wing0 in fwhm_wing0_0_all:
                    for wing1 in fwhm_wing0_1_all:
                        for angle in psf_yl_angle_list:
                            for weight in wing_weight_list:
                                for expo in yl_core_xpo_list:
                                    combinations.append([
                                        np.array([core0, core1]), 
                                        np.array([wing0, wing1]), 
                                        angle, weight, expo
                                    ])
        
        # Filter combinations to avoid redundant work
        logging.info(f"Generated {len(combinations)} parameter combinations")
        filtered_combinations = []
        for combo in combinations:
            params_id = f"{combo[0][0]:.2f}_{combo[0][1]:.2f}_{combo[1][0]:.2f}_{combo[1][1]:.2f}_{combo[2]:.2f}_{combo[3]:.2f}_{combo[4]:.2f}"
            output_path = OUTPUT_DIR / f"spice_psf_{params_id}.png"
            if not output_path.exists():
                filtered_combinations.append(combo)
        
        logging.info(f"Filtered to {len(filtered_combinations)} combinations after excluding existing results")
        
        # Limit to first 400 combinations as in original code
        max_combinations = 400
        if len(filtered_combinations) > max_combinations:
            logging.info(f"Limiting to first {max_combinations} parameter combinations")
            filtered_combinations = filtered_combinations[:max_combinations]

        # Create argument array
        arg_array = []
        for combo in filtered_combinations:
            arg_array.append([
                test_NeVIII_data_bin_trans, spice_hdr_20_NeVIII,
                combo[0], combo[1], combo[2], combo[3], combo[4], 
                rebin_facs, NeVIII_row_20_orig_fit_int,
                NeVIII_row_20_orig_fit_wvl, NeVIII_row_20_orig_fit_wid,
                test_fit_NeVIII_wvl
            ])

        # Process in batches to better manage memory
        batch_size = min(50, len(arg_array))
        num_batches = (len(arg_array) + batch_size - 1) // batch_size
        all_results = []
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(arg_array))
            current_batch = arg_array[batch_start:batch_end]
            
            logging.info(f"Processing batch {batch_idx+1}/{num_batches} with {len(current_batch)} parameter sets")
            
            with ProcessPoolExecutor(max_workers=ncpus) as executor:
                # Use an appropriate chunksize for better load balancing
                chunksize = max(1, len(current_batch) // (ncpus * 2))
                batch_results = list(executor.map(work, current_batch, chunksize=chunksize))
                
                # Filter out None results and extend all_results
                batch_results = [r for r in batch_results if r is not None]
                all_results.extend(batch_results)
            
            # Force garbage collection between batches
            gc.collect()
            
            logging.info(f"Completed batch {batch_idx+1}/{num_batches} with {len(batch_results)} valid results")
        
        # Generate summary report
        if all_results:
            # Sort by ratio (better results first)
            all_results.sort(key=lambda x: x[1])
            
            with open(OUTPUT_DIR / "summary_report.txt", "w") as f:
                f.write("SPICE PSF Parameter Optimization Results\n")
                f.write("======================================\n\n")
                f.write("Top 10 Parameter Sets:\n")
                f.write("Parameter Set | Wavelength Std Ratio | Original Std | Corrected Std\n")
                f.write("-" * 75 + "\n")
                
                for i, (params_id, ratio, orig_std, corr_std) in enumerate(all_results[:10]):
                    f.write(f"{i+1}. {params_id} | {ratio:.4f} | {orig_std:.4f} | {corr_std:.4f}\n")
            
            logging.info(f"Analysis complete. Top result: {all_results[0][0]} with ratio {all_results[0][1]:.4f}")
        else:
            logging.warning("No valid results were produced")
        
    total_runtime = time.time() - start_time
    logging.info(f"Total runtime: {total_runtime/60:.2f} minutes ({total_runtime:.2f} seconds)")

if __name__ == '__main__':
    main(test=False)