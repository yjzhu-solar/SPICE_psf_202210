import numpy as np
import os 
import sys
import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator as lndi

sys.path.append("/cluster/home/zhuyin/scripts/spice-line-fits/linefit_modules/")

from util import get_mask_errs, get_spice_err
from skew_correction import skew_correct, deskew_linefit_window
from skew_parameter_search import search_shifts, shift_holder, refine_points
from linefit_leastsquares import lsq_fitter, lsq_fitter
from linefit_storage import linefits

fitter = lsq_fitter # lsq_fitter
from linefit_leastsquares import check_for_waves

spice_file = "~/work/spice_psf/spice_data/solo_L2_spice-n-ras_20231016T190607_V22_218104099-075_coalign.fits"

with fits.open(spice_file) as hdul:
    hdul.info()
    spice_hdr = hdul[4].header.copy()
    spice_dat = hdul[4].data[0].copy()
spice_dat = spice_dat.transpose([2,1,0]).astype(np.float32)

spice_dx, spice_dy, spice_dl = spice_hdr['CDELT1'],spice_hdr['CDELT2'],10*spice_hdr['CDELT3']
spice_wl0 = 10*spice_hdr['CRVAL3']-spice_dl*spice_hdr['CRPIX3']
spice_la = spice_wl0+spice_dl*np.arange(spice_dat.shape[2],dtype=np.float64)

linelist = {'LyBeta':1025.7}

line_names = list(linelist.keys())
line_waves = [linelist[name] for name in line_names]

centers, lines = check_for_waves(spice_la)

xl, xh, yl, yh = [-5,5,-5,5]
xs_initial, ys_initial = np.array(np.meshgrid(np.linspace(xl,xh,5),np.linspace(yl,yh,5))).transpose([0,2,1])

print(xs_initial, ys_initial)

shift_vars = shift_holder(spice_dat, spice_hdr, fitter.__name__, save_dir='/cluster/home/zhuyin/work/spice_psf/spice_skew_20231016/',
                          linelist=linelist)

sv_initial = search_shifts(spice_dat, spice_hdr, xs_initial, ys_initial,
                           lsq_fitter, search_multi_thread=True, search_nthread=36,
                           yrange_plot_dir='/cluster/home/zhuyin/work/spice_psf/spice_skew_20231016/yrange_plots/',
                           shift_vars=shift_vars, linelist=linelist)

shift_vars.set(sv_initial)

shift_vars.save()

x_refine, y_refine = refine_points(shift_vars,[-5,5],[-5,5], 11, 11, 20)
shift_vars = search_shifts(spice_dat, spice_hdr, x_refine, y_refine,
                           lsq_fitter, shift_vars=shift_vars, search_multi_thread=True, search_nthread=36,
                           yrange_plot_dir='/cluster/home/zhuyin/work/spice_psf/spice_skew_20231016/yrange_plots/',
                           linelist=linelist)
shift_vars.save()


x_refine, y_refine = refine_points(shift_vars,[-5,5],[-5,5], 31, 31, 20)
shift_vars = search_shifts(spice_dat, spice_hdr, x_refine, y_refine,
                           lsq_fitter, shift_vars=shift_vars, search_multi_thread=True, search_nthread=36,
                           yrange_plot_dir='/cluster/home/zhuyin/work/spice_psf/spice_skew_20231016/yrange_plots/',
                           linelist=linelist)
shift_vars.save()

x_refine, y_refine = refine_points(shift_vars,[-5,5],[-5,5], 101, 101, 20)
shift_vars = search_shifts(spice_dat, spice_hdr, x_refine, y_refine,
                           lsq_fitter, shift_vars=shift_vars, search_multi_thread=True, search_nthread=36,
                           yrange_plot_dir='/cluster/home/zhuyin/work/spice_psf/spice_skew_20231016/yrange_plots/',
                           linelist=linelist)
shift_vars.save()

xa = np.array(list(shift_vars.valdict.values()))[:,0]
ya = np.array(list(shift_vars.valdict.values()))[:,1]
dat = np.array(list(shift_vars.valdict.values()))[:,2]

include = (np.abs(xa) > 1.0e-5)*(np.abs(ya) > 1.0e-5)

nx_plot, ny_plot = 41, 41
xya = np.vstack([xa[include],ya[include]]).T
xa0,ya0 = np.array(np.meshgrid(np.linspace(xl,xh,nx_plot),np.linspace(yl,yh,ny_plot))).transpose([0,2,1])
dat_interp = lndi(xya, dat[include])(xa0,ya0)

dat_interp = lndi(xya, dat[include])(xa0,ya0)

sort_interp = np.argsort(dat_interp.flatten())
xsort_interp = xa0.flatten()[sort_interp]
ysort_interp = ya0.flatten()[sort_interp]

xl2, xh2 = xl-0.5*(xh-xl)/(nx_plot-1), xh+0.5*(xh-xl)/(nx_plot-1)
yl2, yh2 = yl-0.5*(yh-yl)/(ny_plot-1), yh+0.5*(yh-yl)/(ny_plot-1)

labelstr = spice_hdr['DATE-OBS']+' '+spice_hdr['EXTNAME']
labelstr = labelstr.replace('-','').replace(':','').replace('  ','_').replace(' ','_')
labelstr = labelstr.replace('/','_')
print(labelstr)

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=[16,9])
plt.suptitle(spice_hdr['DATE-OBS']+' '+spice_hdr['EXTNAME']+': xyshift='+str(np.array([xa[np.argmin(dat)], ya[np.argmin(dat)]])))
axes[0].imshow(np.clip(np.nansum(spice_dat,axis=2).T,0,None)[150:850,:]**0.5,vmin=0,vmax=(100*np.nanmean(spice_dat))**0.5,
               aspect=spice_hdr['CDELT2']/spice_hdr['CDELT1'], origin="lower")
axes[0].set(title='Spectral sum',xlabel='Raster axis @ ypix equivalent -- '+str(spice_hdr['cdelt2'])+'"')
asdfa = axes[1].imshow(dat_interp.T, extent=[xl2, xh2, yl2, yh2],cmap=plt.get_cmap('gray'), origin="lower")
axes[1].plot(xa,ya,'P',markersize=10,linewidth=5)
axes[1].set(title='RMS Doppler variance', xlabel='x shift (arcsecond/angstrom)', ylabel='y shift (arcsecond/angstrom)')
axes[1].legend(['Sampled points'])
fig.colorbar(asdfa, ax=axes[1],location='bottom')
plt.savefig(os.path.join('/cluster/home/zhuyin/work/spice_psf/spice_skew_20231016/yrange_plots/','varplot_'+labelstr+'.png'))
plt.close()






