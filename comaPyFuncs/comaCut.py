import healpy as hp
import numpy as np
import drizzlib
import os, sys
import datetime

from matplotlib import pyplot
from drizzlib import healpix2wcs
from astropy.io import fits as pyfits

print(drizzlib.__file__)

def rotate_coords(nside, theta, phi):
  r=hp.Rotator(coord=['C', 'G'])
  trot, prot=r(theta, phi)
  pix=hp.ang2pix(nside, trot, prot)
  return np.arange(len(theta))[pix]

# Output directory for files
outdir=sys.argv[1]

patch_RA=194.952916667
patch_dec=27.9805555556

nside=2048
npix=700
npatch=1

bad_pix=-1.6375e+30

if not os.path.exists(outdir): os.mkdir(outdir)

if not os.path.exists(outdir+'/hp_proj'): os.mkdir(outdir+'/hp_proj')
if not os.path.exists(outdir+'/drizzlib'): os.mkdir(outdir+'/drizzlib')

output_gal=True
if output_gal:
  map_proj=['GLON-SIN',
            'GLAT-SIN']
  which_coord='G'
  switch_coord=['G']
else:
  map_proj=['RA---TAN',
            'DEC--TAN']
  which_coord='C'
  switch_coord=['G', 'C']

fits_temp=pyfits.open('../pws_template.fits')
try:
  fits_temp[1].header.update('naxis1', npix)
  fits_temp[1].header.update('naxis2', npix)
  fits_temp[1].header.update('crpix1', npix/2+1)
  fits_temp[1].header.update('crpix2', npix/2+1)
  for i in range(1,3):
    fits_temp[1].header.update('ctype'+str(i), map_proj[i-1])
except ValueError:
  fits_temp[1].header.update([('naxis1', npix), ('naxis2', npix), ('crpix1', npix/2+1), ('crpix2', npix/2+1), ('ctype1', map_proj[0]), ('ctype2', map_proj[1])])

y_mapname='../COM_CompMap_Compton-SZMap-milca-ymaps_2048_R2.00.fits'
mapsets=['full', 'first', 'last']

for ims, mapset in enumerate(mapsets):
  ymap=hp.read_map(y_mapname, field=ims)
  if hp.get_nside(ymap)!=nside:
    print('Warning: specified nside (%d) not consistent with nside from healpix map (%d)' % (nside, hp.get_nside(ymap)))
  nside=hp.get_nside(ymap)
  pix=hp.nside2resol(nside)*180./np.pi
  try:
    fits_temp[1].header.update('cdelt1', pix)
    fits_temp[1].header.update('cdelt2', pix)
  except ValueError:
    fits_temp[1].header.update([('cdelt1', pix), ('cdelt2', pix)])

  theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
  patch_cen_j=[patch_RA, patch_dec]
  cen=[np.pi/2.-patch_dec*np.pi/180., patch_RA*np.pi/180.] # (theta,phi) in radians

  r=hp.Rotator(coord=['C', 'G'])
  cen_g=r(cen)
  patch_cen_g=[cen_g[1]*180./np.pi, 90-cen_g[0]*180./np.pi]

  if output_gal:
    patch_cen=patch_cen_g
  else:
    patch_cen=patch_cen_j

  try:
    for icoord in range(1,3):
      fits_temp[1].header.update('crval'+str(icoord), patch_cen[icoord-1])
  except ValueError:
    fits_temp[1].header.update([('crval1', patch_cen[0]), ('crval2', patch_cen[1])])

  # Cut out MILCA map
  if output_gal:
    mapout='%s/hp_proj/map%d_MILCA_Coma_20deg_G_%s.fits' % (outdir,nside,mapset)
    mapout2='%s/drizzlib/map%d_MILCA_Coma_20deg_G_%s.fits' % (outdir,nside,mapset)
  else:
    mapout='%s/hp_proj/map%d_MILCA_Coma_20deg_%s.fits' % (outdir,nside,mapset)
    mapout2='%s/drizzlib/map%d_MILCA_Coma_20deg_%s.fits' % (outdir,nside,mapset)

  y_patch=hp.gnomview(ymap, rot=patch_cen, coord=switch_coord, xsize=npix, reso=pix*60., return_projected_map=True, flip='astro')
  pyplot.close('all')
  y_patch=y_patch.filled(fill_value=y_patch.fill_value)
  y_patch=y_patch[:,::-1]
  freq=220.
  now=datetime.datetime.now()
  try:
    fits_temp[0].header.update('date', now.strftime('%Y-%m-%dT%H:%M:%S'))
    fits_temp[1].header.update('freq', freq*1e9)
    fits_temp[1].header.update('bunit', 'COMPTON-Y')
    # Use astro coordinate convention
    fits_temp[1].header.update('cdelt1', -np.abs(fits_temp[1].header['cdelt1']))
  except ValueError:
    fits_temp[0].header.update([('date', now.strftime('%Y-%m-%dT%H:%M:%S'))])
    fits_temp[1].header.update([('freq', freq*1e9), ('bunit', 'COMPTON-Y')])
    # Use astro coordinate convention
    fits_temp[1].header.update([('cdelt1', -np.abs(fits_temp[1].header['cdelt1']))])

  fits_temp[1].data=y_patch

  fits_temp.verify('fix')
  fits_temp.writeto(mapout, overwrite=True)
  #y_patch2=healpix2wcs.healpix2wcs(y_mapname, hp_field=ims, header=mapout, header_hdu=1, output=mapout2, clobber=True)
  y_patch2=healpix2wcs.healpix2wcs(y_mapname, col_ids=[ims], header=mapout, header_hdu=1, output=mapout2, clobber=True)
  # PwS needs two hdus so copy from the template
  f=pyfits.open(mapout)
  f[1].data=y_patch2
  f.writeto(mapout2, overwrite=True)
  f.close()
