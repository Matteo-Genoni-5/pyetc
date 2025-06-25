import logging
from astropy import constants
import astropy.units as u
from astropy.modeling.models import Sersic2D
from pyetc_dev.version import  __version__
from mpdaf.obj import Spectrum, WaveCoord, Cube, WCS
from mpdaf.obj import gauss_image, mag2flux, flux2mag, moffat_image, Image
from astropy.table import Table
import os, sys
import numpy as np
from mpdaf.log import setup_logging
from scipy.special import erf
from scipy.optimize import root_scalar, minimize_scalar
from scipy.signal import fftconvolve
from skycalc_cli.skycalc import SkyModel
from skycalc_cli.skycalc_cli import fixObservatory
from io import BytesIO
import glob
#from collections import OrderedDict
from spextra import Spextrum
from .specalib import PhotometricSystem, SEDModels, FilterManager
import skycalc_ipy
import time

# global variables
C_cgs = constants.c.cgs.value
H_cgs = constants.h.cgs.value
C_kms = constants.c.to(u.km/u.s).value
g_frac_ima = None
g_size_ima = None
g_nspaxels = None

# # # # for now hard coded here
sat_level = 65000
# # # #

# Initialize photometric system, SED models and filter manager
phot_system = PhotometricSystem()
sed_models = SEDModels()
filter_manager = FilterManager(phot_system)


__all__ = ['ETC', 'asymgauss', 'compute_sky', 'fwhm_asymgauss', 'get_data',
           'get_seeing_fwhm', 'moffat', 'peakwave_asymgauss', 'show_noise',
           'sigma2vdisp', 'update_skytables', 'vdisp2sigma', 'check_range', 
           'check_line']


class ETC:
    """ Generic class for Exposure Time Computation (ETC) """

    def __init__(self, log=logging.INFO):
        self.version = __version__
        #setup_logging(__name__, level=log, stream=sys.stdout)
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False

    def set_logging(self, log):
        """ Change logging value

        Parameters
        ----------
        log : str
             desired log mode "DEBUG","INFO","WARNING","ERROR"

        """

        self.logger.setLevel(log)


    def _info(self, ins_names, full=False):
        """ print detailed information

        Parameters
        ----------
        ins_names : list of str
               list of instrument names (e.g ['ifs','moslr'])

        """  
        self.logger.info('%s ETC version: %s', self.name, self.version)
        if ('desc' in self.tel) and ('version' in self.tel):
            self.logger.info('Telescope %s version %s', self.tel['desc'],self.tel['version'])      
        self.logger.info('Diameter: %.2f m Area: %.1f m2', self.tel['diameter'],self.tel['area'])
        for ins_name in ins_names:
            insfam = getattr(self, ins_name)
            for chan in insfam['channels']:
                ins = insfam[chan]
                self.logger.info('%s type %s Channel %s', ins_name.upper(), ins['type'], chan)
                self.logger.info('\t %s', ins['desc'])
                self.logger.info('\t version %s', ins['version'])
                self.logger.info('\t Obscuration %.3f', ins.get('obscuration', 0))
                self.logger.info('\t Spaxel size: %.2f arcsec Image Quality tel+ins fwhm: %.2f arcsec beta: %.2f ', ins['spaxel_size'], ins['iq_fwhm'], ins['iq_beta'])
                if 'aperture' in ins.keys():
                    self.logger.info('\t Fiber aperture: %.1f arcsec', ins['aperture'])
                self.logger.info('\t Wavelength range %s A step %.2f A LSF %.1f pix Npix %d', ins['instrans'].get_range(),
                                  ins['instrans'].get_step(), ins['lsfpix'], ins['instrans'].shape[0])
                self.logger.info('\t Instrument transmission peak %.2f at %.0f - min %.2f at %.0f',
                                  ins['instrans'].data.max(), ins['instrans'].wave.coord(ins['instrans'].data.argmax()),
                                  ins['instrans'].data.min(), ins['instrans'].wave.coord(ins['instrans'].data.argmin()))
                self.logger.info('\t Detector RON %.1f e- Dark %.1f e-/h', ins['ron'],ins['dcurrent'])
                if full:
                    for sky in ins['sky']:
                        self.logger.info('\t Sky moon %s airmass %s table %s', sky['moon'], sky['airmass'],
                                          os.path.basename(sky['filename']))
                    self.logger.info('\t Instrument transmission table %s', os.path.basename(ins['instrans'].filename))


    def set_obs(self, obs):
        """save obs dictionary to self

        Parameters
        ----------
        obs : dict
            dictionary of observation parameters

        """
        # # # not very useful
        
        #if ('ndit' in obs.keys()) and ('dit' in obs.keys()):
        #    obs['totexp'] = obs['ndit']*obs['dit']/3600.0 # tot integ time in hours
        
        self.obs = obs

    def get_spectral_resolution(self, ins):
        """ return spectral resolving power

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])

        Returns
        -------
        numpy array
            spectral resoving power (lbda/dlbda)

        """
        lsf = ins['lsfpix']*ins['dlbda']
        wave = ins['wave'].coord()
        res = wave/lsf
        return res

    def get_sky(self, ins, moon):
        """ return sky emission and tranmission spectra

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        moon : str
            moon observing condition (e.g "darksky")

        Returns
        -------
        tuple of MPDAF spectra
            emission and absorption sky spectra

        """
        airmass = self.obs['airmass']
        for sky in ins['sky']:
            if np.isclose(sky['airmass'], airmass) and (sky['moon'] == moon):
                return sky['emi'],sky['abs']
        raise ValueError(f"moon {moon} airmass {airmass} not found in loaded sky configurations")

    def get_spec(self, ins, dspec, oversamp=10, lsfconv=True):

        """ compute source spectrum from the model parameters

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        dspec : dict
            dictionary of parameters describing the source spectrum
        oversamp : int
             oversampling factor (Default value = 10)
        lsfconv : bool
             apply LSF convolution (Default value = True)

        Returns
        -------
        MPDAF spectrum
            resulting source spectrum

        """
        lstep = ins['instrans'].get_step()
        l1,l2 = ins['instrans'].get_start(),ins['instrans'].get_end()
        if dspec['type'] == "flatcont":
            dlbda = ins['dlbda']
            wave = ins['instrans'].wave
            k = wave.pixel(dspec['wave'][0], nearest=True)
            l1 = wave.coord(k)
            k = wave.pixel(dspec['wave'][1], nearest=True)
            l2 = wave.coord(k)
            npts = int((l2 - l1)/dlbda + 1.5)
            spec = Spectrum(wave=WaveCoord(cdelt=dlbda, crval=l1), data=np.ones(npts))          
            oversamp = 1  # we do not oversamp for flatcont
        elif dspec['type'] == "template":
            # get template spectrum
            name = dspec['name']
            sp = Spextrum(name)
            redshift = dspec.get('redshift', None)
            # Do we want to redshift this?
            if redshift is not None:
                #print("Redshifting to z={0}".format(redshift))
                sp = sp.redshift(z=redshift)           
            l0,dl = dspec['wave_center'],dspec['wave_width']
            w,y = sp._get_arrays(wavelengths=None)
            w,y = w.value,y.value
            dw = w[1] - w[0]
            spec0 = Spectrum(data=y, wave=WaveCoord(cdelt=dw, crval=w[0]))
            # normalise to 1 in the given window
            if ((l0-dl/2) < w[0]) or ((l0+dl/2) > w[-1]):
                raise ValueError('wave range outside template wavelength limits')
            vmean = spec0.mean(lmin=l0-dl/2,lmax=l0+dl/2)[0]
            spec0.data /= vmean
            # resample
            rspec = spec0.resample(lstep, start=l1)
            rspec = rspec.subspec(lmin=l1, lmax=l2)
            # LSF convolution
            if lsfconv:
                # # # # # don't need to change
                spec = rspec.filter(width=ins['lsfpix'])
            else:
                spec = rspec
            oversamp = 1  # we do not oversamp for template
        elif dspec['type'] == 'line':
            kfwhm = dspec.get('kfwhm', 5)
            dl = kfwhm*10*dspec['sigma']
            if dspec['skew'] == 0:
                l0 = dspec['lbda']
            else:
                l0 = peakwave_asymgauss(dspec['lbda'], dspec['sigma'], dspec['skew'])
            wave = np.arange(dspec['lbda']-dl,dspec['lbda']+dl,lstep/oversamp)
            f = asymgauss(1.0, l0, dspec['sigma'], dspec['skew'], wave)
            sp = Spectrum(data=f, wave=WaveCoord(cdelt=(wave[1]-wave[0]), crval=wave[0]))
            l0,l1,l2 = fwhm_asymgauss(sp.wave.coord(),sp.data)
            dl1,dl2 = l1-l0,l2-l0
            l1,l2 = l0+kfwhm*dl1,l0+kfwhm*dl2
            rspec = sp.subspec(lmin=l1, lmax=l2)
            # LSF convolution
            if lsfconv:
                # # # # # don't need to change
                spec = rspec.filter(width=ins['lsfpix']*oversamp)
            else:
                spec = rspec
        else:
            raise ValueError('Unknown spectral type')
        spec.oversamp = oversamp
        return spec

    def get_ima(self, ins, dima, oversamp=10, uneven=1):
        """ compute source image from the model parameters

         Parameters
         ----------
         ins : dict
             instrument (eg self.ifs['blue'] or self.moslr['red'])
         dima : dict
             dictionary of parameters describing the source spectrum
         oversamp : int
             oversampling factor (Default value = 10)
         uneven : int
              if 1 the size of the image will be uneven (Default value = 1)

         Returns
         -------
         MPDAF image
             image of the source

         """
        if dima['type'] == 'moffat':
            kfwhm = dima.get('kfwhm', 3)
            ima = moffat(ins['spaxel_size'], dima['fwhm'], dima['beta'], dima.get('ell',0),
                         kfwhm=kfwhm, oversamp=oversamp, uneven=uneven)
        elif dima['type'] == 'sersic':
            ima = sersic(ins['spaxel_size'], dima['reff'], dima['n'], dima.get('ell',0),
                         kreff=dima.get('kreff',4), oversamp=oversamp, uneven=uneven)            
        else:
            raise ValueError(f"Unknown image type {dima['type']}")
        ima.oversamp = oversamp
        return ima


    def truncate_spec_adaptative(self, ins, spec, kfwhm):
        """ truncate an emission line spectrum as function of the line FWHM
            the window size is compute as center +/- kfwhm*fwhm


        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        spec : MPFAF spectrum
            source spectrum
        kfwhm : float
            factor relative to the line FWHM,
        Returns
        -------
        tuple
             tspec truncated MPDAF spectrum
             waves numpy array of corresponding wavelengths (A)
             nspectels (int) number of spectels kept
             size (float) wavelength range (A)
             frac_flux (float) fraction flux kept after truncation

        """
        l0,l1,l2 = fwhm_asymgauss(spec.wave.coord(), spec.data)
        temp = ins['instrans'].subspec(lmin=spec.get_start(), lmax=spec.get_end())
        rspec = spec.resample(temp.get_step(unit='Angstrom'), start=temp.get_start(unit='Angstrom'),
                            shape=temp.shape[0])
        # WIP check if rspec has a shape > 0
        rspec.data *= spec.data.sum()/rspec.data.sum()
        nl1 = l0 - kfwhm*(l0-l1)
        nl2 = l0 + kfwhm*(l2-l0)
        if (nl1 < rspec.get_start()) or (nl2 > rspec.get_end()):
            raise ValueError(f'adaptive spectra truncation outside spec limits')
        if nl2-nl1 <= rspec.get_step():
            raise ValueError(f'kfwhm is too small {kfwhm}')
        tspec = rspec.subspec(lmin=nl1, lmax=nl2)
        waves = tspec.wave.coord()
        nspectels = tspec.shape[0]
        size = tspec.get_end() - tspec.get_start()
        frac_flux = tspec.data.sum()
        return tspec,waves,nspectels,size,frac_flux

    def optimum_spectral_range(self, ins, flux, ima, spec):
        """ compute the optimum window range which maximize the S/N

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        flux : float
            flux value in erg/s/cm2 (for line), in erg/s/cm2/A (for cont), / arcsec2 (for sb)
        ima : MPDAF image
            source image
        spec : MPDAF spectrum
            source spectrum

        Returns
        -------
        float
            factor relative to FWHM (kfwhm)

            the kfwhm value is also updated into the obs dictionary
        """
        obs = self.obs
        if obs['spec_range_type'] != 'adaptative':
            raise ValueError('obs spec_range_type must be set to adaptative')
        l0,l1,l2 = fwhm_asymgauss(spec.wave.coord(), spec.data)
        kmax = min((spec.get_end()-l0)/(l2-l0),(l0-spec.get_start())/(l0-l1))
        kmin = max(spec.get_step()/(l2-l0), spec.get_step()/(l0-l1))
        bracket = [5*kmin,0.9*kmax]
        self.logger.debug('Optimizing kwhm in %s', bracket)
        is_ps = False
        if (ima is None) and (obs['ima_type'] == 'ps'):
            is_ps = True
            # we compute the PSF at the central wavelength
            ima = self.get_image_psf(ins, l0)
            obs['ima_type'] = 'resolved'
        res = minimize_scalar(_fun_range, args=(self, ins, flux, ima, spec),
                              bounds=bracket, options=dict(xatol=0.01), method='bounded')
        kfwhm = res.x
        snr = -res.fun
        self.obs['spec_range_kfwhm'] = kfwhm
        tspec,waves,nspectels,size,frac = self.truncate_spec_adaptative(ins, spec, kfwhm)
        if is_ps:
            obs['ima_type'] = 'ps' # set ima_type back to ps
        self.logger.debug('Optimum spectral range nit=%d kfwhm=%.2f S/N=%.1f Size=%.1f Flux=%.2e Frac=%.2f',res.nfev,kfwhm,snr,size,flux,frac)
        return res.x


    def truncate_spec_fixed(self, ins, spec, nsp):
        """ truncate the spectrum to a fixed spectral window size

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        spec : MPDAF spectrum
            source spectrum
        nsp : int
            half number of spectels to use (size is 2 * nsp + 1)

        Returns
        -------
        tuple
             tspec truncated MPDAF spectrum
             waves numpy array of corresponding wavelengths (A)
             nspectels (int) number of spectels kept
             size (float) wavelength range (A)
             frac_flux (float) fraction flux kept after truncation

        """
        l0,l1,l2 = fwhm_asymgauss(spec.wave.coord(), spec.data)
        temp = ins['instrans'].subspec(lmin=spec.get_start(), lmax=spec.get_end())
        rspec = spec.resample(temp.get_step(unit='Angstrom'), start=temp.get_start(unit='Angstrom'),
                            shape=temp.shape[0])
        rspec.crop() # WIP remove any masked data
        rspec.data *= spec.data.sum()/rspec.data.sum()
        k0 = rspec.wave.pixel(l0, nearest=True)
        tspec = rspec[max(k0-nsp,0):min(k0+nsp+1,rspec.shape[0])]
        nspectels = 2*nsp+1
        size = nspectels*ins['dlbda']
        frac_flux = tspec.data.sum()
        if nsp == 0:
            tspec = tspec.data[0]
            waves = rspec.wave.coord(k0)
        else:
            waves = tspec.wave.coord()
        return tspec,waves,nspectels,size,frac_flux

    def adaptative_circular_aperture(self, ins, ima, kfwhm):
        """ truncate the image with a window  size relative to the image FWHM

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image
        kfwhm : float
            factor relative to FWHM to define the truncation aperture

        Returns
        -------
        tuple
             tima truncated MPDAF image
             nspaxels (int) number of spaxels
             size (float) aperture size (diameter, arcsec)
             frac_flux (float) fraction flux kept after truncation+
        """
        peak = ima.peak()
        center = (peak['p'],peak['q'])
        fwhm,_ = ima.fwhm(center=center, unit_center=None, unit_radius=None)
        rad = kfwhm*fwhm*ins['spaxel_size']/ima.oversamp
        tima,nspaxels,size_ima,frac_ima = self.fixed_circular_aperture(ins, ima, rad)
        return tima,nspaxels,size_ima,frac_ima

    def optimum_circular_aperture(self, ins, flux, ima, spec, bracket=[1,5], lrange=None):
        """ compute the optimum aperture which maximize the S/N

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        flux : float
            flux value in erg/s/cm2 (for line), in erg/s/cm2/A (for cont), / arcsec2 (for sb)
        ima : MPDAF image
            source image
        spec : MPDAF spectrum
            source spectrum
        bracket : tuple
             (Default value = [1,5]) :

        lrange : tuple
             wavelength range to compute S/N for cont spectrum (Default value = None)

        Returns
        -------
        float
            factor relative to FWHM (kfwhm)

            the kfwhm value is also updated into the obs dictionary

        """
        obs = self.obs
        if obs['ima_aperture_type'] != 'circular_adaptative':
            raise ValueError('obs ima_aperture_type must be set to circular_adaptative')
        if obs['spec_type'] == 'cont':
            if lrange is None:
                raise ValueError('lrange must be set when spec_type is cont')
            krange = spec.wave.pixel(lrange, nearest=True)
        else:
            krange = None
        is_ps = False
        if (ima is None) and (obs['ima_type'] == 'ps'):
            is_ps = True
            # we compute the PSF at the central wavelength
            l0 = 0.5*(spec.get_end() + spec.get_start())
            ima = self.get_image_psf(ins, l0)
            obs['ima_type'] = 'resolved'
        res = minimize_scalar(_fun_aper, args=(self, ins, flux, ima, spec, krange),
                              bracket=bracket, tol=0.01, method='brent')
        kfwhm = res.x
        snr = -res.fun
        self.obs['ima_kfwhm'] = kfwhm
        tima,nspaxels,size_ima,frac_ima = self.adaptative_circular_aperture(ins, ima, kfwhm)
        self.logger.debug('Optimum circular aperture nit=%d kfwhm=%.2f S/N=%.1f Aper=%.1f Flux=%.2e Frac=%.2f',res.nit,kfwhm,snr,size_ima,flux,frac_ima)
        if is_ps:
            obs['ima_type'] = 'ps' # set ima_type back to ps
        return res.x

    def fixed_circular_aperture(self, ins, ima, radius, cut=1/16):
        """ truncate the image with a fixed size aperture

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image
        radius : float
            aperture radius in arcsec
        cut : float
            spaxels with flux below cut will not be counted (Default value = 1/16)

        Returns
        -------
        tuple
             tima truncated MPDAF image
             nspaxels (int) number of spaxels
             size (float) aperture size (diameter, arcsec)
             frac_flux (float) fraction flux kept after truncation

        """
        peak = ima.peak()
        rad = radius*ima.oversamp/ins['spaxel_size']
        center = (peak['p'],peak['q'])
        tima = ima.copy()
        x, y = np.meshgrid((np.arange(ima.shape[1]) - center[1]),
                               (np.arange(ima.shape[0]) - center[0]))
        ksel = (x**2 + y**2 > rad**2)
        tima.data[ksel] = 0
        rima = tima.rebin(ima.oversamp)
        rima.data *= tima.data.sum()/rima.data.sum()
        ksel = rima.data < cut*rima.data.max()
        rima.mask_selection(ksel)
        rima.crop()
        size = 2*radius
        frac_flux = rima.data.sum()
        ksel = rima.data <= cut*rima.data.max()
        rima.data[ksel] = 0
        rima.mask_selection(ksel)
        nspaxels = np.count_nonzero(rima.data)
        return rima,nspaxels,size,frac_flux

    def square_aperture(self, ins, ima, nsp):
        """ truncate an image on a squared aperture

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image
        nsp : int
            the number of spaxels to use is 2 * nsp +1

        Returns
        -------
        tuple
             tima truncated MPDAF image
             nspaxels (int) number of spaxels
             size (float) aperture size (diameter, arcsec)
             frac_flux (float) fraction flux kept after truncation

        """
        rima = ima.rebin(ima.oversamp)
        rima.data *= ima.data.sum()/rima.data.sum()
        nspaxels = (2*nsp+1)**2
        peak = rima.peak()
        p,q = int(peak['p']+0.5),int(peak['q']+0.5)
        tima = rima[p-nsp:p+nsp+1, q-nsp:q+nsp+1]
        frac_flux = np.sum(tima.data)
        size = (2*nsp+1)*ins['spaxel_size']
        return tima,nspaxels,size,frac_flux

    def get_psf_frac_ima(self, ins, flux, spec, lrange=None, oversamp=10, lbin=1):
        """ compute the flux fraction evolution with seeing for a point source

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        flux : float
            flux value in erg/s/cm2 (for line), in erg/s/cm2/A (for cont), / arcsec2 (for sb)
        ima : MPDAF image
            source image
        spec : MPDAF spectrum
            source spectrum
        lrange : tuple
             wavelength range to compute S/N for cont spectrum (Default value = None)
        oversamp : int
             oversampling factor (Default value = 10)
        lbin : int
             step in wavelength (Default value = 1)

        Returns
        -------
        tuple
            frac_ima (MPDAF spectrum) fraction of flux as function of wavelength
            size_ima (MPDAF spectrum) diameter in arcsec of the aperture
            nspaxels (numpy array of int) corresponding number of spaxels within the aperture

         """
        obs = self.obs
        moon = obs['moon']
        if obs['ima_type'] != 'ps':
            raise ValueError('get_psf_frac_ima only work in ps ima_type')
        obs['ima_type'] = 'resolved' # switch to resolved for the computation
        fwhm = self.get_image_quality(ins, spec)
        beta = ins['iq_beta']
        # if IFS and adaptative, compute the optimal parameters for the two extreme wavelength
        if (ins['type'] == 'IFS') and (obs['ima_aperture_type'] == 'circular_adaptative'):
            self.logger.debug('Computing optimum values for kfwhm')
            if lrange is None:
                l0 = 0.5*(spec.get_end() + spec.get_start())
                dl = spec.get_end() - spec.get_start()
                lrange = [l0 - 0.4*dl, l0 + 0.4*dl]
            kfwhm_edges = []
            for k in [0,-1]:
                ima = moffat(ins['spaxel_size'], fwhm.data[k], beta, oversamp=oversamp)
                ima.oversamp = oversamp
                kfwhm_edges.append(self.optimum_circular_aperture(ins, flux, ima, spec, lrange=lrange))
            self.logger.debug('Optimum values of kfwhm at wavelengths edges: %s', kfwhm_edges)

        # loop on wavelength
        frac_ima = spec.copy()
        size_ima = spec.copy()
        waves = spec.wave.coord()
        nspaxels = np.zeros(len(waves), dtype=int)
        if lbin > 1:
            klist = np.linspace(0, len(waves)-1, lbin, dtype=int)
        else:
            klist = range(len(waves))
        self.logger.debug('Computing frac and nspaxels for %d wavelengths (lbin %d)', len(klist), lbin)
        for k in klist:
            wave = waves[k]
            ima = moffat(ins['spaxel_size'], fwhm.data[k], beta, oversamp=oversamp)
            ima.oversamp = oversamp
            if ins['type'] == 'IFS':
                if obs['ima_aperture_type'] == 'square_fixed':
                    tima,nspa,size,frac = self.square_aperture(ins, ima, obs['ima_aperture_hsize_spaxels'])
                elif obs['ima_aperture_type'] == 'circular_adaptative':
                    kfwhm = np.interp(wave, [waves[0],waves[-1]], kfwhm_edges)
                    tima,nspa,size,frac = self.adaptative_circular_aperture(ins, ima, kfwhm)
            elif ins['type'] == 'MOS':
                tima,nspa,size,frac  = self.fixed_circular_aperture(ins, ima, 0.5*ins['aperture'])
            frac_ima.data[k] = frac
            size_ima.data[k] = size
            nspaxels[k] = nspa
        if lbin > 1:
            self.logger.debug('Performing interpolation')
            ksel = nspaxels == 0
            for k in np.arange(0, len(waves))[ksel]:
                frac_ima.data[k] = np.interp(waves[k],waves[~ksel],frac_ima.data[~ksel])
                size_ima.data[k] = np.interp(waves[k],waves[~ksel],size_ima.data[~ksel])
                nspaxels[k] = int(np.interp(waves[k],waves[~ksel],nspaxels[~ksel]) + 0.5)

        for k in [0,-1]:
            self.logger.debug('At %.1f A  FWHM: %.2f Flux fraction: %.2f Aperture: %.1f Nspaxels: %d',waves[k],fwhm.data[k],frac_ima.data[k],
                          size_ima.data[k], nspaxels[k])
        obs['ima_type'] = 'ps' # switch back to ps

        return frac_ima,size_ima,np.array(nspaxels)

    # # # For the IFS for now
    # # # the Saturation computation is done computing only one image for every wavelength using the 
    # # # SNR target wavelength
    def time_from_source(self, ins, ima, spec, debug=True, sat=True, dit=False):
            """ main routine to perform the NDIT computation for a given source

            Parameters
            ----------
            ins : dict
                instrument (eg self.ifs['blue'] or self.moslr['red'])
            ima : MPDAF image
                source image, can be None for surface brightness source or point source
            spec : MPDAF spectrum
                source spectrum
            debug :
                if True print some info in logger.debug mode (Default value = True)
            sat : bool
                if True compute the NDIT to avoid saturation (Default value = True)
            dit : bool
                if True compute the DIT to achieve the target SNR (Default value = False)
                if False compute the NDIT to achieve the target SNR
            Returns
            -------
            dict
                result dictionary (see documentation)
            """

            start_time = time.time()
            
            obs = self.obs

            # # # Convert the quantities to the correct units, flux percentage
            flux = 1
            tflux = flux

            if obs['spec_type'] == 'cont':
                tflux *= ins['dlbda']
            
            if obs['ima_type'] == 'sb':
                tflux *= ins['spaxel_size']**2
            
            # # # LSF convolution
            spec = spec.filter(width=ins['lsfpix'])

            usedobs = {}

            if dit:
                _checkobs(obs, usedobs, ['moon','ndit','snr','airmass','spec_type','ima_type'])
                if obs['ndit'] is None:
                    raise ValueError('obs ndit cannot be None for dit computation')
            else:
                _checkobs(obs, usedobs, ['moon','dnit','snr','airmass','spec_type','ima_type'])
                if obs['dit'] is None:
                    raise ValueError('obs dit cannot be None for ndit computation')

            moon = obs['moon']

            # the snr we want to achieve
            target_snr = obs['snr']

            if obs['spec_type'] == 'line':
                target_wave = obs['wave_center'] 
            else:
                _checkobs(obs, usedobs, ['snr_wave'])
                target_wave = obs['snr_wave']
            
            wave_idx = spec.wave.pixel(target_wave, nearest=True)

            # # # # we take the spec value only at the target wavelength
            spec_value = spec.data.data[wave_idx]
            wave_value = spec.wave.coord()[wave_idx]

            if (ima is None):
                if (obs['ima_type'] != 'ps') and (obs['ima_type'] != 'sb'):
                    raise ValueError('ima can be None only for ps or sb obs ima_type')

                if (obs['ima_type'] == 'ps'):
                    fwhm = get_seeing_fwhm(obs['seeing'], obs['airmass'], wave_value, self.tel['diameter'], ins['iq_fwhm_tel'], ins['iq_fwhm_ins'])[0]
                    ima = moffat(ins['spaxel_size'], fwhm, ins['iq_beta'], oversamp=10)
                    ima.oversamp = 10
                    if ins['type'] == 'IFS':
                        if obs['ima_aperture_type'] == 'square_fixed':
                            tima,nspaxels,size,frac_flux = self.square_aperture(ins, ima, obs['ima_aperture_hsize_spaxels'])  
                        else:
                            print('No valid aperture type for IFS point source')
                    elif ins['type'] == 'MOS':
                        print('Not available for MOS configurations')

                elif (obs['ima_type'] == 'sb'):
                    if ins['type'] == 'IFS':
                        if obs['ima_aperture_type'] == 'square_fixed':
                            _checkobs(obs, usedobs, ['ima_area'])
                            nspaxels = int(obs['ima_area']/ins['spaxel_size']**2 + 0.5)
                        else:
                            print('No valid aperture type for IFS sb source')
                    elif ins['type'] == 'MOS':
                        print('Not available for MOS configurations')

            else:
                if (obs['ima_type'] != 'resolved'):
                    raise ValueError('ima cannot be None for resolved case')
                
                if (obs['ima_type'] == 'resolved'):
                    psf = self.get_image_psf(ins, wave_value)

                    # # # # copy the WCS from a dummy Moffat since the Sersic does not have it
                    dummy = moffat_image(fwhm=(1,1), n=10, shape=ima.shape, flux=1.0, unit_fwhm=None)

                    # here we do the convolution
                    conv_ima = ima.copy()
                    conv_ima.data = fftconvolve(ima.data, psf.data, mode='same')
                    conv_ima.data /= conv_ima.data.sum()
                    conv_ima.wcs = dummy.wcs
                    conv_ima.oversamp = psf.oversamp

                    if ins['type'] == 'IFS':
                        if obs['ima_aperture_type'] == 'square_fixed':
                            tima,nspaxels,size_ima,frac_flux = self.square_aperture(ins, conv_ima, obs['ima_aperture_hsize_spaxels'])
                        else:
                            print('No valid aperture type for IFS resolved source')
                    elif ins['type'] == 'MOS':
                        print('Not available for MOS configurations')


            # # # # we compute the sky
            if obs['skycalc']:
                sky_emi,sky_abs = obs['skyemi'], obs['skyabs']
            else:
                sky_emi,sky_abs = self.get_sky(ins, moon)

            ins_sky = sky_emi.data.data[wave_idx]
            ins_ins = ins['instrans'].data.data[wave_idx]
            ins_atm = sky_abs.data.data[wave_idx]


            spaxel_area = ins['spaxel_size']**2
            dl = ins['dlbda']

            w = wave_value

            tel_eff_area = self.tel['area'] * (1 - ins.get('obscuration', 0)) # telescope effective area
            a = (w*1.e-8/(H_cgs*C_cgs)) * (tel_eff_area*1.e4) * (ins_atm)
            Kt =  ins_ins * a

            # # # # source in the two cases
            # if dit == True means we want to compute the DIT
            if dit:
                if  obs['ima_type'] == 'sb':
                    nph_source = obs['ndit'] * tflux * nspaxels * spec_value * Kt
                else:
                    nph_source = obs['ndit'] * tflux * frac_flux * spec_value * Kt

                source_noise_sq = nph_source
            
                # # # sky
                nph_sky = obs['ndit'] * ins_sky * ins_ins * tel_eff_area * spaxel_area * nspaxels * (dl/1e4)
                sky_noise_sq = nph_sky

                # # # # ron and dark
                ron_noise_sq = ins['ron']**2*nspaxels*obs['ndit']
                dark_noise_sq = ins['dcurrent']*nspaxels*obs['ndit']/3600

                # # # # we compute the DIT with the root finding
                res = {}
                def equation(x):
                    return x - (target_snr**2 * (ron_noise_sq/x + dark_noise_sq + 
                                   sky_noise_sq + source_noise_sq) / nph_source**2)

                result = root_scalar(equation, bracket=[0.1, 9999], method='brentq')

                if not result.converged:
                    raise RuntimeError("DIT calculation failed to converge")
        
                res['dit'] = result.root
                # # # # #

                print(f"Computed DIT: {res['dit']:.2f} > {np.ceil(res['dit']):.2f} for target SNR: {target_snr:.2f} at wavelength {target_wave:.2f} A, in {nspaxels} spaxels")
            
                res['dit'] = np.ceil(res['dit'])  # round up to the nearest integer
            
                #update it also in the observation dictionary to use it later if you want to compute the SNR
                obs['dit'] = res['dit']

            # if dit == False means we want to compute the NDIT, default
            elif not dit:
                print('Computing DIT for target SNR:', target_snr)
                if  obs['ima_type'] == 'sb':
                    nph_source = obs['dit'] * tflux * nspaxels * spec_value * Kt
                else:
                    nph_source = obs['dit'] * tflux * frac_flux * spec_value * Kt

                source_noise_sq = nph_source
            
                # # # sky
                nph_sky = obs['dit'] * ins_sky * ins_ins * tel_eff_area * spaxel_area * nspaxels * (dl/1e4)
                sky_noise_sq = nph_sky

                # # # # ron and dark
                ron_noise_sq = ins['ron']**2*nspaxels 
                dark_noise_sq = ins['dcurrent']*nspaxels*obs['dit']/3600

                # # # # total noise
                tot_noise_sq = ron_noise_sq + dark_noise_sq + sky_noise_sq + source_noise_sq
            
                res = {}
                res['ndit'] = target_snr**2 * tot_noise_sq / nph_source**2

                print(f"Computed NDIT: {res['ndit']:.2f} > {np.ceil(res['ndit']):.2f} for target SNR: {target_snr:.2f} at wavelength {target_wave:.2f} A, in {nspaxels} spaxels")
            
                res['ndit'] = np.ceil(res['ndit'])  # round up to the nearest integer

                #update it also in the observation dictionary to use it later if you want to compute the SNR
                obs['ndit'] = res['ndit']
            
            
            # # # # Computing the fraction of saturated pixels
            if sat:
                len_waves = len(spec.wave.coord())
                print('% % % Computing fraction of saturated pixels % % %')
                frac_sat = 0

                #we take all the sky
                ins_sky_tot = sky_emi.subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
                ins_ins_tot = ins['instrans'].subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
                ins_atm_tot = sky_abs.subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())

                w = spec.wave.coord()

                a_tot = (w*1.e-8/(H_cgs*C_cgs)) * (tel_eff_area*1.e4) * (ins_atm_tot.data)
                Kt_tot =  ins_ins_tot.data * a_tot
                nph_sky_spaxel = obs['dit'] * ins_sky_tot.data * ins_ins_tot.data * tel_eff_area * spaxel_area * (dl/1e4)

                #equal in every spaxel, we just compute it once
                if obs['ima_type'] == 'sb':
                    nph_source_spaxel = obs['dit'] * tflux * spec.data * Kt_tot
                    tot_counts = nph_source_spaxel + nph_sky_spaxel + ins['ron'] + ins['dcurrent'] * obs['dit']/3600
                    frac_sat = np.sum(tot_counts > sat_level) / len_waves
                else:
                    nph_source = obs['dit'] * tflux * tima.data * spec.data[:, np.newaxis, np.newaxis] * Kt_tot[:, np.newaxis, np.newaxis]
                    tot_counts = nph_source + (nph_sky_spaxel[:, np.newaxis, np.newaxis] + ins['ron'] + ins['dcurrent'] * obs['dit']/3600) * nspaxels
                    frac_sat = np.sum(tot_counts > sat_level) / (len_waves * nspaxels)
                
                print(f"Fraction of saturated voxels: {frac_sat*100:.1f}% for saturation level {sat_level:.2f} counts")
                res['frac_sat'] = frac_sat

            if debug:
                self.logger.debug('Source type %s & %s',
                            obs['ima_type'], 
                            obs['spec_type'])

            end_time = time.time()
            elapsed = end_time - start_time 
            print(f"Time elapsed: {elapsed:.2f} seconds") 

            # We just copy in the result dictionary the obs values in the opposite case
            if 'ndit' not in res:
                res['ndit'] = obs['ndit']
            if 'dit' not in res:
                res['dit'] = obs['dit']

            return res

    def snr_from_source(self, ins, ima, spec, loop=False, debug=True):
        """ main routine to perform the S/N computation for a given source

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image, can be None for surface brightness source or point source
        spec : MPDAF spectrum
            source spectrum
        loop : bool
             set to True for multiple call (used only in ps and cont, Default value = False)
        debug :
             if True print some info in logger.debug mode (Default value = True)

        Returns
        -------
        dict
            result dictionary (see documentation)

         """
        start_time = time.time()
        global g_frac_ima,g_size_ima,g_nspaxels

        # # # LSF convolution
        spec = spec.filter(width=ins['lsfpix'])

        obs = self.obs
        usedobs = {}
        _checkobs(obs, usedobs, ['moon','dit','ndit','airmass','spec_type','ima_type'])
        flux = 1
        tflux = flux
        is_cont = False
        is_line = False
        if obs['spec_type'] == 'cont': # flux is in erg/s/cm2/A
            is_cont = True
            # convert flux if type is cont
            tspec = spec.copy()
            tflux *= ins['dlbda'] # flux in erg/s/cm2/spectel
            nspectels = 1
            frac_spec = 1
            size_spec = spec.get_end() - spec.get_start()
            waves = tspec.wave.coord()
        elif obs['spec_type'] == 'line': #flux in erg/s/cm2
            _checkobs(obs, usedobs, ['spec_range_type'])
            is_line = True
            # truncate spectrum if type is line
            if obs['spec_range_type'] == 'fixed':
                _checkobs(obs, usedobs, ['spec_range_hsize_spectels'])
                tspec,waves,nspectels,size_spec,frac_spec = self.truncate_spec_fixed(ins, spec, obs['spec_range_hsize_spectels'])
            elif obs['spec_range_type'] == 'adaptative':
                # adaptive truncation of spectrum
                _checkobs(obs, usedobs, ['spec_range_kfwhm'])
                tspec,waves,nspectels,size_spec,frac_spec = self.truncate_spec_adaptative(ins, spec, obs['spec_range_kfwhm'])
            else:
                raise ValueError('Unknown spec_range_type')


        is_ps = False
        is_sb = False
        if (ima is None) :
            if (obs['ima_type'] != 'ps') and (obs['ima_type'] != 'sb'):
                raise ValueError('ima can be None only for ps or sb obs ima_type')

            if (obs['ima_type'] == 'ps'):
                # use the seeing PSF to compute frac_ima and nspaxels evolution with wavelength
                if is_cont:
                    if loop:
                        frac_ima,size_ima,nspaxels = g_frac_ima,g_size_ima,g_nspaxels
                    else:
                        lbin = 20 if spec.shape[0]>100 else 1
                        frac_ima,size_ima,nspaxels = self.get_psf_frac_ima(ins, flux, spec, lbin=lbin)
                    frac_flux = frac_ima.copy()
                    frac_flux.data = frac_ima.data*frac_spec
                if is_line:
                    _checkobs(obs, usedobs, ['seeing'])
                    # use a constant PSF computed at the central wavelength the emission line
                    #l0 = 0.5*(tspec.get_end() + tspec.get_start())
                    # # # modified to take the actual central wavelength of the line regardless of the cut
                    l0 = obs['wave_center']
                    fwhm = get_seeing_fwhm(obs['seeing'], obs['airmass'], l0, self.tel['diameter'], ins['iq_fwhm_tel'], ins['iq_fwhm_ins'])[0]
                    ima = moffat(ins['spaxel_size'], fwhm, ins['iq_beta'], oversamp=10)
                    ima.oversamp = 10
                    if debug:
                        self.logger.debug('Computing PSF at %.1f fwhm %.2f beta %.1f',l0,fwhm,ins['iq_beta'])
                    obs['ima_type'] = 'resolved' # change temporarily ima_type
                is_ps = True

            if (obs['ima_type'] == 'sb') :
                # surface brightness
                frac_ima = 1
                tflux *= ins['spaxel_size']**2 #erg.s-1.cm-2/voxels
                if ins['type'] == 'IFS':
                    _checkobs(obs, usedobs, ['ima_area'])
                    nspaxels = int(obs['ima_area']/ins['spaxel_size']**2 + 0.5)
                    size_ima = np.sqrt(obs['ima_area']) # assuming square area
                    area_aper = obs['ima_area']
                elif ins['type'] == 'MOS':
                    nspaxels = int((2*np.pi*ins['aperture']**2/4)/ins['spaxel_size']**2 + 0.5)
                    size_ima = ins['aperture']
                    area_aper = np.pi*size_ima**2/4
                frac_flux = frac_ima*frac_spec

                is_sb = True

        # # # # # we keep the original resolved for the point_source when is the line
        if (obs['ima_type'] == 'resolved') and (is_ps):
            # truncate image if type is resolved
            if ima is None:
                raise ValueError('image cannot be none for resolved image type')
            if ins['type'] == 'IFS':
                _checkobs(obs, usedobs, ['ima_aperture_type'])
                if obs['ima_aperture_type'] == 'square_fixed':
                    _checkobs(obs, usedobs, ['ima_aperture_hsize_spaxels'])
                    tima,nspaxels,size_ima,frac_ima = self.square_aperture(ins, ima, obs['ima_aperture_hsize_spaxels'])
                    area_aper = size_ima**2
                elif obs['ima_aperture_type'] == 'circular_adaptative':
                    _checkobs(obs, usedobs, ['ima_kfwhm'])
                    # adaptive truncation of image
                    tima,nspaxels,size_ima,frac_ima = self.adaptative_circular_aperture(ins, ima, obs['ima_kfwhm'])
                    area_aper = np.pi*size_ima**2/4
                    if debug:
                        self.logger.debug('Adaptive circular aperture diameter %.2f frac_flux %.2f',size_ima,frac_ima)
                else:
                    raise ValueError(f"unknown ima_aperture_type {obs['ima_aperture_type']}")
            elif ins['type'] == 'MOS':
                size_ima = ins['aperture']
                tima,nspaxels,size_ima,frac_ima  = self.fixed_circular_aperture(ins, ima, 0.5*ins['aperture'])
                area_aper = np.pi*size_ima**2/4
            frac_flux = frac_ima*frac_spec
            tima_cube = tima #fictious cube with a single wavelength to use the snr_from_cube method

        # # # # # mofified resolved case
        # # # # # CONVOLUTION WITH VARIABLE PSF
        if (obs['ima_type'] == 'resolved') and not (is_ps):
            # truncate image if type is resolved
            if ima is None:
                raise ValueError('image cannot be none for resolved image type')

            # # # # Get wavelength array of the cutted (or not cutted) spectrum
            waves = tspec.wave.coord()
            n_waves = len(waves)

            # # # # Create arrays for interpolation points
            # # # # # #
            N_wave = 20
            # # # # # #
            
            interp_indices = np.linspace(0, n_waves-1, N_wave, dtype=int)
            interp_waves = waves[interp_indices]

            # # # # Initialize arrays to store results
            tima_arr = []
            nspaxels_arr = []
            size_ima_arr = []
            frac_ima_arr = []

            # # # # copy the WCS from a dummy Moffat since the Sersic does not have it
            dummy = moffat_image(fwhm=(1,1), n=10, shape=ima.shape, flux=1.0, unit_fwhm=None)

            # # # # Compute convolution at interpolation points
            for wave in interp_waves:
                # Get PSF at this wavelength
                psf = self.get_image_psf(ins, wave)

                conv_ima = ima.copy()
                conv_ima.data = fftconvolve(ima.data, psf.data, mode='same')
                conv_ima.data /= conv_ima.data.sum()
                conv_ima.wcs = dummy.wcs
                conv_ima.oversamp = psf.oversamp

                if ins['type'] == 'IFS':
                    _checkobs(obs, usedobs, ['ima_aperture_type'])
                    if obs['ima_aperture_type'] == 'square_fixed':
                        _checkobs(obs, usedobs, ['ima_aperture_hsize_spaxels'])
                        tima,nspaxels,size_ima,frac_ima = self.square_aperture(ins, conv_ima, obs['ima_aperture_hsize_spaxels'])
                        area_aper = size_ima**2
                    elif obs['ima_aperture_type'] == 'circular_adaptative':
                        _checkobs(obs, usedobs, ['ima_kfwhm'])
                        # adaptive truncation of image
                        tima,nspaxels,size_ima,frac_ima = self.adaptative_circular_aperture(ins, conv_ima, obs['ima_kfwhm'])
                        area_aper = np.pi*size_ima**2/4
                        if debug:
                            self.logger.debug('Adaptive circular aperture diameter %.2f frac_flux %.2f',size_ima,frac_ima)
                    else:
                        raise ValueError(f"unknown ima_aperture_type {obs['ima_aperture_type']}")
                elif ins['type'] == 'MOS':
                    size_ima = ins['aperture']
                    tima,nspaxels,size_ima,frac_ima = self.fixed_circular_aperture(ins, conv_ima, 0.5*ins['aperture'])
                    area_aper = np.pi*size_ima**2/4

                # # # # Store results
                tima_arr.append(tima)
                nspaxels_arr.append(nspaxels)
                size_ima_arr.append(size_ima)
                frac_ima_arr.append(frac_ima)

            # # # # Interpolate results for all wavelengths
            nspaxels = np.interp(waves, interp_waves, nspaxels_arr)
            size_ima = np.interp(waves, interp_waves, size_ima_arr)
            frac_ima = np.interp(waves, interp_waves, frac_ima_arr)

            # # # # For tima we need to interpolate the data arrays
            tima = tima_arr[0].copy()  # Use first one as template
            wave_coord = WaveCoord(cdelt=waves[1]-waves[0], crval=waves[0])

            # Reshape arrays per operazioni vettorizzate
            pixel_values = np.array([t.data.ravel() for t in tima_arr])  # shape: (n_waves, n_pixels)

            # Vector interpolation
            tima_data = np.zeros((len(waves), tima.shape[0] * tima.shape[1]))
            for i in range(tima_data.shape[1]):
                tima_data[:,i] = np.interp(waves, interp_waves, pixel_values[:,i])

            # Reshape
            tima_data = tima_data.reshape((len(waves),) + tima.shape)

            # # # # Create a new Cube with the interpolated data
            tima_cube = Cube(data=tima_data, 
            wave=wave_coord,
            wcs=tima.wcs)

            frac_flux = frac_ima*frac_spec

        # perform snr computation
        if isinstance(tspec, (float)):
            ima_data = tflux * tima * tspec
            ima_data.data.mask = tima.data.mask
            res = self.snr_from_ima(ins, ima_data, waves)
        elif is_sb:
            spec_data = tflux * tspec
            res = self.snr_from_spec(ins, spec_data)
        elif is_ps and is_cont:
            spec_data = tflux * tspec
            res = self.snr_from_ps_spec(ins, spec_data, frac_ima, nspaxels)
            res['spec']['frac_spec'] = frac_spec
        else:
            cube_data = tflux * tima_cube * tspec
            res = self.snr_from_cube(ins, cube_data)

        # compute additionnal results
        if (not is_ps) or (obs['ima_type'] == 'resolved'):
            resc = res['cube']
            nvoxels = nspaxels*nspectels
            dl = ins['dlbda']
            vartot = resc['noise']['tot'].copy()
            vartot.data = vartot.data**2
            # sum over spatial axis to get spectra values res['spec']
            nph_sky = resc['nph_sky']*nspaxels
            tot_noise = nph_sky.copy()
            if obs['ima_type'] == 'resolved':
                nph_source = resc['nph_source'].sum(axis=(1,2))
                tot_noise.data = np.sqrt(vartot.sum(axis=(1,2)).data)
            elif obs['ima_type'] == 'sb':
                nph_source = resc['nph_source']*nspaxels
                tot_noise.data = np.sqrt(vartot.data*nspaxels)
            snr = nph_sky.copy()
            snr.data = nph_source.data / tot_noise.data
            sky_noise = nph_sky.copy()
            sky_noise.data = np.sqrt(nph_sky.data)
            source_noise = nph_source.copy()
            source_noise.data = np.sqrt(nph_source.data) 
            detnoise = nspaxels*(resc['noise']['ron']**2+resc['noise']['dark']**2) / tot_noise.data.data**2            
            res['spec'] = dict(snr=snr,
                               snr_mean=snr.data.mean(),
                               snr_med=np.median(snr.data.data),
                               snr_max=snr.data.max(),
                               snr_min=snr.data.min(),
                               frac_flux=frac_flux,
                               frac_ima=frac_ima,
                               frac_spec=frac_spec,
                               nb_spaxels=nspaxels,
                               nph_source=nph_source,
                               nph_sky=nph_sky,
                               noise = dict(ron=np.sqrt(nspaxels)*resc['noise']['ron'],
                                            dark=np.sqrt(nspaxels)*resc['noise']['dark'],
                                            sky=sky_noise,
                                            source=source_noise,
                                            tot=tot_noise,
                                            frac_detnoise_mean=np.mean(detnoise),
                                            frac_detnoise_max=np.max(detnoise),
                                            frac_detnoise_min=np.min(detnoise),
                                            frac_detnoise_med=np.median(detnoise),
                                            frac_detnoise_std=np.std(detnoise),
                                            )
                           )
        # if spec type is line summed over spectral axis to get aperture values res['aper]
        if obs['spec_type'] == 'line':
            sp = res['spec']
            nph_source_aper = sp['nph_source'].data.sum()
            nph_sky_aper = sp['nph_sky'].data.sum()
            tot_noise_aper = np.sqrt(np.sum(sp['noise']['tot'].data**2))
            snr_aper = nph_source_aper/tot_noise_aper
            sky_noise_aper = np.sqrt(np.sum(sp['noise']['sky'].data**2))
            source_noise_aper = np.sqrt(np.sum(sp['noise']['source'].data**2))
            ron_aper = np.sqrt(nvoxels)*resc['noise']['ron']
            dark_aper = np.sqrt(nvoxels)*resc['noise']['dark']
            res['aper'] = dict(snr=snr_aper,
                               size=size_ima,
                               area=area_aper,
                               frac_fluxe=frac_flux,
                               frac_ima=frac_ima,
                               frac_spec=frac_spec,
                               nb_spaxels=nspaxels,
                               nb_spectels=nspectels,
                               nb_voxels=nvoxels,
                               nph_source=nph_source_aper,
                               nph_sky=nph_sky_aper,
                               ron=ron_aper,
                               dark=dark_aper,
                               sky_noise=sky_noise_aper,
                               source_noise=source_noise_aper,
                               tot_noise=tot_noise_aper,
                               frac_detnoise=(ron_aper**2+dark_aper**2)/tot_noise_aper**2,
                               )
        if debug:
            self.logger.debug('Source type %s & %s Flux %.2e S/N %.1f FracFlux %.3f Nspaxels %d Nspectels %d',
                            obs['ima_type'], 
                            obs['spec_type'], 
                            flux,
                            snr_aper if 'aper' in res.keys() else res['spec']['snr_mean'],
                            # For resolved case, use mean of frac_flux
                            float(res['spec']['frac_ima'].mean()[0] if is_ps and is_cont else np.mean(frac_flux)),
                            # Convert nspaxels to int, taking mean if array
                            int(np.mean(res['spec']['nb_spaxels'])+0.5) if is_ps and is_cont else int(np.mean(nspaxels)),
                            int(nspectels))

        res['input']['dl'] = ins['dlbda']
        res['input']['flux'] = flux
        _copy_obs(usedobs, res['input'])
        if ima is not None:
            res['cube']['trunc_ima'] = tima
        res['cube']['trunc_spec'] = tspec
        if is_ps and is_line: # set ima_type back to ps
            obs['ima_type'] = 'ps'
        
        # # # # compute rebinning of the spectrum if needed
        if 'spbin' in obs and obs['spbin'] > 1:
            res['spec']['snr_rebin'] = rebin_spectrum(res['spec']['nph_source'], res['spec']['noise']['tot'], obs['spbin'])
    
        # Update globals added JB
        g_frac_ima,g_size_ima,g_nspaxels = frac_ima,size_ima,nspaxels

        end_time = time.time()
        elapsed = end_time - start_time 
        print(f"Time elapsed: {elapsed:.2f} seconds")  

        return res

    def snr_from_cube(self, ins, cube):
        """ compute S/N from a data cube

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        cube : MPDAF cube
            source data cube in flux/voxels

        Returns
        -------
        dict
            result dictionary

            this routine is called by snr_from_source

        """

        obs = self.obs
        moon = obs['moon']
        # added the computation of the sky from the skycalc instead of local database
        if obs['skycalc']:
            sky_emi,sky_abs = obs['skyemi'], obs['skyabs']
        else:
            sky_emi,sky_abs = self.get_sky(ins, moon)
        # truncate instrans and sky to the wavelength limit of the input cube
        ins_sky = sky_emi.subspec(lmin=cube.wave.get_start(), lmax=cube.wave.get_end())
        ins_ins = ins['instrans'].subspec(lmin=cube.wave.get_start(), lmax=cube.wave.get_end())
        ins_atm = sky_abs.subspec(lmin=cube.wave.get_start(), lmax=cube.wave.get_end())
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area
        dl = cube.wave.get_step(unit='Angstrom')
        w = cube.wave.coord() # wavelength in A
        tel_eff_area = self.tel['area'] * (1 - ins.get('obscuration', 0)) # telescope effective area
        a = (w*1.e-8/(H_cgs*C_cgs)) * (tel_eff_area*1.e4) * (ins_atm.data)
        Kt =  ins_ins * a
        nph_source = cube.copy()
        nph_source.data = obs['dit'] * obs['ndit'] * Kt[:,np.newaxis,np.newaxis].data * cube.data # number of photons received from the source
        nph_sky = ins_sky * ins_ins * obs['dit'] * obs['ndit'] * tel_eff_area * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = 1
        ron_noise = np.sqrt(ins['ron']**2*nb_voxels*obs['ndit'])
        dark_noise = np.sqrt(ins['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)
        sky_noise = np.sqrt(nph_sky)
        source_noise = cube.copy()
        source_noise.data = np.sqrt(nph_source.data)
        skynoise_cube = np.tile(sky_noise.data[:,np.newaxis,np.newaxis], (1,cube.shape[1],cube.shape[2]))
        tot_noise = cube.copy()
        tot_noise.data = np.sqrt(ron_noise**2 + dark_noise**2 + skynoise_cube.data**2 + source_noise.data**2)
        ksel = cube.data == 0
        tot_noise.data[ksel] = 0
        snr = cube.copy()
        snr.data = nph_source.data / tot_noise.data
        res = {}
        res['cube'] = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, tot=tot_noise),
                   snr=snr, nph_source=nph_source, nph_sky=nph_sky)
        res['input'] = dict(flux_source=cube, atm_abs=ins_atm, ins_trans=ins_ins, atm_emi=ins_sky)
        return res


    def snr_from_ima(self, ins, ima, wave):
        """ compute S/N from an image

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image in flux/spaxels
        wave : float
            wavelength in A

        Returns
        -------
        dict
            result dictionary

            this routine is called by snr_from_source
        """
        obs = self.obs
        moon = obs['moon']
        # get instrans and sky tvalue at the given wavelength
        ins_sky = ins[moon].data[ins[moon].wave.pixel(wave, nearest=True)]
        ins_ins = ins['instrans'].data[ins['instrans'].wave.pixel(wave, nearest=True)]
        ins_atm = ins['atmtrans'].data[ins['atmtrans'].wave.pixel(wave, nearest=True)]
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area
        dl = ins['instrans'].wave.get_step(unit='Angstrom')
        tel_eff_area = self.tel['area'] * (1 - ins.get('obscuration', 0)) # telescope effective area
        a = (wave*1.e-8/(H_cgs*C_cgs)) * (tel_eff_area*1.e4) * (ins_atm)
        Kt =  ins_ins * a
        nph_source = ima.copy()
        nph_source.data = obs['dit'] * obs['ndit'] * Kt * ima.data # number of photons received from the source
        nph_sky = ima.copy()
        nph_sky.data[:,:] = ins_sky * ins_ins * obs['dit'] * obs['ndit'] * tel_eff_area * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = 1
        ron_noise = np.sqrt(ins['ron']**2*nb_voxels*obs['ndit'])
        dark_noise = np.sqrt(ins['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)
        sky_noise = nph_sky.copy()
        sky_noise.data = np.sqrt(nph_sky.data)
        source_noise = ima.copy()
        source_noise.data = np.sqrt(nph_source.data)
        tot_noise = ima.copy()
        tot_noise.data = np.sqrt(ron_noise**2 + dark_noise**2 + sky_noise.data**2 + source_noise.data**2)
        ksel = ima.data == 0
        tot_noise.data[ksel] = 0
        snr = ima.copy()
        snr.data = nph_source.data / tot_noise.data
        res = {}
        res['cube'] = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, tot=tot_noise),
                   snr=snr, nph_source=nph_source, nph_sky=nph_sky)
        res['input'] = dict(flux_source=ima, atm_abs=ins_atm, ins_trans=ins_ins, atm_emi=ins_sky)
        return res

    def snr_from_spec(self, ins, spec):
        """compute S/N from a spectrum in flux/spectel

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        spec : MPDAF spectrum
            source spectrum in flux/spectel

        Returns
        -------
        dict
            result dictionary

            this routine is called by snr_from_source


        """
        obs = self.obs
        moon = obs['moon']
        # added the computation of the sky from the skycalc instead of local database
        if obs['skycalc']:
            sky_emi,sky_abs = obs['skyemi'], obs['skyabs']
        else:
            sky_emi,sky_abs = self.get_sky(ins, moon)
        # truncate instrans and sky to the wavelength limit of the input cube
        ins_sky = sky_emi.subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
        ins_ins = ins['instrans'].subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
        ins_atm = sky_abs.subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area
        dl = spec.wave.get_step(unit='Angstrom')
        w = spec.wave.coord() # wavelength in A
        tel_eff_area = self.tel['area'] * (1 - ins.get('obscuration', 0)) # telescope effective area
        a = (w*1.e-8/(H_cgs*C_cgs)) * (tel_eff_area*1.e4) * (ins_atm.data)
        Kt =  ins_ins * a
        nph_source = spec.copy()
        nph_source.data = obs['dit'] * obs['ndit'] * Kt.data * spec.data # number of photons received from the source
        nph_sky = ins_sky * ins_ins * obs['dit'] * obs['ndit'] * tel_eff_area * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = 1
        ron_noise = np.sqrt(ins['ron']**2*nb_voxels*obs['ndit'])
        dark_noise = np.sqrt(ins['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)
        sky_noise = nph_sky.copy()
        sky_noise.data = np.sqrt(nph_sky.data)
        source_noise = spec.copy()
        source_noise.data = np.sqrt(nph_source.data)
        tot_noise = spec.copy()
        tot_noise.data = np.sqrt(ron_noise**2 + dark_noise**2 + sky_noise.data**2 + source_noise.data**2)
        snr = spec.copy()
        snr.data = nph_source.data / tot_noise.data
        res = {}
        res['cube'] = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, tot=tot_noise),
                   snr=snr, nph_source=nph_source, nph_sky=nph_sky)
        res['input'] = dict(flux_source=spec, atm_abs=ins_atm, ins_trans=ins_ins, atm_emi=ins_sky)
        return res

    def snr_from_ps_spec(self, ins, spec, frac_ima, nspaxels):
        """compute S/N for a point source define by a spectrum in flux/spectel

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        spec : MPDAF spectrum
            point source spectrul in flux/spectel
        frac_ima : MPDAF spectrum
            flux fraction recovered in the aperture as function of wavelength
        nspaxels : numpy array of int
            corresponding number of spaxels in the aperture

        Returns
        -------
        dict
            result dictionary

            this routine is called by snr_from_source

        """
        obs = self.obs
        moon = obs['moon']
        # added the computation of the sky from the skycalc instead of local database
        if obs['skycalc']:
            sky_emi,sky_abs = obs['skyemi'], obs['skyabs']
        else:
            sky_emi,sky_abs = self.get_sky(ins, moon)
        # truncate instrans and sky to the wavelength limit of the input cube
        ins_sky = sky_emi.subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
        ins_ins = ins['instrans'].subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
        ins_atm = sky_abs.subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area
        dl = spec.wave.get_step(unit='Angstrom')
        w = spec.wave.coord() # wavelength in A
        tel_eff_area = self.tel['area'] * (1 - ins.get('obscuration', 0)) # telescope effective area
        a = (w*1.e-8/(H_cgs*C_cgs)) * (tel_eff_area*1.e4) * (ins_atm.data)
        Kt =  ins_ins * a
        nph_source = spec.copy()
        nph_source.data = obs['dit'] * obs['ndit'] * Kt.data * spec.data * frac_ima.data # number of photons received from the source
        nph_sky = ins_sky * ins_ins * obs['dit'] * obs['ndit'] * tel_eff_area * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = nspaxels
        nph_sky.data = nph_sky.data * nb_voxels # scale sky by number of spaxels
        ron_noise = spec.copy()
        ron_noise.data = np.sqrt(ins['ron']**2*nb_voxels*obs['ndit'])
        dark_noise = spec.copy()
        dark_noise.data = np.sqrt(ins['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)
        sky_noise = nph_sky.copy()
        sky_noise.data = np.sqrt(nph_sky.data)
        source_noise = spec.copy()
        source_noise.data = np.sqrt(nph_source.data)
        tot_noise = spec.copy()
        tot_noise.data = np.sqrt(ron_noise.data**2 + dark_noise.data**2 + sky_noise.data**2 + source_noise.data**2)
        snr = spec.copy()
        snr.data = nph_source.data / tot_noise.data 
        detnoise = (ron_noise.data.data**2 + dark_noise.data.data**2) / tot_noise.data.data**2   
        res = {}
        res['cube'] = {}
        res['input'] = dict(atm_abs=ins_atm, ins_trans=ins_ins, atm_emi=ins_sky, flux_source=spec)
        res['spec'] = dict(snr=snr,
                           snr_mean=snr.data.mean(),
                           snr_max=snr.data.max(),
                           snr_min=snr.data.min(),
                           snr_med=np.median(snr.data.data),
                           nph_source=nph_source,
                           nph_sky=nph_sky,
                           frac_ima=frac_ima,
                           nb_spaxels=nspaxels,
                           nb_voxels=nb_voxels,
                           nb_spectels=1,                           
                           noise = dict(
                               ron=ron_noise, 
                               dark=dark_noise, 
                               sky=sky_noise, 
                               source=source_noise, 
                               tot=tot_noise,
                               frac_detnoise_mean = np.mean(detnoise),
                               frac_detnoise_max = np.max(detnoise),
                               frac_detnoise_min = np.min(detnoise),
                               frac_detnoise_med = np.median(detnoise),
                               frac_detnoise_std = np.std(detnoise),                               
                               ),   
                           )        
        return res     
            
    def flux_from_source(self, ins, snr, ima, spec, snrcomp=None, flux=None, bracket=(0.1,100000)):
        """compute the flux needed to achieve a given S/N

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        flux : float
            flux value in erg/s/cm2 (for line), in erg/s/cm2/A (for cont), / arcsec2 (for sb)
        ima : MPDAF image
            source image, can be None for surface brightness source or point source
        spec : MPDAF spectrum
            source spectrum
        snrcomp : dict
             method and parameters to derive the S/N target value from the cont spectrum (Default value = None)
        flux :
             starting value of the flux (Default value = None)
        bracket : tuple of float
             interval of flux*1.e-20 for the zero finding routine (Default value = (0.1,100000) :


        Returns
        -------
        dict
            result dictionary (see documentation)

        """
        global g_frac_ima,g_size_ima,g_nspaxels
        if (self.obs['spec_type']=='cont'):
            # decode snrcomp
            if snrcomp is None:
                raise ValueError("snrcomp dict is mandatory for cont spec_type")
            if 'method' not in snrcomp.keys():
                raise ValueError('method is missing in snrcomp')
            if snrcomp['method'] == 'mean':
                if 'waves' not in snrcomp.keys():
                    raise ValueError('waves is missing in snrcomp')
                k1,k2 = spec.wave.pixel(snrcomp['waves'], nearest=True)
                krange = [k1,k2+1,'mean']
            elif snrcomp['method'] == 'sum':
                if 'wave' not in snrcomp.keys():
                    raise ValueError('wave is missing in snrcomp')
                k0 = spec.wave.pixel(snrcomp['wave'], nearest=True)
                if 'npix' not in snrcomp.keys():
                    raise ValueError('npix is missing in snrcomp')
                k1 = int(k0-0.5*snrcomp['npix']+0.5)
                k2 = k1 + snrcomp['npix']
                krange = [k1,k2,'sum']
            else:
                raise ValueError(f"unknown method {snrcomp['method']} in snrcomp")
        else:
            krange = None

        # compute frac_ima and nspaxels only once
        if flux is None:
            flux = 1.e-18
        if (self.obs['ima_type'] == 'ps') and (self.obs['spec_type']=='cont'):
            lbin = 20 if spec.shape[0]>100 else 1
            g_frac_ima,g_size_ima,g_nspaxels = self.get_psf_frac_ima(ins, flux, spec, lbin=lbin)
        res0 = root_scalar(self.fun, args=(snr, ins, ima, spec, krange),
                           method='brenth', bracket=bracket, xtol=1.e-3, maxiter=100)
        flux = res0.root*1.e-20
        res = self.snr_from_source(ins, flux, ima, spec)
        if krange is not None:
            snr1 = np.mean(res['spec']['snr'][krange[0]:krange[1]].data)
            res['spec']['snr_mean'] = snr1
            res['spec']['flux'] = flux
        else:
            snr1 = res['aper']['snr']
            res['aper']['flux'] = flux
        self.logger.debug('SN %.2f Flux %.2e Iter %d Fcall %d converged %s', snr1, flux, res0.iterations,
                          res0.function_calls, res0.converged)
        g_frac_ima,g_size_ima,g_nspaxels = None,None,None

        return res

    def fun(self, flux, snr0, ins, ima, spec, krange):
        """ minimizing function used by flux_from_source

        Parameters
        ----------
        flux : float
            flux value * 1.e-20
        snr0 : float
            target S/N
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image, can be None for surface brightness source or point source
        spec : MPDAF spectrum
            source spectrum
        krange : tuple of int
            wavelength range in spectel to compute the S/N

        Returns
        -------
        float
            S/N - target S/N
        """
        res = self.snr_from_source(ins, flux*1.e-20, ima, spec, loop=True, debug=False)
        if krange is not None:
            if krange[2] == 'mean':
                snr = res['spec']['snr_med']
            elif krange[2] == 'sum':
                snr = np.sum(res['spec']['nph_source'].data[krange[0]:krange[1]]) \
                    / np.sqrt(np.sum(res['spec']['noise']['tot'].data[krange[0]:krange[1]]**2))
        else:
            snr = res['aper']['snr']
        #print(f"flux {flux:.2f} snr {snr:.3f} snr0 {snr0:.1f} diff {snr-snr0:.5f}")
        return snr-snr0

    def get_image_quality(self, ins, spec=None):
        """ compute image quality evolution with wavelength

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        spec : MPDAF spectrum
            use wavelengths from spec, if None use all instrument wavelengths

        Returns
        -------
        numpy array of float
            image quality

        """
        obs = self.obs
        if spec is None:
            iq = ins['instrans'].copy()
        else:
            iq = spec.copy()
        iq.data = get_seeing_fwhm(obs['seeing'], obs['airmass'], iq.wave.coord(),
                                  self.tel['diameter'], ins['iq_fwhm_tel'], ins['iq_fwhm_ins'])[0]
        return iq

    def get_image_psf(self, ins, wave, oversamp=10):
        """ compute PSF image

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        wave : float
            wavelength in A
        oversamp : int
             oversampling factor (Default value = 10)

        Returns
        -------
        MPDAF image
            PSF image

        """
        fwhm = get_seeing_fwhm(self.obs['seeing'], self.obs['airmass'], wave,
                               self.tel['diameter'], ins['iq_fwhm_tel'], ins['iq_fwhm_ins'])[0]
        ima = moffat(ins['spaxel_size'], fwhm, ins['iq_beta'], oversamp=oversamp)
        ima.oversamp = oversamp
        return ima

    def print_aper(self, res, names):
        """ pretty print the apertures results for a set of results

        Parameters
        ----------
        res : dict or list of dict
            result dictionaries deliver by snr_from_source or flux_from_source
        names : str or list of str
            name to identify the result

        Returns
        -------
        astropy table
            table with one column by result
        """
        if not isinstance(res, (list)):
            res,names = [res],[names]
        tab = Table(names=['item']+names, dtype=(len(res)+1)*['S20'])

        for key in res[0]['aper'].keys():
            d = dict(item=key)
            if isinstance(res[0]['aper'][key], (float, np.float64)):
                for n,r in zip(names,res):
                    d[n] = f"{r['aper'][key]:5.4g}"
            else:
                for n,r in zip(names,res):
                    d[n] = f"{r['aper'][key]}"
            tab.add_row(d)
        return tab

    # # # # new methods for the WST - ETC
    ########## build the observation
    def build_obs_full(self, fo):
        """Build observation parameters and setup from input dictionary
    
        Parameters
        ----------
        fo : dict
            Dictionary containing observation parameters

        Returns
        -------
        tuple
            (CONF, obs, spec, ima, spec_input)
            - CONF: instrument configuration
            - obs: observation parameters
            - spec: processed spectrum
            - ima: source image (if resolved)
            - spec_input: input spectrum
        """
        # Get instrument configuration
        insfam = getattr(self, fo["INS"])
        CONF = insfam[fo["CH"]]
        
        # Determine spectral type
        if fo["SPEC"] == 'line':
            dummy_type = 'line'
        elif fo["SPEC"] in ('template', 'pl', 'bb'):
            dummy_type = 'cont'
            
        # Build observation dictionary    
        obs = dict(
            moon=fo["MOON"],
            airmass = fo["AM"],
            seeing = fo["SEE"],
            ndit = fo["NDIT"],
            dit = fo["DIT"],
            spec_type = dummy_type,
            spec_range_type = fo["SPEC_RANGE"],
            spec_range_kfwhm = fo["SPEC_KFWHM"],
            spec_range_hsize_spectels = fo["SPEC_HSIZE"],  
  
            ima_type = fo["OBJ"],
            ima_area = fo["IMA_AREA"],
            ima_aperture_type = fo["IMA_RANGE"],
            ima_kfwhm = fo["IMA_KFWHM"],
            ima_aperture_hsize_spaxels = fo["IMA_HSIZE"],
            skycalc = fo["SKYCALC"],
            wave_center = fo['WAVE_CENTER'],

            snr = fo["SNR"],
            snr_wave = fo["SNR_WAVE"]
        )
        
        # Compute sky if needed
        if fo["SKYCALC"]:
            obs["skyemi"], obs["skyabs"] = self.get_sky2(fo)
        
        # Add spectral binning if specified
        if ('SP_BIN' in fo) and isinstance(fo['SP_BIN'], int) and fo['SP_BIN'] > 1:
            obs["spbin"] = fo["SP_BIN"]
    
        self.set_obs(obs)
        
        # Get spectrum
        spec_input, spec = self.get_spec2(fo)
        
        # Handle resolved source image
        ima = None
        if fo['OBJ'] == 'resolved':
            dima = {
                'type': fo["IMA"],
                'fwhm': fo["IMA_FWHM"],
                'beta': fo["IMA_BETA"], 
                'n': fo["IMA_N"],
                'reff': fo["IMA_REFF"],
                'ell': fo["IMA_ELL"],
                'kfwhm': fo["IMA_KFWHM"],
                'kreff': fo["IMA_KREFF"],
            }
            ima = self.get_ima(CONF, dima)
        
        old_flux = 1
            
        # Optimize spectral range if needed
        if (fo["OPT_SPEC"]) & (fo["SPEC_RANGE"] == "adaptative") & (dummy_type == "line"):
            self.optimum_spectral_range(CONF, old_flux, ima, spec)

        # Optimize image aperture if needed
        if (fo["OPT_IMA"]) & (fo["IMA_RANGE"] == "circular_adaptative") & (fo["OBJ"] in ('ps', 'resolved')):
            A = CONF['lbda1']
            B = CONF['lbda2']
            f = fo['FRAC_SPEC_MEAN_OPT_IMAGE']

            delta = (1 - f) / 2
            AA = A + delta * (B - A)
            BB = B - delta * (B - A)
            
            self.optimum_circular_aperture(CONF, old_flux, ima, spec, lrange=[AA,BB])
    
        return CONF, obs, spec, ima, spec_input

    # # # # SPEC COMPUTATION 
    def get_spec2(self, fo):
        """Get and process spectrum based on input parameters
    
        Parameters
        ----------
        fo : dict
            Dictionary containing spectral parameters
        
        Returns
        -------
        tuple
            (spec_raw, spec_cut)
            - spec_raw: Original spectrum
            - spec_cut: Processed and trimmed spectrum
        """

        insfam = getattr(self, fo["INS"]) 
        CONF = insfam[fo["CH"]]
        
        lstep = CONF['instrans'].get_step()
        l1, l2 = CONF['instrans'].get_start(), CONF['instrans'].get_end()
            
        if fo['SPEC'] == 'template':
            name, DEFAULT_WAVE, flux = sed_models.template(f"{fo['TEMP_NAME']}.dat")
            redshift = fo['Z']
            band = fo['FIL']
            mag = fo['MAG']
            syst = fo['SYS']

            mag, syst = phot_system.auto_conversion(mag, band, syst)

            # Redshift correction
            DEFAULT_WAVE *= (1 + redshift)
            
            # Check range
            check_range(DEFAULT_WAVE, l1, l2)

            _, _, K = filter_manager.apply_filter(DEFAULT_WAVE, flux, band, mag, syst)

        elif fo['SPEC'] == 'bb':
            DEFAULT_WAVE = np.linspace(100, 30000, 10000)
            
            tmp = fo['TEMP']
            band = fo['FIL']
            mag = fo['MAG']
            syst = fo['SYS']
    
            flux = sed_models.blackbody(DEFAULT_WAVE, tmp)
            mag, syst = phot_system.auto_conversion(mag, band, syst)
            _, _, K = filter_manager.apply_filter(DEFAULT_WAVE, flux, band, mag, syst)
    
        elif fo['SPEC'] == 'pl':
            DEFAULT_WAVE = np.linspace(100, 30000, 10000)
            
            indpl = fo['INDEX']
            band = fo['FIL']
            mag = fo['MAG']
            syst = fo['SYS']
            
            flux = sed_models.powerlaw(DEFAULT_WAVE, indpl)
            mag, syst = phot_system.auto_conversion(mag, band, syst)
            _, _, K = filter_manager.apply_filter(DEFAULT_WAVE, flux, band, mag, syst)
    
        elif fo['SPEC'] == 'line':
            DEFAULT_WAVE = np.linspace(100, 30000, 10000)
            center = fo['WAVE_CENTER']
            fwhm = fo['WAVE_FWHM']
            
            check_line(center, fwhm, l1, l2)
            tot_flux = fo['FLUX']
            flux = sed_models.gaussian_line(DEFAULT_WAVE, center, tot_flux, fwhm)
            K = 1
            
        # Put wave and flux*K in a MPDAF object
        spec_raw = Spectrum(data=flux*K, wave=WaveCoord(cdelt=DEFAULT_WAVE[1]-DEFAULT_WAVE[0], 
                                                   crval=DEFAULT_WAVE[0]))

        # Resample
        rspec = spec_raw.resample(lstep, start=l1)
        spec_cut = rspec.subspec(lmin=l1, lmax=l2)
        
        return spec_raw, spec_cut

    def rebin_spectrum(self, nph_source, tot_noise, bin_factor=2):
        """Rebin a MPDAF spectrum and its noise by combining adjacent pixels
    
        Parameters
        ----------
        nph_source : MPDAF Spectrum
            Original signal spectrum
        tot_noise : MPDAF Spectrum 
            Original noise spectrum
        bin_factor : int
            Number of pixels to bin together
        
        Returns
        -------
        bin_snr : MPDAF Spectrum
            Rebinned spectrum of the SNR on original wavelength grid
        """
        # Get original wavelength grid
        waves = nph_source.wave.coord()
    
        # Calculate centers of bins
        n_bins = len(waves) // bin_factor  
        bin_centers = np.array([
            np.mean(waves[i:i+bin_factor])
            for i in range(0, n_bins * bin_factor, bin_factor)
        ])
    
        # Bin the signal (sum)
        binned_signal = np.array([
            np.sum(nph_source.data[i:i+bin_factor]) 
            for i in range(0, n_bins * bin_factor, bin_factor)
        ])
    
        # Bin the noise (quadrature sum)
        binned_noise = np.array([
            np.sqrt(np.sum(tot_noise.data[i:i+bin_factor]**2))
            for i in range(0, n_bins * bin_factor, bin_factor)
        ])
    
        # Create temporary spectra with binned data
        temp_signal = Spectrum(data=binned_signal,
                            wave=WaveCoord(cdelt=nph_source.wave.get_step()*bin_factor,
                                     crval=bin_centers[0]))
    
        temp_noise = Spectrum(data=binned_noise,
                       wave=WaveCoord(cdelt=tot_noise.wave.get_step()*bin_factor,
                                    crval=bin_centers[0]))
    
        # Resample back to original wavelength grid
        final_signal = temp_signal.resample(nph_source.wave.get_step(),
                                     start=nph_source.wave.get_start(),
                                     shape=len(waves))
    
        final_noise = temp_noise.resample(tot_noise.wave.get_step(),
                                   start=tot_noise.wave.get_start(), 
                                   shape=len(waves))
    
        bin_snr = final_signal / final_noise
    
        self.logger.debug('Computing rebinned spectrum for factor %d', bin_factor)
    
        return bin_snr

    def sun_moon_sep(self, fli):
        """Compute sun-moon separation from fractional lunar illumination
        
        Parameters
        ----------
        fli : float
            Fractional lunar illumination (0-1)
            
        Returns
        -------
        float
            Sun-moon separation in degrees
        """
        if not 0 <= fli <= 1:
            raise ValueError("FLI must be between 0 and 1.")
        theta_rad = np.arccos(1 - 2 * fli)  # result in radians
        theta_deg = np.degrees(theta_rad)  # convert to degrees
        return theta_deg

    def compute_sky2(self, fo):
        """Compute sky emission and transmission using skycalc_ipy
        
        Parameters
        ----------
        fo : dict
            Dictionary containing observation parameters
            
        Returns
        -------
        tuple
            (dict, table)
            - dict containing emission and transmission spectra
            - skycalc results table
        """
        insfam = getattr(self, fo["INS"]) 
        CONF = insfam[fo["CH"]]
        
        mss = self.sun_moon_sep(fo['FLI'])
        airmass = fo['AM']
        pwv = fo['PWV']
        allowed_pwv = [0.05, 0.01, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0]
        closest_value = min(allowed_pwv, key=lambda v: abs(v - pwv))

        if pwv not in allowed_pwv:
            self.logger.warning(f"PWV value not allowed, assigned the closest one: {pwv} → {closest_value}")
            pwv = closest_value
        
        skycalc = skycalc_ipy.SkyCalc()

        skycalc["msolflux"] = 130
        skycalc['observatory'] = 'paranal'
        skycalc['airmass'] = airmass
        skycalc['pwv'] = pwv
        if 'MS_ANGLE' in fo:
            skycalc['moon_sun_sep'] = fo['MS_ANGLE']
        else:
            skycalc['moon_sun_sep'] = mss
        
        skycalc['wmin'] = CONF['lbda1']/10
        skycalc['wmax'] = CONF['lbda2']/10
        skycalc['wdelta'] = CONF['dlbda']/10
        skycalc['wgrid_mode'] = 'fixed_wavelength_step'
        
        tbl = skycalc.get_sky_spectrum(return_type="tab-ext")
        
        if abs(tbl['lam'][0]*10 - CONF['lbda1'])>CONF['dlbda'] or \
        abs(tbl['lam'][-1]*10 - CONF['lbda2'])>CONF['dlbda'] or \
        abs(tbl['lam'][1]- tbl['lam'][0])*10 - CONF['dlbda']>0.01:
            raise ValueError(f'Incompatible bounds between called configuration and setup')
        
        d = dict()
        d['emi_orig'] = Spectrum(data=tbl['flux'], wave=CONF['wave'])     
        
        # # # LSF convolution
        d['emi'] = d['emi_orig'].filter(width=CONF['lsfpix'])

        d['abs'] = Spectrum(data=tbl['trans'], wave=CONF['wave'])

        return d, tbl 

    def get_sky2(self, fo):
        """Get sky emission and transmission spectra
        
        Parameters
        ----------
        fo : dict
            Dictionary containing observation parameters
            
        Returns
        -------
        tuple
            (emission spectrum, absorption spectrum)
        """
        d, _ = self.compute_sky2(fo)
        return d['emi'], d['abs']

    def print_info_models_filters(self):
        """Print information about available SED models and filters"""
        self.logger.info("Available Vega filters:")
        print(filter_manager.phot_system.filters_vega)
        self.logger.info("Available AB filters:")
        print(filter_manager.phot_system.filters_AB)
        self.logger.info("Available SED models:")
        print(sed_models.eso_spectra_files.keys())

def _copy_obs(obs, res):
    """ copy obs dict in res dict """
    for key,val in obs.items():
        res[key] = val

def asymgauss(ftot, l0, sigma, skew, wave):
    """compute asymetric gaussian

    Parameters
    ----------
    ftot : float
        total flux
    l0 : float
        peak wavelength in A
    sigma : float
        sigma in A
    skew : float
        skew parameter (0 for a gaussian)
    wave : numpy array of float
        wavelengths in A

    Returns
    -------
    numpy array of float
        asymetric gaussian values

    """
    dl = wave - l0
    g = np.exp(-dl**2/(2*sigma**2))
    f = 1 + erf(skew*dl/(1.4142135623730951*sigma))
    h = f*g
    h = ftot * h/h.sum()
    return h

def peakwave_asymgauss(lpeak, sigma, skew, dl=0.01):
    """ compute the asymetric gaussian wavelength paramater to get the given peak wavelength

    Parameters
    ----------
    lpeak : float
        peak wavelength in A
    sigma : float
        sigma in A
    skew : float
        skew parameter (0 for a gaussian)
    dl : float
       step in wavelength (A, Default value = 0.01)

    Returns
    -------
    float
        wavelength (A) of the asymetric gaussian

     """
    wave = np.arange(lpeak-5*sigma,lpeak+5*sigma,dl)
    res0 = root_scalar(_funasym, args=(lpeak, sigma, skew, wave), method='brentq',
                          bracket=[lpeak-2*sigma,lpeak+2*sigma], rtol=1.e-3, maxiter=100)
    return res0.root

def _funasym(l0, lpeak, sigma, skew, wave):
    """ function used to minimize in peakwave_asymgauss """
    f = asymgauss(1.0, l0, sigma, skew, wave)
    k = f.argmax()
    zero = wave[k] - lpeak
    #print(zero)
    return zero

def _fun_aper(kfwhm, obj, ins, flux, ima, spec, krange=None):
    """ function used to minimize in optimum_circular_aperture """
    obj.obs['ima_kfwhm'] = kfwhm
    res = obj.snr_from_source(ins, flux, ima, spec, debug=False)
    if krange is None:
        snr = res['aper']['snr']
    else:
        snr = np.mean(res['spec']['snr'].data[krange[0]:krange[1]])
    return -snr

def _fun_range(kfwhm, obj, ins, flux, ima, spec):
    """ function used to minimize in optimum_spectral_range """
    obj.obs['spec_range_kfwhm'] = kfwhm
    res = obj.snr_from_source(ins, flux, ima, spec, debug=False)
    snr = res['aper']['snr']
    #print(kfwhm, snr)
    return -snr

def vdisp2sigma(vdisp, l0):
    """compute sigma in A from velocity dispersion in km/s

    Parameters
    ----------
    vdisp : float
        velocity dispersion (km/s)
    l0 : float
        wavlenegth (A)

    Returns
    -------
    float
        sigma in A

       """
    return vdisp*l0/C_kms

def sigma2vdisp(sigma, l0):
    """compute sigma in A from velocity dispersion in km/s

    Parameters
    ----------
    sigma : float
        sigma in A
    l0 : float
        wavelength in A

    Returns
    -------
    float
        velocity dispersion in km/s

     """
    return sigma*C_kms/l0

def fwhm_asymgauss(lbda, flux):
    """ compute the FWHM of an asymmetric gaussian

    Parameters
    ----------
    lbda : numpy array of float
        wavelength array in A
    flux : numpy array of float
        asymetric gaussian values

    Returns
    -------
    tuple
        l0,l1,l2
        l0 peak wavelength (A)
        l1 blue wavelength at FWHM (A)
        l2 red wavelength at FWHM (A)

    """
    g = flux/flux.max()
    kmax = g.argmax()
    l0 = lbda[kmax]
    l1 = None
    for k in range(kmax,0,-1):
        if g[k] < 0.5:
            l1 = np.interp(0.5, [g[k],g[k+1]],[lbda[k],lbda[k+1]])
            break
    if l1 is None:
        return None
    l2 = None
    for k in range(kmax,len(lbda),1):
        if g[k] < 0.5:
            l2 = np.interp(0.5, [g[k],g[k-1]],[lbda[k],lbda[k-1]])
            break
    if l2 is None:
        return None
    return l0,l1,l2

def moffat(samp, fwhm, beta, ell=0, kfwhm=3, oversamp=1, uneven=1):
    """ compute a 2D Moffat image

    Parameters
    ----------
    samp : float
        image sampling in arcsec
    fwhm : float
        FWHM of the MOFFAT (arcsec)
    beta : float
        MOFFAT shape parameter (beta > 4 for Gaussian, 1 for Lorentzien)
    ell : float
         image ellipticity (Default value = 0)
    kfwhm : float
         factor relative to the FWHM to compute the size of the image (Default value = 2)
    oversamp : int
         oversampling gfactor (Default value = 1)
    uneven : int
         if 1 the image size will have an uneven number of spaxels (Default value = 1)

    Returns
    -------
    MPDAF image
         MOFFAT image
    """


    ns = (int((kfwhm*fwhm/samp+1)/2)*2 + uneven)*oversamp
    pixfwhm = oversamp*fwhm/samp
    pixfwhm2 = pixfwhm*(1-ell)
    ima = moffat_image(fwhm=(pixfwhm2,pixfwhm), n=beta, shape=(ns,ns), flux=1.0, unit_fwhm=None)
    ima.data /= ima.data.sum()
    return ima

def sersic(samp, reff, n, ell=0, kreff=4, oversamp=1, uneven=1):
    """ compute a 2D Sersic image

    Parameters
    ----------
    samp : float
        image sampling in arcsec
    reff : float
        effective radius (arcsec)
    n : float
        Sersic index (4 for elliptical, 1 for elliptical disk)
    ell : float
         image ellipticity (Default value = 0)
    kreff : float
         factor relative to the effective radius to compute the size of the image (Default value = 3)
    oversamp : int
         oversampling gfactor (Default value = 1)
    uneven : int
         if 1 the image size will have an uneven number of spaxels (Default value = 1)

    Returns
    -------
    MPDAF image
         Sersic image
    """

    ns = (int((kreff*reff/samp+1)/2)*2 + uneven)*oversamp
    pixreff = oversamp*reff/samp          
    x,y = np.meshgrid(np.arange(ns), np.arange(ns))
    x0,y0 = ns/2-0.5,ns/2-0.5
    
    mod = Sersic2D(amplitude=1, r_eff=pixreff, n=n, x_0=x0, y_0=y0,
                   ellip=ell, theta=0)
    data = mod(x, y)            
    ima = Image(data=data)
    ima.data /= ima.data.sum()

    # copy the WCS from a dummy Moffat since the Sersic does not have it 
    dummy = moffat_image(fwhm=(1,1), n=10, shape=(ns,ns), flux=1.0, unit_fwhm=None)
    ima.wcs = dummy.wcs

    return ima

def compute_sky(lbda1, lbda2, dlbda, lsf, moon, airmass=1.0):
    """ compute Paranal sky model from ESO skycalc

    Parameters
    ----------
    lbda1 : float
        starting wavelength (A)
    lbda2 : float
        ending wavelength (A)
    dlbda : float
        step in wavelength (A)
    lsf : float
        LSF size in spectels
    moon : str
        moon brightness (eg darksky)
    airmass : float
         observation airmass (Default value = 1.0)

    Returns
    -------
    astropy table
        sky table as computed by skycalc

    """
    skyModel = SkyModel()
    if moon == 'darksky':
        sep = 0
    elif moon == 'greysky':
        sep = 90
    elif moon == 'brightsky':
        sep = 180
    else:
        raise ValueError(f'Error in moon {moon}')
    skypar = dict(wmin=lbda1*0.1, wmax=lbda2*0.1, wdelta=dlbda*0.1,
                  lsf_gauss_fwhm=lsf, lsf_type='Gaussian', airmass=airmass,
                  moon_sun_sep=sep, observatory='paranal')
    skypar = fixObservatory(skypar)
    skyModel.callwith(skypar)
    f = BytesIO()
    f = BytesIO(skyModel.data)
    tab = Table.read(f)
    tab.iden = f"{moon}_{airmass:.1f}"
    return tab

def show_noise(r, ax, legend=False, title='Noise fraction'):
    """ plot the noise characteristics from the result dictionary

    Parameters
    ----------
    r : dict
          result dictionary that contain the noise results
    ax : amtplolib axis
          axis where to plot
    legend : bool
         if True display legend on the figure (Default value = False)
    title : str
         title to display (Default value = 'Noise fraction')

    """
    rtot = r['tot']
    f = (r['sky'].data/rtot.data)**2
    ax.plot(rtot.wave.coord(), f, color='r', label='sky' if legend else None)
    f = (r['source'].data/rtot.data)**2
    ax.plot(rtot.wave.coord(), f, color='b', label='source' if legend else None)
    f = (r['ron']/rtot.data)**2 if isinstance(r['ron'],float) else (r['ron'].data/rtot.data)**2
    ax.plot(rtot.wave.coord(), f, color='g', label='ron' if legend else None)
    f = (r['dark']/rtot.data)**2 if isinstance(r['dark'],float) else (r['dark'].data/rtot.data)**2
    ax.plot(rtot.wave.coord(), f, color='m', label='dark' if legend else None)
    if isinstance(r['ron'],float):
        f = (r['ron']**2 + r['dark']**2)/rtot.data**2
    else:
        f = (r['ron'].data**2 + r['dark'].data**2)/rtot.data**2
    ax.plot(rtot.wave.coord(), f, color='k', label='ron+dark' if legend else None)
    if legend:
        ax.legend(loc='upper right')
    ax.axhline(0.5, color='k', alpha=0.2)
    ax.axhline(0.5, color='k')
    ax.set_title(title)


def get_data(obj, chan, name, refdir):
    """ retreive instrument data from the associated setup files

    Parameters
    ----------
    obj : ETC class
        instrument class (e.g. etc.ifs)
    chan : str
        channel name (eg 'red')
    name : str
        instrument name (eg 'ifs')
    refdir : str
        directory path where the setup fits file can be found

    """
    ins = obj[chan]
    ins['wave'] = WaveCoord(cdelt=ins['dlbda'], crval=ins['lbda1'], cunit=u.angstrom)

    flist = glob.glob(os.path.join(refdir,f"{name}_{chan}_*sky_*.fits"))
    flist.sort()
    ins['sky'] =[]
    moons = []
    for fname in flist:
        f = os.path.basename(fname).split('_')
        moon = f[2]
        moons.append(moon)
        airmass = float(f[3][:-5])
        d = dict(moon=moon, airmass=airmass)
        tab = Table.read(fname, unit_parse_strict="silent")
        for key,val in [['LSFPIX',ins['lsfpix']],
                         ['LBDA1',ins['lbda1']],
                         ['LBDA2',ins['lbda2']],
                         ['DLBDA',ins['dlbda']],
                         ['MOON', moon],
                         ['AIRMASS', airmass]
                         ]:
            if key in tab.meta:
                if isinstance(tab.meta[key], float):
                    if not np.isclose(tab.meta[key], val):
                        raise ValueError(f"Incompatible {key} values between {fname} and setup")
                else:
                    if tab.meta[key] != val:
                        raise ValueError(f"Incompatible {key} values between {fname} and setup")
        if abs(tab['lam'][0]*10 - ins['lbda1'])>ins['dlbda'] or \
           abs(tab['lam'][-1]*10 - ins['lbda2'])>ins['dlbda'] or \
           abs(tab['lam'][1]-tab['lam'][0])*10 - ins['dlbda']>0.01:
            raise ValueError(f'Incompatible bounds between {fname} and setup')
        d['emi'] = Spectrum(data=tab['flux'], wave=ins['wave'])
        d['abs'] = Spectrum(data=tab['trans'], wave=ins['wave'])
        d['filename'] = fname
        ins['sky'].append(d)
    filename = f'{name}_{chan}_noatm.fits'

#    print("I am loading {0}".format(filename))
#    exit()
    trans=Table.read(os.path.join(refdir,filename), unit_parse_strict="silent")
    if trans['WAVE'][0]*10 > ins['lbda1'] or \
       trans['WAVE'][-1]*10 < ins['lbda2'] :

        print("Transmission goes from {0} to {1}".format(trans['WAVE'][0],
                                                         trans['WAVE'][-1]))
        print("While lbda1={0} and lbda2={1}".format(ins['lbda1'],
                                                     ins['lbda2']))
        
        raise ValueError(f'Incompatible bounds between {filename} and setup')
    ins['instrans'] = Spectrum(data=np.interp(ins['sky'][0]['emi'].wave.coord(),trans['WAVE']*10,trans['TOT']),
                                                wave=ins['sky'][0]['emi'].wave)
    ins['instrans'].filename = filename
    ins['skys'] = list(set(moons))
    ins['wave'] = ins['instrans'].wave
    ins['chan'] = chan
    ins['name'] = name
    return

def update_skytables(logger, obj, name, chan, moons, airmass, refdir, overwrite=False, debug=False):
    """ update setup sky files for a change in setup parameters

    Parameters
    ----------
    logger : logging instance
        logger to print progress
    obj : dict
        instrument dictionary
    name : str
        instrument name
    chan: str
        channel name
    moons : list of str
        list of moon sky conditions eg ['darksky']
    airmass : list of float
        list of airmass
    refdir : str
        path name where to write the reference setup fits file
    overwrite : bool
         if True overwrite existing file (Default value = False)
    debug : bool
         if True do not try to write(used for unit test, Default value = False)

    """
    for am in airmass:
        for moon in moons:
            tab = compute_sky(obj['lbda1'], obj['lbda2'], obj['dlbda'],
                              obj['lsfpix'], moon, am)
            tab.meta['lsfpix'] = obj['lsfpix']
            tab.meta['moon'] = moon
            tab.meta['airmass'] = am
            tab.meta['lbda1'] = obj['lbda1']
            tab.meta['lbda2'] = obj['lbda2']
            tab.meta['dlbda'] = obj['dlbda']
            fname = f"{name}_{chan}_{moon}_{am:.1f}.fits"
            filename = os.path.join(refdir, fname)
            logger.info('Updating file %s', filename)
            if debug:
                logger.info('Debug mode table not saved to file')
            else:
                tab.write(filename, overwrite=overwrite)


def get_seeing_fwhm(seeing, airmass, wave, diam, iq_tel, iq_ins):
    """ compute FWHM for the Paranal ESO ETC model

    Parameters
    ----------
    seeing : float
        seeing (arcsec) at 5000A
    airmass : float
        airmass of the observation
    wave : numpy array of float
        wavelengths in A
    diam : float
        telescope primary mirror diameter in m
    iq_tel : float of numpy array
        image quality of the telescope
    iq_ins : float of numpy array
        image quality of the instrument

    Returns
    -------
    numpy array of float
        FWHM (arcsec) as function of wavelengths

    """
    
    r0 = 0.1*seeing**(-1)*(wave/5000)**(1.2)*airmass**(-0.6) #modified by Jose based on fiber-injection losses for ESPRESSO (Schmidt+24)
    #r0 = 0.188 # for VLT (in ETC)
    l0 = 46 # for VLT (in ETC)
    Fkolb = 1/(1+300*diam/l0)-1
    iq_atm = seeing*(wave/5000)**(-1/5)*airmass**(3/5) * \
        np.sqrt(1+Fkolb*2.183*(r0/l0)**0.356)
    iq = np.sqrt(iq_atm**2 + iq_tel**2 + iq_ins**2)
    iq_before_ins = np.sqrt(iq_atm**2 + iq_tel**2)
    return iq, iq_before_ins

def _checkobs(obs, saved, keys):
    """ check existence and copy keywords from obs to saved """
    for key in keys:
        if key not in obs.keys():
            raise KeyError(f'keyword {key} missing in obs dictionary')
        saved[key] = obs[key]

# # # # new methods for the WST - ETC
#convenient function for the range check
def check_range(arr, M_min, M_max):
    
    if arr[0] > M_min:
        print('Trace starts after the first pixel!')
    if arr[-1] < M_max:
        print('Trace ends before the last pixel!')

def check_line(cen, fwhm, M_min, M_max):
    if cen > M_max:
        print('Line outside the last pixel!')
    elif cen + fwhm > M_max:
        print('Line near the last pixel!')
    if cen < M_min:
        print('Line outside the first pixel!')
    elif cen - fwhm < M_min:
        print('Line near the first pixel!')


