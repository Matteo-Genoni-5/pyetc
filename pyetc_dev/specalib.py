import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, integrate, interpolate
import pandas as pd
from astropy import units as u
from astropy.constants import h, c, k_B
from importlib.resources import files #this works only for Python > 3.9
import pkg_resources

class PhotometricSystem:
    """
    Class for managing photometric systems (Vega and AB) and conversions between them.
    """
    def __init__(self):
        # Definition of available systems and filters, should be modified as needed
        self.MAG_SYSTEMS = ["Vega", "AB"]
        
        # From https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
        self.filters_vega = ["U", "B", "V", "R", "I", "J", "H", "K"]
        self.ab_vega_diff_vf = [0.79, -0.09, 0.02, 0.21, 0.45, 0.91, 1.39, 1.85]
        
        # Should be properly computed, since for now we are using the conversion from https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
        self.filters_AB = ["uSDSS", "gSDSS", "rSDSS", "iSDSS", "zSDSS", "uLSST", "gLSST", "rLSST", "iLSST", "zLSST"]
        self.ab_vega_diff_abf = [0.91, -0.08, 0.16, 0.37, 0.54, 0.91, -0.08, 0.16, 0.37, 0.54]
        
        # Zero points for Vega from 
        # https://www.eso.org/observing/etc/doc/skycalc/helpskycalc.html#mags
        # BP: [0: ll in A, 1: Fv in [erg/cm^2/s/Hz], 2: Fll in [erg/cm^2/s/A], 3: PHll in [Photons/cm^2/s/A] ]
        self.VEGA_flux_zeropoints = {
            "U": [3600., None, 4.18023e-9, 757.5], 
            "B": [4380., None, 6.60085e-9, 1455.4],
            "V": [5450., None, 3.60994e-9, 990.4],
            "R": [6410., None, 2.28665e-9, 737.9],
            "I": [7980., None, 1.22603e-9, 492.5],
            "J": [12200., None, 3.12e-10, 191.6],
            "H": [16300., None, 1.14e-10, 93.5],
            "K": [21900., None, 3.94e-11, 43.4]
        }
        
        # Zero points for AB from 
        # http://svo2.cab.inta-csic.es/theory/fps/index.php?id=SLOAN & id=LSST
        # BP: [0: ll in A, 1: Fv in [erg/cm^2/s/Hz], 2: Fll in [erg/cm^2/s/A], 3: PHll in [Photons/cm^2/s/A] ]
        self.AB_flux_zeropoints = {
            "uSDSS": [3542.10, 3631., 8.88093e-9, None],
            "gSDSS": [4723.59, 3631., 4.79807e-9, None],
            "rSDSS": [6201.71, 3631., 2.78937e-9, None],
            "iSDSS": [7672.59, 3631., 1.82728e-9, None],
            "zSDSS": [10500.61, 3631., 9.28119e-10, None],
            "uLSST": [3641.17, 3631., 8.57499e-9, None],
            "gLSST": [4704.08, 3631., 4.83202e-9, None],
            "rLSST": [6155.82, 3631., 2.83044e-9, None],
            "iLSST": [7504.64, 3631., 1.91692e-9, None],
            "zLSST": [8695.51, 3631., 1.43756e-9, None]
        }
            
        # From SOXS, ask Matteo G. where these were taken from, need to compare the transmission curves and ZPs with the one above taken from SVO
        #self.AB_flux_zeropoints2 = {
            #"uSDSS": [3560., 3631, 859.5e-11, 1539.3],
            #"gSDSS": [4830., 3631, 466.9e-11, 1134.6],
            #"rSDSS": [6260., 3631, 278.0e-11, 875.4],
            #"iSDSS": [7670., 3631, 185.2e-11, 714.5],
            #"zSDSS": [9100., 3631, 131.5e-11, 602.2]
        #}    

        self.band_filters = self._load_filter_profiles()  
    
    def _load_filter_profiles(self):
        """Loads filter transmission profiles from files."""
        filters_folder = pkg_resources.resource_filename("pyetc_dev", "Band_Filters/")  # Replace with the correct path
        band_filters = {}
        
        for filename in os.listdir(filters_folder):
            if filename.endswith(".txt"):
                with open(filters_folder + filename, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
                    data = np.loadtxt([line for line in lines if not line.strip().startswith(('#', '!'))])
                    band_name = filename[:-4]  # Remove '.txt' extension
                    band_filters[band_name] = data
        return band_filters
                
    def get_flux_zeropoint(self, band, system="Vega", quantity="Fll"):
        """
        Returns the zero point flux for a given filter and system.
        
        Parameters:
        - band: filter name (e.g. 'U', 'g')
        - system: 'Vega' or 'AB'
        - quantity: 'Fv' [erg/cm²/s/Hz], 'Fll' [erg/cm²/s/Å], 'PHll' [photons/cm²/s/Å]
        
        Returns:
        - lambda_eff: effective wavelength [Å]
        - zeropoint: zero point value in requested units
        """
        if system == "Vega":
            if band not in self.filters_vega:
                raise ValueError(f"Invalid Vega filter. Choose from: {self.filters_vega}")
            zp_dict = self.VEGA_flux_zeropoints
        elif system == "AB":
            if band not in self.filters_AB:
                raise ValueError(f"Invalid AB filter. Choose from: {self.filters_AB}")
            zp_dict = self.AB_flux_zeropoints
        else:
            raise ValueError("Invalid photometric system. Choose 'Vega' or 'AB'")
        
        if quantity == "Fv":
            sel = 1
        elif quantity == "Fll":
            sel = 2
        elif quantity == "PHll":
            sel = 3
        else:
            raise ValueError("Invalid quantity. Choose 'Fv', 'Fll' or 'PHll'")
        
        return zp_dict[band][0], zp_dict[band][sel]
    
    def convert_magnitude(self, mag, band, from_system, to_system):
        """
        Converts magnitude between photometric systems.
        
        Parameters:
        - mag: magnitude value
        - band: filter name
        - from_system: source system ('Vega' or 'AB')
        - to_system: target system ('Vega' or 'AB')
        
        Returns:
        - mag_converted: converted magnitude
        """
        if from_system == to_system:
            return mag
        
        if from_system == "Vega" and to_system == "AB":
            if band not in self.filters_AB:
                raise ValueError(f"Invalid AB filter for conversion. Choose from: {self.filters_AB}")
            return mag + self.ab_vega_diff_abf[self.filters_AB.index(band)]
        
        elif from_system == "AB" and to_system == "Vega":
            if band not in self.filters_vega:
                raise ValueError(f"Invalid Vega filter for conversion. Choose from: {self.filters_vega}")
            return mag - self.ab_vega_diff_vf[self.filters_vega.index(band)]
        
        else:
            raise ValueError("Unsupported conversion")

    def auto_conversion(self, mag, band, sys):
        """
        Automatically converts magnitude to the appropriate system for the given band.
        
        Parameters:
        - mag: magnitude
        - band: filter name
        - sys: current system ('Vega' or 'AB')
        
        Returns:
        - new_mag: converted magnitude
        - new_sys: new system
        """
        new_sys = sys
        new_mag = mag
        if sys == "Vega":
            if band not in self.filters_vega:
                new_sys = 'AB'
                new_mag = self.convert_magnitude(mag, band, sys, new_sys)
        elif sys == "AB":
            if band not in self.filters_AB:
                new_sys = 'Vega'
                new_mag = self.convert_magnitude(mag, band, sys, new_sys)
        
        if new_sys != sys:
            print("Wrong coupling of filter and system")
            print("Conversion:")
            print(f"{sys} > {new_sys}")
            print(f"{mag} > {new_mag}")
        return new_mag, new_sys
    
class SEDModels:
    """
    Class for generating spectral energy distribution (SED) models.
    """
    def __init__(self):
        # Save all filenames from ESO_original_spectra/ directory to a dictionary
        self.eso_spectra_files = {}
        eso_spectra_dir = pkg_resources.resource_filename("pyetc_dev", "ESO_original_spectra/") #Replace also this with your path
        try:
            for idx, filename in enumerate(os.listdir(eso_spectra_dir)):
                if os.path.isfile(os.path.join(eso_spectra_dir, filename)):
                    # Using filename without extension as the key
                    name = os.path.splitext(filename)[0]
                    self.eso_spectra_files[name] = os.path.join(eso_spectra_dir, filename)
        except FileNotFoundError:
            print(f"Warning: Directory {eso_spectra_dir} not found")
        
    @staticmethod
    def blackbody(wavelength, temperature):
        """
        Generates a blackbody spectrum.
        
        Parameters:
        - wavelength: wavelength array [Å]
        - temperature: temperature [K]
        
        Returns:
        - flux: flux [erg/s/cm²/Å]
        """
        wavelength_m = wavelength * 1e-10  # Convert Å to meters
        exponent = h.value * c.value / (wavelength_m * k_B.value * temperature)
        
        # Avoid overflow for large exponents
        #exponent = np.where(exponent > 100, 100, exponent)
        
        flux = (2 * h.value * c.value**2 / (wavelength_m**5 * (np.exp(exponent) - 1))) * 1e-10
        return flux * 1e7  # Convert W/m²/m to erg/s/cm²/Å
    
    @staticmethod
    def powerlaw(wavelength, slope, norm_wavelength=5500, norm_flux=1.0):
        """
        Generates a power law spectrum.
        
        Parameters:
        - wavelength: wavelength array [Å]
        - slope: power law index
        - norm_wavelength: normalization wavelength [Å]
        - norm_flux: flux at normalization wavelength
        
        Returns:
        - flux: normalized flux
        """
        return norm_flux * (wavelength / norm_wavelength)**slope
    
    # We could add the asymmetric gaussian in principle
    @staticmethod
    def gaussian_line(wavelength, center, flux, fwhm):
        """
        Generates a gaussian spectral line.
        
        Parameters:
        - wavelength: wavelength array [Å]
        - center: line center [Å]
        - flux: integrated line flux [erg/s/cm²]
        - fwhm: full width at half maximum [Å]
        
        Returns:
        - flux: flux [erg/s/cm²/Å]
        """
        sigma = fwhm / 2.355
        exponent = -0.5 * ((wavelength - center) / sigma)**2
        return (flux / (sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)
    
    @staticmethod
    # We could add some kind of smoothing here
    def interpolate_spectrum(wavelength, target_wavelength, flux):
        """
        Interpolates a spectrum to new wavelengths.
        
        Parameters:
        - wavelength: original wavelengths [Å]
        - target_wavelength: target wavelengths [Å]
        - flux: original flux [any unit]
        
        Returns:
        - flux_interp: interpolated flux
        """
        return np.interp(target_wavelength, wavelength, flux)
    
    @staticmethod
    def _resolve_template_path(filename):
        if os.path.dirname(filename):
            return filename
        else:
            # Prende la directory in cui si trova questo file (specalib.py)
            base_dir = os.path.dirname(__file__)
            template_dir = os.path.join(base_dir, 'ESO_original_spectra')
            return os.path.join(template_dir, filename)
        
    @staticmethod
    def template(filename, waveunit='AA', unitsf='Fll'):
        """
        Reads a two-column file, skipping lines that start with '#' or '!'.
        
        Parameters:
        - filename: path to the template file
        - waveunit: wavelength unit, 'AA' (default) or 'nm'
        - unitsf: flux unit 'Fll' or 'PHll'

        Returns:
        - tem: template name
        - wave: wavelength array [Å]
        - flux: flux array
        """

        filepath = SEDModels._resolve_template_path(filename)

        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
    
        # Filter lines not starting with "#" or "!"
        data = np.loadtxt([line for line in lines if not line.strip().startswith(('#', '!'))])
        wave = data[:, 0]
        flux = data[:, 1]
        tem = os.path.basename(filename).split('.')[0]
            
        if waveunit == 'nm':
            wave *= 10
            
        if unitsf == 'PHll':
            flux = 1.98644746e-08 * (flux / wave)
        return tem, wave, flux

class FilterManager:
    """
    Class for managing filters and normalizations.
    """
    def __init__(self, phot_system):
        self.phot_system = phot_system
    
    def get_filter_profile(self, band):
        """
        Returns the transmission profile of a filter.
        
        Parameters:
        - band: filter name
        
        Returns:
        - wavelength: wavelength array [Å]
        - transmission: transmission [0-1]
        """
        if band not in self.phot_system.band_filters:
            raise ValueError(f"Filter {band} not found")
        return self.phot_system.band_filters[band].T[0], self.phot_system.band_filters[band].T[1]
    
    def apply_filter(self, wavelength, flux, band, mag=None, system="Vega", typeSP="Fll"):
        """
        Applies a filter to a spectrum and normalizes to a magnitude.
        
        Parameters:
        - wavelength: spectrum wavelength array [Å]
        - flux: spectrum flux [erg/s/cm²/Å]
        - band: filter name
        - mag: target magnitude (None for no normalization)
        - system: photometric system ('Vega' or 'AB')
        - typeSP: flux type ('Fll' or 'PHll')
        
        Returns:
        - common_wave: common wavelength grid
        - filtered_flux: filtered and normalized flux
        - K: normalization factor
        """
        # Get filter profile
        filt_wave, filt_trans = self.get_filter_profile(band)
        
        # Interpolate spectrum and filter to a common grid
        min_wave = max(wavelength.min(), filt_wave.min())
        max_wave = min(wavelength.max(), filt_wave.max())
        common_wave = np.linspace(min_wave, max_wave, 1000)
        
        flux_interp = SEDModels.interpolate_spectrum(wavelength, common_wave, flux)
        trans_interp = SEDModels.interpolate_spectrum(filt_wave, common_wave, filt_trans)
        
        # Apply filter
        filtered_flux = flux_interp * trans_interp  # To be corrected, only works with integral 1?
        
        # Normalize if requested
        K = 1.0
        if mag is not None:
            # Get zero point
            _, zp_ph = self.phot_system.get_flux_zeropoint(band, system, typeSP)
            zp_ph_scaled = zp_ph * (2.512**(-mag))
            
            # Calculate normalization factor
            integral_flux = integrate.trapezoid(filtered_flux, common_wave)
            integral_trans = integrate.trapezoid(trans_interp, common_wave)
            
            K = (zp_ph_scaled * integral_trans) / integral_flux
            filtered_flux *= K 
        
        return common_wave, filtered_flux, K

# Plot for comparisons
def plot_spectra_comparison(wave_coarse, flux_coarse, wave, flux, 
                           label_coarse, label, title=None, namepng=None):
    """
    Plots the coarse spectrum, interpolated fine spectrum, and original fine spectrum for comparison.
    
    Parameters:
    - wave_coarse: coarse wavelength grid
    - flux_coarse: coarse flux
    - wave: fine wavelength grid
    - flux: fine flux
    - label_coarse: label for coarse spectrum
    - label: label for fine spectrum
    - title: plot title
    - namepng: filename to save the plot (without extension)
    """
    
    flux_interpolated = SEDModels.interpolate_spectrum(wavelength=wave,
                                    target_wavelength=wave_coarse, flux=flux)
    
    # To ensure it works also for zeros
    mask = (flux_interpolated != 0) & (flux_coarse != 0)
    relative_diff = np.zeros_like(flux_interpolated)  # Initialize with zeros
    relative_diff[mask] = (flux_interpolated[mask] - flux_coarse[mask]) / flux_interpolated[mask] * 100
    
    # Old approach
    # relative_diff = (flux_interpolated - flux_coarse) / flux_interpolated * 100
    mean_diff = np.mean(relative_diff)
    std_diff = np.std(relative_diff)
    
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Upper plot: Spectra ---
    ax[0].plot(wave_coarse, flux_coarse, 'o-', markersize=0, label=label_coarse, color='blue', alpha=0.5)
    ax[0].plot(wave_coarse, flux_interpolated, 's--', markersize=0, label=label, color='red', alpha=0.5)
    ax[0].set_ylabel("Flux density [erg cm$^{-2}$ s$^{-1}$ \u00c5$^{-1}$]")
    ax[0].legend()
    ax[0].set_title(title if title else "Spectra Comparison")
    ax[0].grid()
    
    # --- Lower plot: Relative Difference ---
    ax[1].plot(wave_coarse, relative_diff, 'o-', markersize=0, color='black',
              label=f"Mean: ({mean_diff:.2f}±{std_diff:.2f})%")
    ax[1].axhline(0, color='gray', linestyle='--', lw=1)
    ax[1].set_ylabel("Relative Diff (%)")
    ax[1].set_xlabel("Wavelength [\u00c5]")
    ax[1].legend()
    ax[1].grid()
    
    if namepng is not None:
        plt.savefig(f"{namepng}.png", dpi=300, bbox_inches='tight')
    plt.show()

    #to do:
    # ADD GAIA FILTERS
    # ADD y, Y, Z filters
    # properly compute magnitude conversions based on the new ZPs
    # handle also Fnu spectra for the templates?