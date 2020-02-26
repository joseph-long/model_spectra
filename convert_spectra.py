import tarfile, lzma, gzip, glob
import re
import os.path

import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
import astropy.units as u

from joblib import Parallel, delayed
from functools import partial

def wien_peak(T):
    return ((2898 * u.um * u.K) / T).to(u.um)

# > The file names contain the main parameters of the models:
# lte{Teff/10}-{Logg}{[M/H]}a[alpha/H].GRIDNAME.7.spec.gz/bz2/xz
# is the synthetic spectrum for the requested effective temperature
# (Teff),surface gravity (Logg), metallicity by log10 number density with
# respect to solar values ([M/H]), and alpha element enhencement relative     
# to solar values [alpha/H]. The model grid is also mentioned in the name.
#
# — [README](https://phoenix.ens-lyon.fr/Grids/FORMAT)

AMES_COND_NAME_RE = re.compile(r'SPECTRA/lte(?P<t_eff_over_100>[^-]+)-(?P<log_g>[\d\.]+)-(?P<M_over_H>[\d.]+).AMES-Cond-2000.(?:spec|7).gz')

BT_SETTL_NAME_RE = re.compile(r'SPECTRA/lte(?P<t_eff_over_100>[^-]+)-(?P<log_g>[\d\.]+)-(?P<M_over_H>[\d.]+)a(?P<alpha_over_H>[-+\d.]+).BT-Settl.spec.7.xz')

def filepath_to_params(filepath, compiled_regex):
    match = compiled_regex.match(filepath)
    if match is None:
        return None
    parts = match.groupdict()
    t_eff = 100 * float(parts['t_eff_over_100'])
    log_g = float(parts['log_g'])
    M_over_H = float(parts['M_over_H'])
    return t_eff, log_g, M_over_H

# For regridding to common wavelengths
COMMON_WL_START = 0.5 * u.um
COMMON_WL_END = 6 * u.um
COMMON_WL_STEP = 5e-4 * u.um  # coarsest sampling of AMES-Cond and BT-Settl, at long end of Cond

COMMON_WL = np.arange(COMMON_WL_START.value, COMMON_WL_END.value, COMMON_WL_STEP.value) * u.um

def make_filepath_lookup(archive_tarfile, name_regex):
    '''Loop through all files in a tarfile of spectra
    and return `lookup` mapping (T_eff, log_g, M_over_H) tuples
    to filenames, `all_params` a dict of sets for 'T_eff',
    'log_g', 'M_over_H' values present
    '''
    filepath_lookup = {}
    all_params = {
        'T_eff': set(),
        'log_g': set(),
        'M_over_H': set(),
    }
    for name in archive_tarfile.getnames():
        parsed_params = filepath_to_params(name, name_regex)
        if parsed_params is None:
            continue  # skip entry for the 'SPECTRA' dir itself
        filepath_lookup[parsed_params] = name
        T_eff, log_g, M_over_H = parsed_params
        all_params['T_eff'].add(T_eff)
        all_params['log_g'].add(log_g)
        all_params['M_over_H'].add(M_over_H)
    return filepath_lookup, all_params

def parse_float(val):
    try:
        return float(val)
    except ValueError:
        return float(val.replace(b'D', b'e'))

BT_SETTL_DILUTION_FACTOR = -8
AMES_COND_DILUTION_FACTOR = -26.9007901434

# column1: wavelength in Angstroem
# column2: 10**(F_lam + DF) to convert to Ergs/sec/cm**2/A
# column3: 10**(B_lam + DF) i.e. the blackbody fluxes of same Teff in same units.
# -- https://phoenix.ens-lyon.fr/Grids/FORMAT
# Looks like this means column2 is F_lam in the equation in the README, column3 is B_lam
FORTRAN_FLOAT_REGEX = re.compile(b'([-+]?[\d.]+[De][-+][\d]+|Inf)')
BT_SETTL_REGEX = re.compile(b'^\s*([\d.]+)\s*([-+]?[\d.]+D[-+][\d]+)\s*([-+]?[\d.]+D[-+][\d]+).+$')
AMES_COND_REGEX = re.compile(b'^\s*([-+]?[\d.]+D[-+][\d]+)\s*([-+]?[\d.]+D[-+][\d]+)\s*([-+]?[\d.]+D[-+][\d]+).+$')

ORIG_FLUX_UNITS = u.erg * u.s**-1 * u.cm**-2 * u.AA**-1
FLUX_UNITS = u.W * u.m**-2 * u.um**-1

def parse_bt_settl_stacked_format(file_handle):
    raise NotImplementedError("No stacked-format BT-Settl spectra, right?")

def parse_bt_settl_row(row):
    match = BT_SETTL_REGEX.match(row)
    if match is None:
        raise ValueError("Unparseable row: {}".format(row))
    wavelength_bytes, flux_bytes, bb_flux_bytes = match.groups()
    wavelength_aa = parse_float(wavelength_bytes)
    flux = np.power(10, parse_float(flux_bytes) + BT_SETTL_DILUTION_FACTOR)
    bb_flux = np.power(10, parse_float(bb_flux_bytes) + BT_SETTL_DILUTION_FACTOR)
    return wavelength_aa, flux, bb_flux

def parse_ames_cond_stacked_format(file_handle):
    params_line = next(file_handle)
    assert b'Teff, logg, [M/H]' in params_line, "file doesn't start with params"
    n_wls_line = next(file_handle)
    assert b'number of wavelength points' in n_wls_line, "file doesn't include # wl points"
    numbers = re.findall(rb'(\d+)', n_wls_line)
    assert len(numbers) == 1, "More than 1 number found for # of wls"
    number_of_wls = int(numbers[0])
    wavelengths = []
    fluxes = []
    bb_fluxes = []
    wls_to_go = number_of_wls
    list_to_populate = wavelengths
    for line in file_handle:
        if wls_to_go < 1:
            raise Exception("Too much data?")
        numbers = FORTRAN_FLOAT_REGEX.findall(line)
        n_parts = len(line.split())
        if n_parts != len(numbers):
            raise RuntimeError(f'Got {n_parts} parts, only {len(numbers)} matched as floats')
        list_to_populate.extend([float(numstr.replace(b'D', b'e').replace(b'Inf', b'0')) for numstr in numbers])
        wls_to_go -= len(numbers)
        if wls_to_go == 0:
            if list_to_populate is wavelengths:
                list_to_populate = fluxes
            elif list_to_populate is fluxes:
                list_to_populate = bb_fluxes
            wls_to_go = number_of_wls
        elif wls_to_go < 0:
            if list_to_populate is wavelengths:
                raise RuntimeError(f'Parser overshot on wavelengths in {file_handle}')
            elif list_to_populate is fluxes:
                raise RuntimeError(f'Parser overshot on fluxes in {file_handle}')
            else:
                raise RuntimeError(f'Parser overshot on blackbody fluxes in {file_handle}')
    if not len(wavelengths) == len(fluxes) == len(bb_fluxes):
        raise RuntimeError(f'Mismatched lengths: {len(wavelengths)} wavelengths, {len(fluxes)} fluxes, {len(bb_fluxes)} BB fluxes in {file_handle}')
    wavelengths = np.asarray(wavelengths)
    fluxes = np.asarray(fluxes) * np.power(10, AMES_COND_DILUTION_FACTOR)
    bb_fluxes = np.asarray(bb_fluxes) * np.power(10, AMES_COND_DILUTION_FACTOR)
    return wavelengths, fluxes, bb_fluxes

def parse_ames_cond_row(row):
    match = AMES_COND_REGEX.match(row)
    if match is None:
        raise ValueError("Unparseable row: {}".format(row))
    wavelength_bytes, flux_bytes, bb_flux_bytes = match.groups()
    wavelength_aa = parse_float(wavelength_bytes)
    flux = np.power(10, parse_float(flux_bytes) + AMES_COND_DILUTION_FACTOR)
    bb_flux = np.power(10, parse_float(bb_flux_bytes) + AMES_COND_DILUTION_FACTOR)
    return wavelength_aa, flux, bb_flux

def apply_ordering_and_units(wls, fluxes, bb_fluxes):
    wls = np.asarray(wls)
    fluxes = np.asarray(fluxes)
    bb_fluxes = np.asarray(bb_fluxes)
    sorter = np.argsort(wls)
    wls = wls[sorter]
    fluxes = fluxes[sorter]
    bb_fluxes = bb_fluxes[sorter]
    return (
        (wls * u.Angstrom).to(u.um),
        (fluxes * ORIG_FLUX_UNITS).to(FLUX_UNITS),
        (bb_fluxes * ORIG_FLUX_UNITS).to(FLUX_UNITS)
    )

def resample_spectrum(orig_wls, orig_fluxes, new_wls):
    unit = orig_fluxes.unit
    wls = orig_wls.to(new_wls.unit).value
    # Some model spectra don't actually cover the full wavelength range
    return interp1d(wls, orig_fluxes.value, fill_value=0.0, bounds_error=False)(new_wls.value) * unit

STACKED_FILENAMES_REGEX = re.compile(r'.*\.spec(\.gz)?$')

def _load_one_spectrum(name, file_handle, row_parser_function, stacked_parser_function):
    if STACKED_FILENAMES_REGEX.match(name):
        try:
            wls, fluxes, bb_fluxes = stacked_parser_function(file_handle)
        except Exception as e:
            print(name, e)
            raise
    else:
        wls, fluxes, bb_fluxes = [], [], []
        for row_bytes in file_handle:
            try:
                wl, f, bb = row_parser_function(row_bytes)
            except ValueError as e:
                print(e)
                continue
            wls.append(wl)
            fluxes.append(f)
            bb_fluxes.append(bb)
    model_wls, model_fluxes, model_bb_fluxes = apply_ordering_and_units(wls, fluxes, bb_fluxes)
    
    # Sanity check temperature
    # peak_wl = model_wls[np.argmax(model_bb_fluxes)]
    # if not np.abs(peak_wl - wien_peak(T_eff * u.K)) < 0.1 * u.um:
    #     raise RuntimeError(f"{peak_wl} vs {wien_peak(T_eff * u.K)} ")
    resampled_fluxes = resample_spectrum(model_wls, model_fluxes, COMMON_WL)
    resampled_bb_fluxes = resample_spectrum(model_wls, model_bb_fluxes, COMMON_WL)
    return resampled_fluxes, resampled_bb_fluxes

def _load_grid_spectrum(archive_filename, filepath_lookup, idx, params, 
                        row_parser_function, stacked_parser_function, decompressor):
    archive_tarfile = tarfile.open(archive_filename)
    T_eff, log_g, M_over_H = params
    filepath = filepath_lookup[params]
    n_spectra = len(filepath_lookup)
    print(f'{idx+1}/{n_spectra} T_eff={T_eff} log g={log_g} M/H={M_over_H}: {filepath}')
    specfile = decompressor(archive_tarfile.extractfile(filepath))
    try:
        resampled_fluxes, resampled_bb_fluxes = _load_one_spectrum(filepath, specfile, row_parser_function, stacked_parser_function)
    except Exception as e:
        print(f'Exception {e} processing {filepath}')
        raise
    return resampled_fluxes, resampled_bb_fluxes

def load_bt_settl_model(filepath):
    with open(filepath, 'rb') as file_handle:
        resampled_fluxes, resampled_bb_fluxes = _load_one_spectrum(
            filepath,
            file_handle,
            parse_bt_settl_row,
            parse_bt_settl_stacked_format
        )
    return COMMON_WL.copy(), resampled_fluxes, resampled_bb_fluxes

def load_ames_cond_model(filepath):
    with open(filepath, 'rb') as file_handle:
        resampled_fluxes, resampled_bb_fluxes = _load_one_spectrum(
            filepath,
            file_handle,
            parse_ames_cond_row,
            parse_ames_cond_stacked_format
        )
    return COMMON_WL.copy(), resampled_fluxes, resampled_bb_fluxes

def load_all_spectra(archive_filename, sorted_params, filepath_lookup, row_parser_function, stacked_parser_function, decompressor):
    n_spectra = len(sorted_params)
    all_spectra = np.zeros((n_spectra,) + COMMON_WL.shape) * FLUX_UNITS
    all_bb_spectra = np.zeros((n_spectra,) + COMMON_WL.shape) * FLUX_UNITS
    loader = partial(_load_grid_spectrum,
        archive_filename=archive_filename,
        filepath_lookup=filepath_lookup,
        row_parser_function=row_parser_function,
        stacked_parser_function=stacked_parser_function,
        decompressor=decompressor
    )
    results = Parallel(n_jobs=-1)(
        delayed(loader)(idx=idx, params=params) for idx, params in enumerate(sorted_params)
    )
    #     # Sanity check temperature
    #     peak_wl = model_wls[np.argmax(model_bb_fluxes)]
    #     if not np.abs(peak_wl - wien_peak(T_eff * u.K)) < 0.1 * u.um:
    #         raise RuntimeError(f"{peak_wl} vs {wien_peak(T_eff * u.K)} ")
        
    #     all_spectra[idx] = resample_spectrum(model_wls, model_fluxes, COMMON_WL)
    #     all_bb_spectra[idx] = resample_spectrum(model_wls, model_bb_fluxes, COMMON_WL)
    for idx, (fluxes, bb_fluxes) in enumerate(results):
        all_spectra[idx] = fluxes
        all_bb_spectra[idx] = bb_fluxes
    return all_spectra, all_bb_spectra


def load_grid(archive_filename, filename_regex, row_parser_function, stacked_parser_function, decompressor, _debug_first_n=None):
    archive_tarfile = tarfile.open(archive_filename)
    filepath_lookup, all_params = make_filepath_lookup(archive_tarfile, filename_regex)
    sorted_params = list(sorted(filepath_lookup.keys()))
    if _debug_first_n is not None:
        sorted_params = sorted_params[:_debug_first_n]
    all_spectra, all_bb_spectra = load_all_spectra(
        archive_filename,
        sorted_params,
        filepath_lookup,
        row_parser_function,
        stacked_parser_function,
        decompressor
    )
    return sorted_params, all_spectra, all_bb_spectra

def convert_grid(archive_filename, filename_regex, row_parser_function, stacked_parser_function, decompressor):
    sorted_params, all_spectra, all_bb_spectra = load_grid(
        archive_filename,
        filename_regex,
        row_parser_function,
        stacked_parser_function,
        decompressor
    )
    hdulist = fits.HDUList([fits.PrimaryHDU(),])
    
    T_eff = [row[0] for row in sorted_params]
    log_g = [row[1] for row in sorted_params]
    M_over_H = [row[2] for row in sorted_params]
    params_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='T_eff', format='E', array=T_eff),
        fits.Column(name='log_g', format='E', array=log_g),
        fits.Column(name='M_over_H', format='E', array=M_over_H),
        fits.Column(name='index', format='J', array=np.arange(len(sorted_params)))
    ])
    params_hdu.header['EXTNAME'] = 'PARAMS'
    hdulist.append(params_hdu)
    
    wls_hdu = fits.ImageHDU(COMMON_WL.value)
    wls_hdu.header['EXTNAME'] = 'WAVELENGTHS'
    wls_hdu.header['UNIT'] = str(COMMON_WL.unit)
    hdulist.append(wls_hdu)

    flux_hdu = fits.ImageHDU(all_spectra.value)
    flux_hdu.header['EXTNAME'] = 'MODEL_SPECTRA'
    flux_hdu.header['UNIT'] = str(all_spectra.unit)
    hdulist.append(flux_hdu)
    
    bb_hdu = fits.ImageHDU(all_bb_spectra.value)
    bb_hdu.header['EXTNAME'] = 'BLACKBODY_SPECTRA'
    bb_hdu.header['UNIT'] = str(all_bb_spectra.unit)
    hdulist.append(bb_hdu)
    
    return hdulist

if __name__ == "__main__":
    import urllib.request
    
    settl_url = 'https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011c/SPECTRA.tar'
    settl_tarfile = './BT-Settl_CIFIST2011c_SPECTRA.tar'
    
    if not os.path.exists(settl_tarfile):
        print(f'Retrieving {settl_tarfile} from {settl_url}')
        urllib.request.urlretrieve(settl_url, settl_tarfile)
    
    settl_fits_output = './BT-Settl_CIFIST2011c_spectra.fits'
    if os.path.exists(settl_fits_output):
        print(f'{settl_fits_output} exists, remove to reprocess')
    else:
        print("Processing BT-Settl models")
        settl_hdul = convert_grid(
            settl_tarfile,
            BT_SETTL_NAME_RE,
            parse_bt_settl_row,
            parse_bt_settl_stacked_format,
            lzma.open
        )
        settl_hdul.writeto(settl_fits_output)

    cond_url = 'https://phoenix.ens-lyon.fr/Grids/AMES-Cond/SPECTRA.tar'
    cond_tarfile = './AMES-Cond-SPECTRA.tar'
    
    if not os.path.exists(cond_tarfile):
        print(f'Retrieving {cond_tarfile} from {cond_url}')
        urllib.request.urlretrieve(cond_url, cond_tarfile)
    
    cond_fits_output = './AMES-Cond_spectra.fits'
    if os.path.exists(cond_fits_output):
        print(f'{cond_fits_output} exists, remove to reprocess')
    else:
        print("Processing AMES-Cond models")
        cond_hdul = convert_grid(
            cond_tarfile,
            AMES_COND_NAME_RE,
            parse_ames_cond_row,
            parse_ames_cond_stacked_format,
            gzip.open
        )
        cond_hdul.writeto(cond_fits_output) 

