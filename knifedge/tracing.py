# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit
import attr
import matplotlib.pyplot as plt
import matplotlib
import yaml

matplotlib.rc('font', family='DejaVu Sans')

#TODO: use ODR instead of curve_fit to include z-error: http://stackoverflow.com/questions/26058792/correct-fitting-with-scipy-curve-fit-including-errors-in-x
#

@attr.s
class BeamProfile:
    """A cross-sectional profile of a Gaussian beam.
    """

    z = attr.ib(default=0)
    w = attr.ib(default=1)
    err_z = attr.ib(default=0)
    err_w = attr.ib(default=0)


def profile_8416(z=0, x84=0, x16=1, err_x=0, err_z=0):
    """Create BeamProfile from a 84/16 measurement.
    """

    data = {'method': '1684', 'inputs': locals()}

    w = abs(x84 - x16)
    if err_x is not None:
        err_w = np.sqrt(2) * err_x
    else:
        err_w = None

    profile = BeamProfile(z, w, err_z, err_w)
    profile._data = data

    return profile


def profile_9010(z=0, x90=0, x10=1, err_x=0, err_z=0):
    """Create BeamProfile from a 90/10 measurement.
    """

    data = {'method': '1090', 'inputs': locals()}

    w = 1.28 * abs(x90 - x10)
    if err_x is not None:
        err_w = np.sqrt(2) * err_x
    else:
        err_w = None

    profile = BeamProfile(z, w, err_z, err_w)
    profile._data = data

    return profile


def traces_from_file(filename):
    with open(filename, 'r') as file:
        yaml_data = list(yaml.safe_load_all(file))

    for data in yaml_data:
        print(data)


class BeamProfileSampled(BeamProfile):
    """Beam profile sampled by intensity measurements at multiple positions
    of the knife edge.
    """


@attr.s
class BeamTrace:
    """A trace of the size of a Gaussian beam along its axis.
    """

    label = attr.ib(default="")
    wavelength = attr.ib(default=.001550)
    profiles = attr.ib(default=[])
    fit_params = attr.ib(default=None)
    fit_params_error = attr.ib(default=None)

    def __init__(self):
        self.profiles = []
        self.fit_params = None

    def add_profile(self, profile, update_fit=True):
        self.profiles.append(profile)
        self.sort_profiles()
        if update_fit:
            self.fit_trace()

    def sort_profiles(self):
        self.profiles.sort(key=lambda _: _.z)

    def spotsize(self, z, z0, w0, m2=1):
        return w0 * np.sqrt(1 + ((z - z0) / (np.pi * w0**2 /
                                             self.wavelength / m2))**2)

    def fit_trace(self, p0=None):
        z = [p.z for p in self.profiles]
        w = [p.w for p in self.profiles]
        err_w = [p.err_w for p in self.profiles]
        sigma = err_w if all(err_w) else None
        absolute_sigma = all(err_w)
        bounds = ([-np.inf, 0, 1], [np.inf, np.inf, np.inf])
        popt, pcov = curve_fit(self.spotsize, z, w, p0, sigma, absolute_sigma,
                               bounds=bounds)
        self.fit_params = popt
        self.fit_params_error = np.sqrt(np.diag(pcov))

        print(self.format_fit_result())

    def plot_trace(self):
        z = [p.z for p in self.profiles]
        w = [p.w for p in self.profiles]
        err_w = [p.err_w for p in self.profiles]
        plt.errorbar(z, w, err_w, fmt='.k')
        plt.xlabel('z [mm]')
        plt.ylabel('w [mm]')

        if self.fit_params is not None:
            zs = np.linspace(min(z), max(z), 200)
            ws = self.spotsize(zs, *self.fit_params)
            plt.plot(zs, ws)
            plt.text(.1, .9, self.format_fit_result(),
                     verticalalignment='top',
                     transform=plt.gca().transAxes,
                     bbox=dict(facecolor='red', alpha=0.2))

    def format_fit_result(self):
        p_strings = ['z₀: {:.1f} ± {:.1f} mm',
                     'w₀: {:.4f} ± {:.4f} mm',
                     'M²: {:.2f} ± {:.2f}']
        return '\n'.join([s.format(p, e) for s, p, e in
                          zip(p_strings, self.fit_params,
                                         self.fit_params_error)])


def test_code():


    traces_from_file('test-tracings.yml')

    import random
    error = .03
    t = BeamTrace("test", .001550)
    z = np.linspace(-200, 800, 7)
    w = t.spotsize(z, 100, .3, 1) * np.random.normal(1, error, len(z))
    err_z = np.zeros(len(z))
    err_w = np.ones(len(z)) * error

    profiles = list(map(BeamProfile, z, w, err_z, err_w))
    t = BeamTrace("test", .001550, profiles)

    print(t)

    # for p in [p1, p2, p3, p4]:
    #     t.add_profile(p, update_fit=True)

    # print(t)

    t.fit_trace()


    t.plot_trace()
    plt.show()


if __name__ == '__main__':
    test_code()