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
    z_error = attr.ib(default=0)
    w_error = attr.ib(default=0)


class BeamProfileSampled(BeamProfile):
    """Beam profile sampled by intensity measurements at multiple positions
    of the knife edge.
    """


@attr.s
class BeamTrace:
    """A trace of the size of a Gaussian beam along its axis.
    """

    label = attr.ib(default="")
    wavelength = attr.ib(default=1550)
    profiles = attr.ib(default=attr.Factory(list))
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
        zR = np.pi * w0**2 / (1e-6 * self.wavelength * m2)
        return w0 * np.sqrt(1 + ((z - z0) / zR)**2)

    def fit_trace(self, p0=None):
        z = [p.z for p in self.profiles]
        w = [p.w for p in self.profiles]
        if p0 is None:
            p0 = [z[w.index(min(w))],
                  min(w),
                  1]
        w_error = [p.w_error for p in self.profiles]
        sigma = w_error if all(w_error) else None
        absolute_sigma = all(w_error)
        bounds = ([-np.inf, 0, 1], [np.inf, np.inf, np.inf])
        popt, pcov = curve_fit(self.spotsize, z, w, p0, sigma, absolute_sigma,
                               bounds=bounds)
        self.fit_params = popt
        self.fit_params_error = np.sqrt(np.diag(pcov))

        print(self.format_fit_result())

    def format_fit_result(self):
        p_strings = ['z₀: {:.1f} ± {:.1f} mm',
                     'w₀: {:.4f} ± {:.4f} mm',
                     'M²: {:.2f} ± {:.2f}']
        return '\n'.join([s.format(p, e) for s, p, e in
                          zip(p_strings, self.fit_params,
                                         self.fit_params_error)])


def profile_8416(z=0, x84=0, x16=1, x_error=0, z_error=0):
    """Create BeamProfile from a 84/16 measurement.
    """

    data = {'method': '90/10', 'inputs': locals()}

    w = abs(x84 - x16)
    if x_error is not None:
        w_error = np.sqrt(2) * x_error
    else:
        w_error = None

    profile = BeamProfile(z, w, z_error, w_error)
    profile._data = data

    return profile


def profile_9010(z=0, x90=0, x10=1, x_error=0, z_error=0):
    """Create BeamProfile from a 90/10 measurement.
    """

    data = {'method': '90/10', 'inputs': locals()}

    w = abs(x90 - x10) / 1.28
    if x_error is not None:
        w_error = np.sqrt(2) * x_error
    else:
        w_error = None

    profile = BeamProfile(z, w, z_error, w_error)
    profile._data = data

    return profile


def traces_from_file(filename):
    with open(filename, 'r') as file:
        yaml_data = list(yaml.safe_load_all(file))

    traces = []

    for trace_data in yaml_data:
        try:
            z_offset = trace_data['z_offset']
            dz = trace_data['dz']
            measurements = trace_data['measurements']
            label = trace_data['label']
            wavelength = trace_data['wavelength']
            method = trace_data['method']
        except KeyError as err:
            print('Missing key:', err)
            return

        assert(len(dz) == len(measurements))
        assert(method in ['90/10', '84/16'])

        trace = BeamTrace(label, wavelength)

        if method == '84/16':
            x_error = trace_data.get('x_error', 0)
            z_error = trace_data.get('z_error', 0)
            for _dz, _meas in zip(dz, measurements):
                trace.add_profile(profile_8416(z_offset + _dz, *_meas,
                                               x_error=x_error,
                                               z_error=z_error),
                                  update_fit=False)

        if method == '90/10':
            x_error = trace_data.get('x_error', 0)
            z_error = trace_data.get('z_error', 0)
            for _dz, _meas in zip(dz, measurements):
                trace.add_profile(profile_9010(z_offset + _dz, *_meas,
                                               x_error=x_error,
                                               z_error=z_error),
                                  update_fit=False)

        print('\nBeam trace:', label)
        print('Method: {} | z_offset: {} mm | Wavelength: {} nm'.format(
            method, z_offset, wavelength))
        print('--- Fit result from {} profiles: ---'.format(len(dz)))
        trace.fit_trace()
        print('------------------------------------')
        traces.append(trace)

    return traces


def plot_trace(trace, fig=None, ax=None, figsize=(8, 6)):
    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    z = [p.z for p in trace.profiles]
    w = [p.w for p in trace.profiles]
    w_error = [p.w_error for p in trace.profiles]

    ax.errorbar(z, w, w_error, fmt='.k')
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('w [mm]')
    ax.set_ylim(ymin=0)
    ax.set_title(trace.label)

    if trace.fit_params is not None:
        zs = np.linspace(min(z), max(z), 200)
        ws = trace.spotsize(zs, *trace.fit_params)
        ax.plot(zs, ws)
        ax.text(.1, .9, trace.format_fit_result(),
                 verticalalignment='top',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='red', alpha=0.2))

    return fig, ax


def test_code():


    traces = traces_from_file('test-tracings.yml')
    print(traces)

    plot_trace(traces[0])
    plt.show()

    # error = .03
    # t = BeamTrace("test", 1550)
    # z = np.linspace(0, 600, 7)
    # w = t.spotsize(z, 100, .3, 1) * np.random.normal(1, error, len(z))
    # z_error = np.zeros(len(z))
    # w_error = np.ones(len(z)) * error
    #
    # print(z)
    # print(w)
    #
    # profiles = list(map(BeamProfile, z, w, z_error, w_error))
    # t = BeamTrace("test", .001550, profiles)
    #
    # print(t)
    #
    # for p in [p1, p2, p3, p4]:
    #     t.add_profile(p, update_fit=True)

    # print(t)

    # t.fit_trace()


    # t.plot_trace()
    # plt.show()


if __name__ == '__main__':
    test_code()