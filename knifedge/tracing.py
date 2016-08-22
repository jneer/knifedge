import numpy as np
import attr

@attr.s
class BeamProfile:
    """A cross-sectional profile of a Gaussian beam.
    """

    z = attr.ib(default=0)
    w = attr.ib(default=1)
    errors = attr.ib(default={'z': None, 'w': None})


def profile_1684(z=0, x16=0, x84=1, error_x=None, error_z=None):
    """Create BeamProfile from a 16/84 measurement.
    """

    data = {'method': '1684', 'inputs': locals()}

    w = abs(x84 - x16)
    if error_x is not None:
        error_w = np.sqrt(2) * error_x
    else:
        error_w = None
    errors = {'z': error_z, 'w': error_w}

    profile = BeamProfile(z, w, errors)
    profile._data = data

    return profile




class BeamProfileSampled(BeamProfile):
    """Beam profile sampled by intensity measurements at multiple positions
    of the knife edge.
    """


class BeamTrace:
    """A trace of the size of a Gaussian beam along its axis.
    """

    def __init__(self):
        pass


if __name__ == '__main__':
    p1 = BeamProfile()
    print(p1)

    p2 = profile_1684(0, .3, .12)
    print(p2)
    print(p2._data)