# KnifEdge

The beam width parameter of a laser beam with a Gaussian profile can be
measured with the _knife-edge method_.

This package includes a general class for plotting and fitting of beam profiles,
along with more specific classes for different methods of obtaining the
profiles: swinging ruler, manual micrometer positioning, my custom LEGO NXT
profiler.

## Installation

    python setup.py develop

## Usage

    import knifedge as ke

### Manual measurements in text file

For manual knife-edge measurements using the 84/16 (or 90/10) method,
the easiest usage is to enter the measurements and their metadata into
a YAML formatted document with the following fields:

    ---
    label: after 150mm lens
    method: '90/10'
    z_offset: 313
    x_error: 0
    wavelength: 1550
    dz: [0, 25, 50, 100, 150, 300]
    measurements:
    - [12.301, 13.035]
    - [11.512, 12.086]
    - [11.906, 12.359]
    - [11.748, 11.945]
    - [12.630, 12.857]
    - [11.494, 12.520]
    
* `label`: Text describing the beam.
* `method`: _'90/10'_ or _'84/16'_. Other methods will be implemented 
later.
* `z_offset`: Distance (in mm) to a fixed reference point on the table.
* `x_error`: Optional: Estimate of the RMS error in determining the 
x-values. Default value is 0 if not states.
* `wavelength`: Wavelength of the light in nm.
* `dz`: Positions along the beam (in mm) at which the beam sizes have
been measured. Must be same length as `measurements`.
* `measurements`: List of two-element lists of the measured x positions
of the knife-edge at which the laser beam intensity reaches the 84% and 
16% levels (or 90%/10%) of its full intensity.

Multiple traces can be contained in the same file, separated by the
YAML document separator _---_.

Loading and fitting is done in a single line, plotting can be done
afterwards:

    traces = ke.traces_from_file('path/to/measurement.yml')
    for trace in traces:
        ke.plot_trace(trace)
        
