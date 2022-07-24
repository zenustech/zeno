What is Aquila?
===============

Aquila is an open source and cross-platform DSP (Digital Signal Processing)
library for C++11.

[![Build Status](https://travis-ci.org/zsiciarz/aquila.png?branch=master)](https://travis-ci.org/zsiciarz/aquila)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/github/zsiciarz/aquila?branch=master&svg=true)](https://ci.appveyor.com/project/zsiciarz/aquila)
[![Coverage Status](https://coveralls.io/repos/zsiciarz/aquila/badge.png)](https://coveralls.io/r/zsiciarz/aquila)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/2786/badge.svg)](https://scan.coverity.com/projects/2786)


Example
=======

```cpp
#include "aquila/aquila.h"

int main()
{
    // input signal parameters
    const std::size_t SIZE = 64;
    const Aquila::FrequencyType sampleFreq = 2000, f1 = 125, f2 = 700;

    Aquila::SineGenerator sine1(sampleFreq);
    sine1.setAmplitude(32).setFrequency(f1).generate(SIZE);
    Aquila::SineGenerator sine2(sampleFreq);
    sine2.setAmplitude(8).setFrequency(f2).setPhase(0.75).generate(SIZE);
    auto sum = sine1 + sine2;

    Aquila::TextPlot plot("Input signal");
    plot.plot(sum);

    // calculate the FFT
    auto fft = Aquila::FftFactory::getFft(SIZE);
    Aquila::SpectrumType spectrum = fft->fft(sum.toArray());

    plot.setTitle("Spectrum");
    plot.plotSpectrum(spectrum);

    return 0;
}
```

For more usage examples see the `examples` directory or
[browse them online](http://aquila-dsp.org/articles/examples/).


Features
========

 * various signal sources (generators, text/binary/WAVE files)
 * signal windowing and filtering
 * performing operations on a frame-by-frame basis
 * calculating energy, power, FFT and DCT of a signal
 * feature extraction, including MFCC and HFCC features, widely used in
   speech recognition
 * pattern matching with DTW (dynamic time warping) algorithm


Requirements
============

The following dependencies are required to build the library from source.

 * a modern C++11 compiler
 * CMake >= 2.8


Contributing
============

See CONTRIBUTING.md for some guidelines how to get involved.


License
=======

The library is licensed under the MIT (X11) license. A copy of the license
is distributed with the library in the LICENSE file.


Authors
=======

Zbigniew Siciarz (zbigniew at siciarz dot net)

This library includes code from Takuya Ooura's mathematical software packages,
which are available at http://www.kurims.kyoto-u.ac.jp/~ooura/.
