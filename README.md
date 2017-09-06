# Getting Started with General Purpose GPU Computing using Some Simple Experiments
           
# Introduction

This is some simple experimental GPU matrix multiplication code, using
raw CUDA coding and arrayfire approaches. The repository includes code,
resulting output on my test machine (an aging commodity Windows laptop
equipped with an i5-3230M CPU @ 2.60GHz and a GeForce GT 740M GPU) and
discussion (in this file). Putting together this material was my first
exposure to GPU programming. Reproducing and understanding the
experiments here might serve as a reasonable first step for someone else
(which is one reason for publishing this on GitHub).

# Program Descriptions

Program arrayfire-matrix-multiply.cpp demonstrates matrix multiplication
using the Arrayfire GPU programming library.
[Arrayfire](https:/github.com/arrayfire/arrayfire) is an open source
library, hosted on Github. It seeks to provide an abstracted interface
to GPU facilities. It includes a variety of "backends", which are
specific implementations on particular platforms. I have built and run
the program with three different backends, and I include the output
generated on my machine with each. The three backends I tried are; CUDA,
CPU and UNIFIED. CUDA is, of course, NVidia's GPU programming model. CPU
implements the facilities on the Intel CPU, without the need for an
actual GPU, but at the cost of performance (obviously). UNIFIED is an
interesting approach, at runtime it probes the machine for the highest
performing hardware available and dynamically switches to that. On a
CUDA equipped machine, it will use the CUDA facilities. The advantage is
that one build can use CUDA if present, but still work on non CUDA
equipped machines.

Program cuda-matrix-multiply.cu demonstrates matrix multiplication using
a "closer to the metal" approach, without the Arrayfire abstraction
layer. This program is built using the NVIDIA Cuda Development Kit
version 8.0. In fact this program is an adaption of sample code provided
with that kit. Again I have included the output generated on my machine
by this program.

# Performance

Here are the timing summaries for a 512x512 single precision floating
point matrix multiplication extracted from the raw cuda, arrayfire
(cuda) and arrayfire (cpu) output files.

### Raw CUDA
- Naive CPU calculation time 209.00 milliseconds
- Simple GPU time 17.08 milliseconds
- Simple GPU time minus host<->device overhead 14.54 milliseconds
- Shared GPU time 8.55 milliseconds
- Shared GPU time minus host<->device overhead 6.01 milliseconds

### Arrayfire CPU
- Naive CPU time 210.94 milliseconds
- Arrayfire GPU time 4.21 milliseconds
- For comparison, arrayfire GPU time using arrayfire hi-res timer 4.42 millisecs

### Arrayfire GPU
- Naive CPU time 207.00 milliseconds
- Arrayfire GPU time 0.99 milliseconds
- For comparison, arrayfire GPU time using arrayfire hi-res timer 1.14 millisecs

### Collating, rounding and ranking
- Arrayfire GPU 1 ms
- Arrayfire CPU 4 ms
- Raw CUDA 6 ms
- Naive CPU time 200 ms

Note that the best apples to apples comparison here excludes the host to
device overhead, since the Arrayfire calculations all occur in GPU
(device) memory. More about host to device transfer in the Arrayfire
review later.

The worst performing calculation is not surprisingly a straightforward
(naive) C/C++ for loop construct. The best performing calculation is the
Arrayfire CUDA calculation, which was 200 times faster. It's not
surprising that the Arrayfire CUDA calculation is a lot faster than the
simple and unoptimised raw CUDA code from the CUDA development kit.
However it is surprising to me at least that Arrayfire also managed to
beat the raw CUDA code without using the GPU. Poking a little into the
Arrayfire source code it appears that Arrayfire actually delegates this
calculation to one of a number of alternative open source "BLAS" (Basic
Linear Algebra Subroutines) libraries. The options appear to be
"OpenBLAS", "MKL", "Atlas" and "LAPACKE". It would be interesting to
drill down into the details, and ultimately presumably a combination of
multithreading and SIMD instructions in the CPU. This is for further
study.

A NxN matrix multiplication entails N multiplications and N additions
for each of NxN elements, so 2*N^3 floating point operations total. For
N=512, 2*N^3 = 2.7e8. Performing that many floating point operations in
1ms implies 2.7e11 Flops or 270 GFlops. Not too shabby for an obsolete
consumer grade device that cost around a thousand dollars perhaps four
years ago. I wonder what 270 GFlops would have cost in the 1970s (say)?

# Notes on Floating Point Precision

I chose matrix multiplication for my first GPU code because I was
anxious to make a start without the need to revisit or learn anything at
all mathematically sophisticated. It's something simple enough that I
can still remember the principles from my undergraduate mathematics.
Perhaps for the same reason there was no shortage of pre-existing code
to tinker with. One thing I didn't like about the pre-existing examples
I saw though was that they didn't bother actually verifying the results!
Typically they would just generate random matrices to multiply. Not
satisfying. I chose instead to use large 512x512 (to make timing easier)
square matrices initialised by the pattern M[row,column] =
M[row+column][row+column]. Writing simple verification code also served
to give me a naive CPU implementation as a timing baseline for free.

One puzzling thing was that the  GPU implementations (both raw CUDA and
Arrayfire) verified perfectly against the naive implementation, but the
Arrayfire CPU implementation did not. I immediately jumped to the
(wrong) conclusion that this meant that there was something wrong with
the CPU implementation. Digging a little deeper I realised that single
precision floating point, with only a 24 bit mantissa was insufficiently
precise to generate 9 digit integer array elements without rounding
errors (2^24 = 16 million, more or less).

As I developed the code I highlighted the (non-)"problem" with a couple
of sample elements, M[0][0] and M[511][511]. Fortunately I can overcome
the self-imposed complexity problem posed my example matrices with a
little bit of nifty maths to clarify the whole situation.

The matrices being multiplied are both M[0][0]=0, M[0][1]=1...
M[0][511]=511,M[1][0]=1,M[1][1]=2...M[511][511]=1022. So M[0][0] of the
product is 0^2 + 1^2 + 2^2 .... + 511^2 and M[511][511] of the product
is 511^2 + 512^2 .... + 1022^2.

Since 1^2 + 2^2 + 3^2 + ... n^2 = n(n+1)(2n+1)/6  (from the much
neglected Schaum's Mathematical Handbook on my bookshelf), M[0][0] =
n(n+1)(2n+1)/6 for n=511 and M[511][511] = m(m+1)(2m+1)/6 -
n(n+1)(2n+1)/6 for m=1022 and n=510. This gives us, exactly

M[0][0] = 44608256 and
M[511][511] = 311996160

So the naive CPU and all GPU implementation calculations have a small error;

M[0][0] = 44608184, relative error 1.6e-6, one part in 0.6 million
M[511][511] = 311995968, relative error 6.1e-7, one part in 1.6 million

And the Arrayfire CPU implementation calculations have different small errors;

M[0][0] = 44608204, relative error 1.2e-6 one part in 0.8 million
M[511][511] = 311995904, relative error 8.2e-7 one part in 1.2 million.

# A Brief Review of Arrayfire

The Arrayfire github page outlines the purpose of Arrayfire rather well;

Quoting directly;

ArrayFire is a general-purpose library that simplifies the process of developing software that targets parallel and massively-parallel architectures including CPUs, GPUs, and other hardware acceleration devices.

Several of ArrayFire's benefits include:

- Easy to use, stable, well-documented API
- Rigorously tested for performance and accuracy
- Commercially friendly open-source licensing
- Commercial support from ArrayFire

ArrayFire provides software developers with a high-level abstraction of data which resides on the accelerator, the af::array object. Developers write code which performs operations on ArrayFire arrays which, in turn, are automatically translated into near-optimal kernels that execute on the computational device.

ArrayFire is successfully used on devices ranging from low-power mobile phones to high-power GPU-enabled supercomputers. ArrayFire runs on CPUs from all major vendors (Intel, AMD, ARM), GPUs from the prominent manufacturers (NVIDIA, AMD, and Qualcomm), as well as a variety of other accelerator devices on Windows, Mac, and Linux.

End of Quote.

Arrayfire has 1.8K stars on Github, and an [active forum on Google Groups](http://go.arrayfire.com/e/37882/2017-08-03/5d32fq/584044717forum/#!forum/arrayfire-users)

The company behind Arrayfire (also called Arrayfire) is to be commended on open sourcing a mature commercial product
using best practice methods (Github hosting, liberal licencing). A lot of effort has gone into packaging the library
professionally for Windows, Mac and Linux. Usually you do not need to build from source.

My experiences with Arrayfire were a little mixed, but generally positive. The product is thoroughly documented, but
using some kind of automatically-generated-from-source-comments tooling. This method means that the documentation
is often frustratingly terse, and after a while you learn that the automatically generated "click here for more information"
links tend to be empty promises (there is no more information). Going to the API header files doesn't help, they contain
the same information, if anything obfuscated by the generation tooling. For an example of the difficulties, consider
the issue of transferring data between host and device. To use a GPU to provide an accelerated calculation for your normal
CPU hosted code involves first a host->device transfer, then a device->host transfer. Even for my simple test I needed to
perform these steps, but I was frustrated because the Arrayfire examples focus on manipulating data that is already
in the GPU (device) memory. Eventually I worked out how to perform the transfers, but if you take a look at the
host->device code in af_test_init() and the device->host code in af_test_verify() you will see that write() and host()+freeHost()
methods (respectively) are not particularly coherent or regular. They and are also rather feebly documented.

Ostensibly Arrayfire is a C++ library, but the online documentation doesn't offer class reference information, only
function reference information. This is quite telling, it's really a C library in disguise in my opinion (this is not
necessarily a bad thing). In other words, getting things done in Arrayfire typically involves invoking functions
on af::array object references, and infrequently involves using af::array methods.

Fortunately there is a wealth of example code provided and it's not difficult to get started by simply adapting
some appropriate example. Arrayfire appears to be wide and deep, example code is provided in machine learning,
linear algebra, computer vision, image processing, financial calculations and graphics realms.

Obviously this mini-project is far from sufficient to determine how Arrayfire would measure up to a much more
demanding examination.

# Reproducing these Experiments

I have restricted myself to presenting just two .cu and .cpp files here without any ancillary support files, because
it really doesn't make sense to duplicate supporting detailed boilerplate files and associated installation instructions.
The following meta instructions should be sufficient (Note that I used Windows - adjust as necessary for a different OS);

1) To reproduce the raw cuda experiment, first install the NVidia's CUDA Development Kit, build one of NVidia's simple
samples (I used simpleMultiCopy) then replace the sample .cu file with cuda-matrix-multiply.cu. Note that (at least
as of Version 8.0) of the toolkit, you can drill down into the samples directory hierarchy and find a solution/project
for a single sample - there's no need to deal with the complexity inherent in building a whole group of samples at once.

2) To reproduce the Arrayfire experiment, first install the prerequisites then the binary Arrayfire installer. The
prerequisites are required if you want to use back ends other than CPU. Step 1) above doubles as installing sufficient
prerequisites for the CUDA and UNIFIED back ends. There is another back end I haven't mentioned yet, OPENCL. This is
how support is provided for non NVidia GPUs. Personally, I haven't tried to use the OPENCL back end at all. First
make sure you can build the helloworld example successfully, then replace helloworld.cpp with arrayfire-matrix-multiply.cpp

It is convenient that the NVidia CUDA toolkit samples come with multiple project and solution files for all recent versions of Visual
Studio. In contrast the equivalent Arrayfire examples have a single Visual Studio 2013 project and solution file. I didn't
have any problem opening this file with a more recent version of Visual Studio though.

I did fall into a world of pain though when I tried to rebuild Arrayfire
from source. I wanted to build Arrayfire version 3.4.3 because it
supports the older CUDA toolkit version 6.5 (at the time of writing the
most recent version Arrayfire is V3.5.0 and the most recent version of
the CUDA toolkit is version 8.0). Why did I want to use older versions
of both systems? Because I was using another laptop with an even older,
slower NVidia GPU (the GeForce 320M) and it is not supported by Cuda
toolkit V8.0. Basically all my attempts foundered in a sea of dependency
hellfire - as the brutally complicated CMake scripts complained about
different and slightly incompatible Visual Studio versions in code I was
compiling versus the various pre-built dependencies that you need to do
a full Arrayfire build (remember the stuff about OpenBlas, MKL, LAPACKE
etc. above?). To be honest, this is not an unusual scenario when trying
to build massive open source projects on Windows (it's definitely an
afterthought for most open source developers, even if this is not the
case for Arrayfire itself you are still dealing with multiple
dependencies). I would probably have prevailed eventually, if I had not
had the easy bail out option of another machine which could use the
latest software versions.

I haven't investigated how difficult it is to distribute Windows applications
built with Arrayfire to end users without requiring them to download the
Arrayfire prerequisites (basically the NVidia toolkit presumably to, allow
use of the UNIFIED backend to dynamically switch between CUDA and CPU according
to the presence or absence of NVidia GPU hardware).

# Next Steps

If I develop this project further my next two steps would be;

1) Improve my understanding of exactly how the raw cuda matrix calculations work

2) Make another attempt at rebuilding Arrayfire from source, with a view to
drilling down into the implementation details with a source level debugger.
