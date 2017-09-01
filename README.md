# gpu-experiments

This is some simple experimental GPU matrix multiplication code, using raw CUDA coding and arrayfire approaches

Program arrayfire-matrix-multiply.cpp demonstrates matrix multiplication using the Arrayfire GPU programming
library. [Arrayfire](https:/github.com/arrayfire/arrayfire) is an open source library, hosted on github. It
seeks to provide an abstracted interface to GPU facilities. It includes a variety of "backends", which are
specific implementations on particular platforms. I have built and run the program with three different
backends, and I include the output generated on my machine with each. The three backends I tried are; CUDA,
CPU and UNIFIED. CUDA is, of course, NVidia's GPU programming model. CPU implements the facilities on the
Intel CPU, without the need for an actual GPU, but at the cost of performance (obviously). UNIFIED is an
interesting approach, at runtime it probes the machine for the highest performing hardware available and
dynamically switches to that. On a CUDA equipped machine, it will use the CUDA facilities. The advantage is
that one build can use CUDA if present, but still work on non CUDA equipped machines.

Program cuda-matrix-multiply.cu demonstrates matrix multiplication using a "closer to the metal" approach,
without the arrayfire abstraction layer. This program is built using the NVIDIA Cuda Development Kit version
8.0. In fact this program is an adaption of sample code provided with that kit. Again I have included the
output generated on my machine by this program.

