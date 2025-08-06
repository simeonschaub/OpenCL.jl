# OpenCL.jl

*Julia interface for the OpenCL parallel computation API*

This package aims to be a complete solution for OpenCL programming in Julia, similar in
scope to [PyOpenCL](http://mathema.tician.de/software/pyopencl/) for Python. It provides a high level API for OpenCL to make programing
hardware accelerators, such as GPUs, FPGAs, and DSPs, as well as multicore CPUs much less
onerous.

!!! warning "Help Wanted"
    OpenCL.jl needs your help! If you can help maintaining this package, please reach out on
    the [JuliaLang Slack](https://julialang.org/slack/) #gpu channel

!!! note "Major Changes"
    OpenCL.jl is currently undergoing major changes. If you have old code,
    developed for OpenCL.jl v0.9, please check [`NEWS.md`](https://github.com/JuliaGPU/OpenCL.jl/blob/master/NEWS.md) for an overview of the
    changes.

## Quick Start

After [installation](@ref Installation), you can verify your setup:

```julia-repl
julia> using OpenCL

julia> OpenCL.versioninfo()
OpenCL.jl version 0.10.0

Toolchain:
 - Julia v1.10.5
 - OpenCL_jll v2024.5.8+1

Available platforms: 3
 - Portable Computing Language
   version: OpenCL 3.0 PoCL 6.0  Linux, Release, RELOC, SPIR-V, LLVM 15.0.7jl, SLEEF, DISTRO, POCL_DEBUG
   · cpu-haswell-AMD Ryzen 9 5950X 16-Core Processor (fp64, il)
 - NVIDIA CUDA
   version: OpenCL 3.0 CUDA 12.6.65
   · NVIDIA RTX 6000 Ada Generation (fp64)
 - Intel(R) OpenCL Graphics
   version: OpenCL 3.0
   · Intel(R) Arc(TM) A770 Graphics (fp16, il)
```

## Simple Vector Addition Example

Here's a quick example of vector addition using OpenCL C kernels:

```julia
using OpenCL, pocl_jll

const source = """
   __kernel void vadd(__global const float *a,
                      __global const float *b,
                      __global float *c) {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }"""

a = rand(Float32, 50_000)
b = rand(Float32, 50_000)

d_a = CLArray(a)
d_b = CLArray(b)
d_c = similar(d_a)

p = cl.Program(; source) |> cl.build!
k = cl.Kernel(p, "vadd")

clcall(k, Tuple{CLPtr{Float32}, CLPtr{Float32}, CLPtr{Float32}},
       d_a, d_b, d_c; global_size=size(a))

c = Array(d_c)

@assert a + b ≈ c
```

## Native Julia Kernels

If your platform supports SPIR-V, you can write kernels directly in Julia:

```julia
using OpenCL, pocl_jll

function vadd(a, b, c)
    gid = get_global_id(1)
    @inbounds c[gid] = a[gid] + b[gid]
    return
end

a = rand(Float32, 50_000)
b = rand(Float32, 50_000)

d_a = CLArray(a)
d_b = CLArray(b)
d_c = similar(d_a)

@opencl global_size=size(a) vadd(d_a, d_b, d_c)

c = Array(d_c)

@assert a + b ≈ c
```

## Credit

This package is heavily influenced by the work of others:

  * [PyOpenCL](http://mathema.tician.de/software/pyopencl/) by Andreas Klockner
  * [oclpb](https://github.com/srossross/oclpb) by Sean Ross
  * [Boost.Compute](https://github.com/kylelutz/compute) by Kyle Lutz
  * [rust-opencl](https://github.com/luqmana/rust-opencl)
