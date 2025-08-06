# Installation

## System Requirements

Before installing OpenCL.jl, you need to have an OpenCL driver installed on your system.

### Option 1: System-wide OpenCL Driver

You can install an OpenCL driver system-wide using your package manager:

**Ubuntu/Debian:**
```bash
sudo apt-get install ocl-icd-libopencl1 opencl-headers clinfo
```

**macOS:**
OpenCL is included with the system (no additional installation needed).

**Windows:**
Install drivers from your GPU vendor:
- **NVIDIA**: Download and install the latest NVIDIA drivers
- **AMD**: Download and install the latest AMD drivers
- **Intel**: Download and install Intel OpenCL runtime

### Option 2: PoCL (CPU backend)

For a CPU-only OpenCL implementation that works across platforms, you can use `pocl_jll.jl`:

```julia
using Pkg
Pkg.add("pocl_jll")
```

!!! warning "Platform Loading Order"
    OpenCL computes the list of platforms [only once](https://github.com/KhronosGroup/OpenCL-ICD-Loader/blob/d547426c32f9af274ec1369acd1adcfd8fe0ee40/loader/linux/icd_linux.c#L234-L238).
    Therefore if `using pocl_jll` is executed after `OpenCL.versioninfo()` or other calls to the OpenCL API
    then it won't affect the list of platforms available and you will need to restart the Julia session
    and run `using pocl_jll` before `OpenCL` is used.

## Installing OpenCL.jl

Once you have an OpenCL driver installed, add OpenCL.jl to your Julia environment:

```julia
using Pkg
Pkg.add("OpenCL")
```

## Testing Your Installation

Test your installation by checking the available platforms and devices:

```julia-repl
julia> using OpenCL

julia> OpenCL.versioninfo()
```

You should see output similar to:

```
OpenCL.jl version 0.10.0

Toolchain:
 - Julia v1.10.5
 - OpenCL_jll v2024.5.8+1

Available platforms: 1
 - Portable Computing Language
   version: OpenCL 3.0 PoCL 6.0  Linux, Release, RELOC, SPIR-V, LLVM 15.0.7jl, SLEEF, DISTRO, POCL_DEBUG
   · cpu-haswell-AMD Ryzen 9 5950X 16-Core Processor (fp64, il)
```

If you see available platforms and devices, your installation is working correctly!

## Troubleshooting

### No Platforms Found

If you get an error about no OpenCL platforms being found:

1. Make sure you have OpenCL drivers installed
2. Try using `pocl_jll` as described above
3. On Linux, make sure the OpenCL ICD loader is installed
4. Restart Julia and try again

### Platform-specific Issues

**Linux:**
- Install `clinfo` and run it to verify OpenCL installation
- Make sure `/etc/OpenCL/vendors/` contains ICD files

**macOS:**
- OpenCL is deprecated on macOS but still available
- Consider using Metal.jl for newer Apple hardware

**Windows:**
- Make sure GPU drivers are up to date
- Try running as administrator if permissions issues occur
