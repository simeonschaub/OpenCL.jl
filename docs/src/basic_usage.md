# Basic Usage

This section covers the fundamental concepts and basic usage patterns of OpenCL.jl.

## Core Concepts

### Arrays and Memory Management

OpenCL.jl provides `CLArray` for GPU memory management:

```julia
using OpenCL

# Create arrays on the device
a = CLArray([1.0f0, 2.0f0, 3.0f0, 4.0f0])

# Or transfer from host
host_array = rand(Float32, 1000)
device_array = CLArray(host_array)

# Transfer back to host
result = Array(device_array)
```

### Writing OpenCL C Kernels

The traditional way to use OpenCL is by writing kernel source code in OpenCL C:

```julia
using OpenCL, pocl_jll

# Define kernel source
const source = """
   __kernel void vadd(__global const float *a,
                      __global const float *b,
                      __global float *c) {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }"""

# Create input data
a = rand(Float32, 50_000)
b = rand(Float32, 50_000)

# Transfer to device
d_a = CLArray(a)
d_b = CLArray(b)
d_c = similar(d_a)  # allocate output array

# Compile and build the program
p = cl.Program(; source) |> cl.build!
k = cl.Kernel(p, "vadd")

# Execute the kernel
clcall(k, Tuple{CLPtr{Float32}, CLPtr{Float32}, CLPtr{Float32}},
       d_a, d_b, d_c; global_size=size(a))

# Get results
c = Array(d_c)

@assert a + b ≈ c
```

### Kernel Execution Parameters

When calling kernels, you can specify various execution parameters:

```julia
# Basic execution with global work size
clcall(kernel, argtypes, args...; global_size=(1024,))

# With local work size (work group size)
clcall(kernel, argtypes, args...;
       global_size=(1024,), local_size=(64,))

# Multi-dimensional work
clcall(kernel, argtypes, args...;
       global_size=(256, 256), local_size=(16, 16))

# Using a specific command queue
queue = cl.CmdQueue()
clcall(kernel, argtypes, args...;
       global_size=(1024,), queue=queue)
```

### Memory Types and Pointers

OpenCL.jl supports different memory address spaces:

```julia
# Global memory (default for CLArray)
CLPtr{Float32}  # __global float*

# Local/shared memory
# (allocated per work group, used for communication between work items)

# Private memory
# (per work item, typically registers or stack)
```

### Error Handling

OpenCL operations can throw exceptions. Always handle potential errors:

```julia
try
    # OpenCL operations
    result = Array(device_array)
catch e
    if isa(e, OpenCL.CLError)
        println("OpenCL error: ", e)
    else
        rethrow()
    end
end
```

### Platform and Device Selection

You can query and select specific platforms and devices:

```julia
# List all platforms
platforms = cl.platforms()

# Get devices for a platform
devices = cl.devices(platforms[1])

# Create context with specific device
ctx = cl.Context(devices[1])

# Set as current context
cl.device!(devices[1])
```

### Memory Transfer Patterns

Efficient memory management is crucial for performance:

```julia
# Synchronous transfer (blocking)
host_data = Array(device_array)

# Asynchronous transfer (non-blocking)
queue = cl.CmdQueue()
event = cl.enqueue_read_buffer(queue, device_array.data, host_data)
cl.wait(event)

# Mapped memory (when supported)
# allows zero-copy access in some cases
```

## Best Practices

1. **Reuse arrays**: Avoid frequent allocation/deallocation
2. **Batch operations**: Minimize kernel launch overhead
3. **Optimize work group sizes**: Usually multiples of 32 or 64
4. **Profile your code**: Use OpenCL profiling tools
5. **Handle errors gracefully**: Always check for OpenCL errors
