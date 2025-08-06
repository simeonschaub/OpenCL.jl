# Native Julia Kernels

One of the most powerful features of OpenCL.jl is the ability to write kernels directly in Julia, without needing to write OpenCL C code. This requires a platform that supports SPIR-V.

## The `@opencl` Macro

The `@opencl` macro compiles Julia functions to OpenCL kernels:

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

# Launch the kernel
@opencl global_size=size(a) vadd(d_a, d_b, d_c)

c = Array(d_c)
@assert a + b ≈ c
```

## Kernel Functions

### Work Item Functions

Access work item information using OpenCL built-in functions:

```julia
function kernel_example(data)
    # Get current work item indices
    gid = get_global_id(1)        # Global ID in dimension 1
    lid = get_local_id(1)         # Local ID within work group
    gsize = get_global_size(1)    # Total number of work items
    lsize = get_local_size(1)     # Work group size

    # Work with the data
    if gid <= length(data)
        @inbounds data[gid] *= 2.0f0
    end
    return
end
```

### Multi-dimensional Kernels

Work with 2D and 3D data:

```julia
function matrix_kernel(matrix)
    i = get_global_id(1)  # row
    j = get_global_id(2)  # column

    if i <= size(matrix, 1) && j <= size(matrix, 2)
        @inbounds matrix[i, j] *= 2.0f0
    end
    return
end

# Launch with 2D work size
matrix = CLArray(rand(Float32, 256, 256))
@opencl global_size=size(matrix) matrix_kernel(matrix)
```

### Local Memory

Use local (shared) memory for communication within work groups:

```julia
function reduction_kernel(input, output)
    lid = get_local_id(1)
    gid = get_global_id(1)
    lsize = get_local_size(1)

    # Declare local memory (shared within work group)
    local_data = @localmem Float32 (256,)  # Adjust size as needed

    # Load data into local memory
    if gid <= length(input)
        @inbounds local_data[lid] = input[gid]
    else
        @inbounds local_data[lid] = 0.0f0
    end

    # Synchronize work items in the work group
    barrier()

    # Perform reduction in local memory
    stride = lsize ÷ 2
    while stride > 0
        if lid <= stride
            @inbounds local_data[lid] += local_data[lid + stride]
        end
        barrier()
        stride ÷= 2
    end

    # Write result
    if lid == 1
        @inbounds output[get_group_id(1)] = local_data[1]
    end
    return
end
```

## Compilation Options

### Kernel Configuration

Control kernel compilation with keyword arguments:

```julia
# Specify kernel name
@opencl kernel="my_kernel" global_size=(1024,) my_function(args...)

# Always inline functions
@opencl always_inline=true global_size=(1024,) my_function(args...)

# Launch vs. compilation only
kernel = @opencl launch=false my_function(args...)  # Compile only
kernel(args...; global_size=(1024,))                # Launch later
```

### Type Constraints

Ensure type stability for better performance:

```julia
function typed_kernel(a::CLDeviceArray{Float32},
                     b::CLDeviceArray{Float32},
                     c::CLDeviceArray{Float32})
    gid = get_global_id(1)
    if gid <= length(a)
        @inbounds c[gid] = a[gid] + b[gid]
    end
    return
end
```

## Advanced Features

### Atomic Operations

Use atomic operations for thread-safe updates:

```julia
function atomic_increment(counters, data)
    gid = get_global_id(1)

    if gid <= length(data)
        value = @inbounds data[gid]
        if value > 0.5f0
            # Atomically increment counter
            atomic_add!(pointer(counters, 1), Int32(1))
        end
    end
    return
end
```

### Mathematical Functions

Many Julia math functions are supported:

```julia
function math_kernel(input, output)
    gid = get_global_id(1)

    if gid <= length(input)
        @inbounds x = input[gid]
        @inbounds output[gid] = sin(x) * cos(x) + sqrt(abs(x))
    end
    return
end
```

## Best Practices for Native Kernels

1. **Keep kernels simple**: Complex control flow can hurt performance
2. **Use `@inbounds`**: Skip bounds checking for better performance
3. **Minimize memory access**: Coalesce memory accesses when possible
4. **Use local memory**: For data reuse within work groups
5. **Profile and optimize**: Use OpenCL profiling tools
6. **Handle edge cases**: Check array bounds to avoid out-of-bounds access

## Limitations

- Not all Julia features are supported in kernels
- Dynamic memory allocation is not allowed
- Exception handling is limited
- Some standard library functions may not be available
- Type inference must be successful at compile time
