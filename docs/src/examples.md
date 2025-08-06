# Examples

This section provides various examples demonstrating different aspects of OpenCL.jl usage.

## Repository Examples

OpenCL.jl comes with several examples in the `examples` folder. You can access them by:

```julia
using OpenCL
cd(joinpath(dirname(pathof(OpenCL)), ".."))
```

Or view them online at the [OpenCL.jl repository](https://github.com/JuliaGPU/OpenCL.jl/tree/master/examples).

### Performance Benchmark

A simple performance comparison between CPU and OpenCL:

```julia
# examples/performance.jl
using OpenCL, BenchmarkTools

function cpu_vadd!(a, b, c)
    for i in eachindex(a)
        c[i] = a[i] + b[i]
    end
end

function opencl_vadd()
    a = rand(Float32, 10^6)
    b = rand(Float32, 10^6)

    # CPU version
    c_cpu = similar(a)
    @btime cpu_vadd!($a, $b, $c_cpu)

    # OpenCL version
    d_a = CLArray(a)
    d_b = CLArray(b)
    d_c = similar(d_a)

    @btime begin
        @opencl global_size=size($d_a) vadd($d_a, $d_b, $d_c)
        Array($d_c)
    end
end
```

### Matrix Multiplication

Examples of matrix multiplication with different optimizations:

```julia
# Basic matrix multiplication
function matmul_basic(A, B, C)
    i = get_global_id(1)
    j = get_global_id(2)

    if i <= size(C, 1) && j <= size(C, 2)
        sum = 0.0f0
        for k in 1:size(A, 2)
            @inbounds sum += A[i, k] * B[k, j]
        end
        @inbounds C[i, j] = sum
    end
    return
end

# Usage
A = CLArray(rand(Float32, 256, 128))
B = CLArray(rand(Float32, 128, 256))
C = CLArray(zeros(Float32, 256, 256))

@opencl global_size=size(C) matmul_basic(A, B, C)
```

## Jupyter Notebooks

The repository includes several Jupyter notebooks with interactive examples:

### Julia Set Fractals

Create beautiful fractal images using OpenCL:

```julia
using OpenCL, Colors, Plots

function julia_set_kernel(output, c_real, c_imag, width, height, max_iter)
    i = get_global_id(1)
    j = get_global_id(2)

    if i <= height && j <= width
        # Map pixel to complex plane
        x = (j - width/2) * 4.0f0 / width
        y = (i - height/2) * 4.0f0 / height

        z_real = x
        z_imag = y

        iter = 0
        while iter < max_iter && (z_real^2 + z_imag^2) < 4.0f0
            temp = z_real^2 - z_imag^2 + c_real
            z_imag = 2.0f0 * z_real * z_imag + c_imag
            z_real = temp
            iter += 1
        end

        @inbounds output[i, j] = Float32(iter) / Float32(max_iter)
    end
    return
end

# Generate Julia set
width, height = 800, 600
output = CLArray(zeros(Float32, height, width))

@opencl global_size=(height, width) julia_set_kernel(
    output, -0.7f0, 0.27015f0, width, height, 100)

# Convert to image
result = Array(output)
heatmap(result, color=:hot, aspect_ratio=:equal)
```

### Mandelbrot Set

Similar to Julia set but with varying c parameter:

```julia
function mandelbrot_kernel(output, width, height, max_iter)
    i = get_global_id(1)
    j = get_global_id(2)

    if i <= height && j <= width
        # Map pixel to complex plane
        c_real = (j - width/2) * 4.0f0 / width
        c_imag = (i - height/2) * 4.0f0 / height

        z_real = 0.0f0
        z_imag = 0.0f0

        iter = 0
        while iter < max_iter && (z_real^2 + z_imag^2) < 4.0f0
            temp = z_real^2 - z_imag^2 + c_real
            z_imag = 2.0f0 * z_real * z_imag + c_imag
            z_real = temp
            iter += 1
        end

        @inbounds output[i, j] = Float32(iter) / Float32(max_iter)
    end
    return
end
```

## Hands-on OpenCL Examples

The `examples/hands_on_opencl` directory contains Julia translations of the "Hands On OpenCL" exercises:

### Exercise 4: Vector Addition Chain

Demonstrates chaining multiple kernel calls:

```julia
# Chain of vector additions: D = A + B + C
function vadd_chain()
    n = 1024
    A = CLArray(rand(Float32, n))
    B = CLArray(rand(Float32, n))
    C = CLArray(rand(Float32, n))
    D = CLArray(zeros(Float32, n))
    temp = CLArray(zeros(Float32, n))

    # temp = A + B
    @opencl global_size=(n,) vadd(A, B, temp)

    # D = temp + C
    @opencl global_size=(n,) vadd(temp, C, D)

    return Array(D)
end
```

### Exercise 5: Vector A·B + C

More complex arithmetic operations:

```julia
function vabc_kernel(a, b, c, result)
    gid = get_global_id(1)
    if gid <= length(result)
        @inbounds result[gid] = a[gid] * b[gid] + c[gid]
    end
    return
end

# Usage
a = CLArray(rand(Float32, 1024))
b = CLArray(rand(Float32, 1024))
c = CLArray(rand(Float32, 1024))
result = similar(a)

@opencl global_size=size(a) vabc_kernel(a, b, c, result)
```

### Exercise 9: Pi Calculation

Monte Carlo estimation of π:

```julia
function pi_kernel(results, num_points)
    gid = get_global_id(1)

    if gid <= length(results)
        count = 0
        seed = gid

        for i in 1:num_points
            # Simple random number generator
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            x = Float32(seed) / Float32(0x7fffffff)

            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            y = Float32(seed) / Float32(0x7fffffff)

            if x^2 + y^2 <= 1.0f0
                count += 1
            end
        end

        @inbounds results[gid] = count
    end
    return
end

# Estimate π
num_work_items = 1024
points_per_item = 10000
results = CLArray(zeros(Int32, num_work_items))

@opencl global_size=(num_work_items,) pi_kernel(results, points_per_item)

total_inside = sum(Array(results))
total_points = num_work_items * points_per_item
pi_estimate = 4.0 * total_inside / total_points

println("π estimate: ", pi_estimate)
println("Error: ", abs(pi_estimate - π))
```

## Array Transpose Example

Efficient matrix transpose with local memory:

```julia
function transpose_kernel(input, output, local_size)
    # Local memory tile
    tile = @localmem Float32 (local_size, local_size)

    # Global indices
    global_i = get_global_id(1)
    global_j = get_global_id(2)

    # Local indices
    local_i = get_local_id(1)
    local_j = get_local_id(2)

    # Read into local memory
    if global_i <= size(input, 1) && global_j <= size(input, 2)
        @inbounds tile[local_i, local_j] = input[global_i, global_j]
    end

    barrier()

    # Write transposed data
    output_i = get_group_id(2) * local_size + local_i
    output_j = get_group_id(1) * local_size + local_j

    if output_i <= size(output, 1) && output_j <= size(output, 2)
        @inbounds output[output_i, output_j] = tile[local_j, local_i]
    end
    return
end

# Usage
input = CLArray(rand(Float32, 1024, 1024))
output = CLArray(zeros(Float32, 1024, 1024))
local_size = 16

@opencl global_size=size(input) local_size=(local_size, local_size) transpose_kernel(input, output, local_size)
```

These examples demonstrate the versatility and power of OpenCL.jl for various computational tasks. For more examples and detailed implementations, check out the [examples directory](https://github.com/JuliaGPU/OpenCL.jl/tree/master/examples) in the repository.
