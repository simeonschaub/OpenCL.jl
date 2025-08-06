# API Reference

```@contents
Pages = ["api.md"]
Depth = 2
```

## Arrays

```@docs
CLArray
CLDeviceArray
CLPtr
```

## Kernel Execution

```@docs
@opencl
clfunction
```

## OpenCL Low-Level Interface

For low-level OpenCL operations, OpenCL.jl provides access to the `cl` module which wraps the OpenCL C API.

Common functions include:
- `cl.Program` - Compile OpenCL C programs
- `cl.Kernel` - Create kernels from programs
- `cl.Buffer` - Allocate device memory
- `cl.CmdQueue` - Command queue operations
- `cl.Context` - OpenCL contexts
- `cl.Device` - Device queries and selection

Refer to the OpenCL C specification for detailed documentation of these functions.

## Julia GPU Array Interface

OpenCL.jl implements the GPUArrays.jl interface, providing familiar array operations that work on the GPU:

- Broadcasting: `a .+ b`
- Reductions: `sum(a)`, `maximum(a)`
- Linear algebra: Basic BLAS operations
- Array indexing and manipulation

## Device-Side Functions

The following functions are available within OpenCL kernels written in Julia:

### Work Item Functions

- `get_global_id(dim)` - Get global work item ID
- `get_local_id(dim)` - Get local work item ID
- `get_global_size(dim)` - Get global work size
- `get_local_size(dim)` - Get local work group size
- `get_num_groups(dim)` - Get number of work groups
- `get_group_id(dim)` - Get work group ID
- `get_work_dim()` - Get work dimensions
- `barrier()` - Synchronize work items in a work group

### Memory Functions

- `@localmem T (size...)` - Allocate local/shared memory

### Mathematical Functions

Most Julia mathematical functions are supported in OpenCL kernels, including:

- Basic arithmetic: `+`, `-`, `*`, `/`, `%`, `^`
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Exponential: `exp`, `exp2`, `exp10`, `expm1`
- Logarithmic: `log`, `log2`, `log10`, `log1p`
- Power: `sqrt`, `cbrt`, `pow`
- Rounding: `floor`, `ceil`, `round`, `trunc`
- Comparison: `min`, `max`, `abs`, `sign`
- Special: `fma`, `copysign`, `ldexp`, `frexp`

### Atomic Functions

For thread-safe operations within kernels:

- `atomic_add!(ptr, val)` - Atomic addition
- `atomic_sub!(ptr, val)` - Atomic subtraction
- `atomic_xchg!(ptr, val)` - Atomic exchange
- `atomic_cmpxchg!(ptr, cmp, val)` - Atomic compare-and-swap
- `atomic_inc!(ptr)` - Atomic increment
- `atomic_dec!(ptr)` - Atomic decrement
- `atomic_and!(ptr, val)` - Atomic bitwise AND
- `atomic_or!(ptr, val)` - Atomic bitwise OR
- `atomic_xor!(ptr, val)` - Atomic bitwise XOR
- `atomic_min!(ptr, val)` - Atomic minimum
- `atomic_max!(ptr, val)` - Atomic maximum
