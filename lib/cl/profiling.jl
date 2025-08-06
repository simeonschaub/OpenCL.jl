function khrIcdLayerAdd(library_name::String)
    @ccall libopencl.khrIcdLayerAdd(library_name::Cstring)::Ptr{Cvoid}
end

using opencl_kernel_profiler_jll

const lib = Ref{Ptr{Cvoid}}(C_NULL)

macro profile(ex)
    quote
        q = queue()
        dest = joinpath(mktempdir(; cleanup = false), "opencl_kernel_profile.trace")
        res = withenv("CLKP_TRACE_DEST" => dest) do
            if lib[] == C_NULL
                lib[] = cl.khrIcdLayerAdd(opencl_kernel_profiler_jll.libopencl_kernel_profiler_path)
            else
                ccall(Libc.Libdl.dlsym(lib[], :startTracing), Cvoid, ())
            end
            q′ = queue!()
            res = $(esc(ex))
            finish(q′)
            ccall(Libc.Libdl.dlsym(lib[], :stopTracing), Cvoid, ())
            res
        end
        @show dest
        queue!(q)
        res
    end
end
