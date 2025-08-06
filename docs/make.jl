using OpenCL, pocl_jll
using Documenter

DocMeta.setdocmeta!(OpenCL, :DocTestSetup, :(using OpenCL, pocl_jll); recursive = true)

makedocs(
    sitename = "OpenCL.jl",
    format = Documenter.HTML(
        canonical = "https://juliagpu.github.io/OpenCL.jl/stable/",
        edit_link = "master",
        assets = String[],
    ),
    modules = [OpenCL],
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Installation" => "installation.md",
            "Basic Usage" => "basic_usage.md",
            "Native Kernels" => "native_kernels.md",
            "Examples" => "examples.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :none,
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

deploydocs(
    repo = "github.com/JuliaGPU/OpenCL.jl.git",
    devbranch = "master",
)
