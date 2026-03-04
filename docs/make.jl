using Documenter: Documenter, DocMeta, deploydocs, makedocs
using ITensorFormatter: ITensorFormatter
using KroneckerArrays: KroneckerArrays

DocMeta.setdocmeta!(
    KroneckerArrays, :DocTestSetup, :(using KroneckerArrays); recursive = true
)

ITensorFormatter.make_index!(pkgdir(KroneckerArrays))

makedocs(;
    modules = [KroneckerArrays],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "KroneckerArrays.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/KroneckerArrays.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"]
    ),
    pages = ["Home" => "index.md", "Reference" => "reference.md"]
)

deploydocs(;
    repo = "github.com/ITensor/KroneckerArrays.jl", devbranch = "main", push_preview = true
)
