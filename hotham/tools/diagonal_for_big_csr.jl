using NPZ
using SparseArrays
using LinearAlgebra
using Pardiso
using LinearMaps
using JSON
using Printf
using Arpack
using Dates
using Glob

function construct_linear_map(H, S)
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.COMPLEX_HERM_INDEF)
    pardisoinit(ps)
    fix_iparm!(ps, :N)
    # 显式设置Pardiso使用的线程数
    num_threads = haskey(ENV, "MKL_NUM_THREADS") ? parse(Int, ENV["MKL_NUM_THREADS"]) : 1
    set_nprocs!(ps, num_threads)
    H_pardiso = get_matrix(ps, H, :N)
    b = rand(ComplexF64, size(H, 1))
    set_phase!(ps, Pardiso.ANALYSIS)
    pardiso(ps, H_pardiso, b)
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, H_pardiso, b)
    return (
        LinearMap{ComplexF64}(
            (y, x) -> begin
                set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
                pardiso(ps, y, H_pardiso, S * x)
            end,
            size(H, 1);
            ismutating=true
        ),
        ps
    )
end

function load_sparse_matrix(filepath::String)
    f = npzread(filepath)
    data = f["data"]
    indices = f["indices"] .+ 1
    indptr = f["indptr"] .+ 1
    shape = f["shape"]
    return SparseMatrixCSC(shape[1], shape[2], indptr, indices, data)
end

function solve_eigen_problem(calc_center::Float64, num_band::Int)
    h_files = Glob.glob("*_H.npz", pwd())
    h_file = h_files[1]
    dir_path = dirname(h_file)
    base_name = basename(h_file)
    println("Found H file:", basename(h_file))

    s_file = joinpath(dir_path, replace(base_name, "_H.npz" => "_S.npz"))
    if !isfile(s_file)
        error("Can't find Overlap_Matrix: $(s_file)")
    end
    println("Found S file: $(basename(s_file))")

    href_file = joinpath(dir_path, replace(base_name, "_H.npz" => "_Href.npz"))
    if isfile(href_file)
        println("Found H_ref file: $(basename(href_file))")
    else
        println("No H_ref found")
    end
    flush(stdout)

    H = load_sparse_matrix(h_file)
    S = load_sparse_matrix(s_file)
    
    println("[START]Linear Map [" * Dates.format(Dates.now() + Hour(8), "mm-dd HH:MM:SS") * "] ")
    flush(stdout)
    time_lm_start = time()
    lm, ps = construct_linear_map(Hermitian(H) - (calc_center) * Hermitian(S), Hermitian(S))
    time_lm = time() - time_lm_start
    println("[DONE]Linear Map [" * Dates.format(Dates.now() + Hour(8), "mm-dd HH:MM:SS") * "]\n")
    @printf("[TIME]Linear Map: %.3f s\n", time_lm)
    println("[START]Diagonalization [" * Dates.format(Dates.now() + Hour(8), "mm-dd HH:MM:SS") * "] ")
    flush(stdout)
    time_eigs_start = time()
    vals, ncon, niter = eigs(lm, nev=num_band, which=:LM, ritzvec=false, maxiter=100000, tol=1e-12)
    time_eigs = time() - time_eigs_start
    println("[DONE]Diagonalization [" * Dates.format(Dates.now() + Hour(8), "mm-dd HH:MM:SS") * "]\n")
    @printf("[TIME] Diagonalization: %.3f s\n", time_eigs)
    println("==============================")
    println("[OUT]Converged Eigs: $(ncon)")
    println("[OUT]Iteration Times: $(niter)")
    flush(stdout)
    set_phase!(ps, Pardiso.RELEASE_ALL)
    pardiso(ps)
    vals = real(1 ./ vals) .+ (calc_center)
    open("./eig_Pred.txt", "a") do file
        for val in vals
            @printf(file, "%.14e ", val)
        end
        @printf(file, "\n")
        flush(file)
    end

    if isfile(href_file)
        H = load_sparse_matrix(href_file)
        S = load_sparse_matrix(s_file)
        println("Building Linear map... [" * Dates.format(Dates.now() + Hour(8), "mm-dd HH:MM:SS") * "] ")
        flush(stdout)
        time_lm_start = time()
        lm, ps = construct_linear_map(Hermitian(H) - (calc_center) * Hermitian(S), Hermitian(S))
        time_lm = time() - time_lm_start
        @printf("Built Linear map: %.3f s[%s]\n", time_lm, Dates.format(Dates.now() + Hour(8), "mm-dd HH:MM:SS"))
        println("Diagonalizing... [" * Dates.format(Dates.now() + Hour(8), "mm-dd HH:MM:SS") * "] ")
        flush(stdout)
        time_eigs_start = time()
        vals, ncon, niter = eigs(lm, nev=num_band, which=:LM, ritzvec=false, maxiter=100000, tol=1e-12)
        time_eigs = time() - time_eigs_start
        @printf("Diagonalization time: %.3f s [%s]\n", time_eigs, Dates.format(Dates.now() + Hour(8), "mm-dd HH:MM:SS"))
        println("Converged eigs: $(ncon), after iteration: $(niter)")
        flush(stdout)
        set_phase!(ps, Pardiso.RELEASE_ALL)
        pardiso(ps)
        vals = real(1 ./ vals) .+ (calc_center)
        open("./eig_Ref.txt", "a") do file
            for val in vals
                @printf(file, "%.14e ", val)
            end
            @printf(file, "\n")
            flush(file)
        end
    end
end

function main()
    if length(ARGS) < 2
        error("Usage: julia diagonal.jl calc_center num_band")
    end
    calc_center = parse(Float64, ARGS[1])
    num_band = parse(Int, ARGS[2])
    solve_eigen_problem(calc_center, num_band)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end