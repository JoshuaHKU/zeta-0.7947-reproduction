// -*- coding: utf-8 -*-
// =====================================================================
// mladder_fast.cu — 快速 GPU/CPU 中点阶梯引擎 (论文 §5.5 连通常数)
// Fast CUDA/OpenMP engine for the grouped midpoint ladder
//     ∫O_bC_b = Σ_{(P,σ)} sign · [ov·ov integral]
//
// 与 gpu/midpoint_ladder_gpu.py 及 certification/midpoint_ladder.py
// **数学等价**, 业务逻辑不变; 仅重写执行方式:
// Mathematically IDENTICAL to the Python engines; only the execution
// is rewritten.  Three exact (result-preserving) accelerations:
//
//  (1) 支撑集裁剪 / support pruning.  ovV = clamp(1-(max-min),0) over
//      the v-prefix walk {0,W_1..W_{b-1}} vanishes unless the walk
//      spread is < 1.  Points with ovV = 0 contribute exactly 0.0 to
//      the sum, so skipping them changes nothing.  The surviving set
//      has volume exactly b (vs 4^{b-1} for the full box) and is
//      prefix-decomposable: given a partial walk with running
//      (mx,mn), the next coordinate must lie in the CONTIGUOUS
//      interval [mx-1-W, mn+1-W].  => exact enumeration of the
//      nonzero points, 51x (b=5) / 171x (b=6) / 585x (b=7) fewer.
//
//  (2) 项树 / term trie.  The (partition, cyclic-order) terms are in
//      bijection with chains  0∈S_1 ⊊ … ⊊ S_{m-1} ⊊ [b]  with sign
//      (-1)^{m-1}.  Chains sharing a prefix share their running
//      max/min, so the 150/1082/9366 terms are evaluated by one DFS
//      over a trie whose nodes are exactly the terms.  A node whose
//      overlap already vanishes has vanishing overlap in its whole
//      subtree (spread is monotone), so the subtree is skipped —
//      again dropping only exact zeros.
//
//  (3) 融合内核 / fusion.  The Python version makes ~1082 separate
//      elementwise passes over a multi-GB chunk per grid block; here
//      one thread evaluates a whole point with everything (walk,
//      positions, trie stacks) in registers.  Templated on (b, L) so
//      every array index is a compile-time constant.
//
// fp64 throughout.  v1-slice checkpointing preserved, so runs split
// across devices/nodes by [i0,i1) and the partial sums add.
//
// build:  nvcc -O3 -std=c++14 -Xcompiler "-O3 -fopenmp" \
//              -gencode arch=compute_70,code=sm_70 \
//              -gencode arch=compute_86,code=sm_86 \
//              -o mladder_fast mladder_fast.cu
// usage:  ./mladder_fast B DV [--i0 I] [--i1 I] [--dev N|--cpu]
//                            [--sym] [--tag NAME] [--quiet]
//         ./mladder_fast --trie B          (term/trie census)
// =====================================================================
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------
// 项树构造 (host) / term-trie construction
// node word: [subtree_size:21][delta_mask:7][depth:4]
// ---------------------------------------------------------------------
static int trie_dfs(int b, unsigned U, unsigned delta, int depth,
                    std::vector<uint32_t>& nodes)
{
    const unsigned full = (1u << b) - 1u;
    const size_t me = nodes.size();
    nodes.push_back(0u);
    const unsigned comp = full & ~U;
    for (unsigned s = comp; s; s = (s - 1u) & comp) {   // nonempty subsets
        if (depth == 0 && !(s & 1u)) continue;  // 首块必含 0 / first block ∋ 0
        unsigned U2 = U | s;
        if (U2 == full) continue;               // chain stays proper
        trie_dfs(b, U2, s, depth + 1, nodes);
    }
    const int sz = (int)(nodes.size() - me);
    nodes[me] = ((uint32_t)sz << 11) | ((delta & 127u) << 4) | (uint32_t)depth;
    return sz;
}

static std::vector<uint32_t> build_trie(int b)
{
    std::vector<uint32_t> nodes;
    trie_dfs(b, 0u, 0u, 0, nodes);
    return nodes;
}

// ---------------------------------------------------------------------
// 项树遍历 / trie walk — template recursion keeps the (pos,max,min)
// stack in registers.  sub = Σ_nodes (-1)^depth · clamp(1-spread,0)
// ---------------------------------------------------------------------
template<int B, int D, bool DONE> struct WalkImpl;

template<int B, int D> struct WalkImpl<B, D, true> {
    HD static void go(int&, const uint32_t*, int, const double*,
                      double, double, double, double&) {}
};

template<int B, int D> struct WalkImpl<B, D, false> {
    HD static void go(int& i, const uint32_t* nodes, int nn, const double* v,
                      double pp, double pmx, double pmn, double& sub)
    {
        while (i < nn) {
            const uint32_t nd = nodes[i];
            const int d = (int)(nd & 15u);
            if (d != D) return;                       // unwind to ancestor
            const unsigned msk = (nd >> 4) & 127u;
            double p = pp;
            for (int j = 0; j < B; ++j) if ((msk >> j) & 1u) p += v[j];
            const double mx = fmax(pmx, p), mn = fmin(pmn, p);
            const double w = 1.0 - (mx - mn);
            if (w <= 0.0) {                           // exact zero subtree
                i += (int)(nd >> 11);
            } else {
                sub += (D & 1) ? -w : w;
                ++i;
                WalkImpl<B, D + 1, (D + 1 >= B)>::go(i, nodes, nn, v,
                                                     p, mx, mn, sub);
            }
        }
    }
};

template<int B>
HD inline double trie_sub(const double* v, const uint32_t* nodes, int nn)
{
    double sub = 1.0;            // root: empty chain, ov = 1
    int i = 1;
    WalkImpl<B, 1, (1 >= B)>::go(i, nodes, nn, v, 0.0, 0.0, 0.0, sub);
    return sub;
}

// ---------------------------------------------------------------------
// 尾部格点枚举 / tail enumeration over the remaining coordinates
// ---------------------------------------------------------------------
struct Ctx {
    int n; double start, dv;
    const uint32_t* nodes; int nnodes;
};

HD inline void coord_range(const Ctx& c, double W, double mx, double mn,
                           int& lo, int& hi)
{
    // W' = W+v must satisfy max(mx,W')-min(mn,W') <= 1  ⟺  W' ∈ [mx-1, mn+1]
    // 外扩 1e-12 保证绝不丢掉任何 ovV>0 的点 / outward slack: never drop
    const double a = mx - 1.0 - W - 1e-12;
    const double bb = mn + 1.0 - W + 1e-12;
    lo = (int)ceil((a - c.start) / c.dv);
    hi = (int)floor((bb - c.start) / c.dv);
    if (lo < 0) lo = 0;
    if (hi > c.n - 1) hi = c.n - 1;
}

template<int B, int LEV, bool LAST> struct TailImpl;

template<int B, int LEV> struct TailImpl<B, LEV, true> {   // point complete
    HD static double go(const Ctx& c, double* v, double, double mx, double mn)
    {
        const double ovV = 1.0 - (mx - mn);
        if (ovV <= 0.0) return 0.0;
        double s = 0.0;
        for (int j = 0; j < B - 1; ++j) s += v[j];
        v[B - 1] = -s;                       // v_{b-1} = -(v_0+…+v_{b-2})
        return ovV * trie_sub<B>(v, c.nodes, c.nnodes);
    }
};

template<int B, int LEV> struct TailImpl<B, LEV, false> {
    HD static double go(const Ctx& c, double* v, double W, double mx, double mn)
    {
        int lo, hi;
        coord_range(c, W, mx, mn, lo, hi);
        double acc = 0.0;
        for (int i = lo; i <= hi; ++i) {
            const double x = c.start + i * c.dv;
            v[LEV] = x;
            const double W2 = W + x;
            acc += TailImpl<B, LEV + 1, (LEV + 1 >= B - 1)>::go(
                       c, v, W2, fmax(mx, W2), fmin(mn, W2));
        }
        return acc;
    }
};

// valid prefix levels are 1..B-2; clamp so the dispatch switch below does
// not instantiate impossible (B,L) pairs / 夹紧模板参数, 避免非法实例化
#define CLL(BB, LL) ((LL) <= (BB) - 2 ? (LL) : (BB) - 2)

template<int B, int L>
HD inline double eval_prefix(const Ctx& c, const double* pv,
                             double W, double mx, double mn)
{
    double v[B];
    for (int j = 0; j < L; ++j) v[j] = pv[j];
    return TailImpl<B, L, (L >= B - 1)>::go(c, v, W, mx, mn);
}

// ---------------------------------------------------------------------
// CUDA kernel — 一线程一前缀 / one thread per prefix
// prefix buffer is SoA: col j at pref[j*npref + t], cols = L(v) + W,mx,mn
// ---------------------------------------------------------------------
#ifdef __CUDACC__
template<int B, int L>
__global__ void k_run(Ctx c, const double* __restrict__ pref, long npref,
                      double* __restrict__ out)
{
    const long stride = (long)gridDim.x * blockDim.x;
    double acc = 0.0;
    for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
         t < npref; t += stride) {
        double pv[(L > 0 ? L : 1)];
        for (int j = 0; j < L; ++j) pv[j] = pref[(long)j * npref + t];
        acc += eval_prefix<B, L>(c, pv,
                                 pref[(long)L * npref + t],
                                 pref[(long)(L + 1) * npref + t],
                                 pref[(long)(L + 2) * npref + t]);
    }
    __shared__ double sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sh[0];
}
#endif

// ---------------------------------------------------------------------
// 前缀枚举 (host) / prefix enumeration — exact staircase, contiguous
// index range per level
// ---------------------------------------------------------------------
struct PrefBuf {
    int L = 0;
    std::vector<std::vector<double> > col;   // L+3 columns
    void init(int L_, size_t cap) {
        L = L_; col.assign(L_ + 3, std::vector<double>());
        for (auto& c : col) c.reserve(cap);
    }
    size_t size() const { return col[0].size(); }
    void clear() { for (auto& c : col) c.clear(); }
    inline void push(const double* v, double W, double mx, double mn) {
        for (int j = 0; j < L; ++j) col[j].push_back(v[j]);
        col[L].push_back(W); col[L + 1].push_back(mx); col[L + 2].push_back(mn);
    }
};

static void gen_prefix(const Ctx& c, int L, int lev, double* v,
                       double W, double mx, double mn, PrefBuf& out)
{
    if (lev == L) { out.push(v, W, mx, mn); return; }
    int lo, hi;
    coord_range(c, W, mx, mn, lo, hi);
    for (int i = lo; i <= hi; ++i) {
        const double x = c.start + i * c.dv;
        v[lev] = x;
        const double W2 = W + x;
        gen_prefix(c, L, lev + 1, v, W2, fmax(mx, W2), fmin(mn, W2), out);
    }
}

// ---------------------------------------------------------------------
// dispatch helpers
// ---------------------------------------------------------------------
static double kahan_sum(const std::vector<double>& x)
{
    double s = 0.0, comp = 0.0;
    for (double xi : x) { double y = xi - comp, t = s + y; comp = (t - s) - y; s = t; }
    return s;
}

#ifdef __CUDACC__
static double* d_pref = nullptr;
static double* d_out  = nullptr;
static uint32_t* d_nodes = nullptr;
static long d_pref_cap = 0;
static int  d_out_cap = 0;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s @%d: %s\n", #x, __LINE__, \
            cudaGetErrorString(e)); exit(1);} } while (0)

static double gpu_chunk(int b, int L, const Ctx& hc, PrefBuf& buf,
                        long off, long npref)
{
    if (npref == 0) return 0.0;
    const int ncol = L + 3;
    if (npref * ncol > d_pref_cap) {
        if (d_pref) CK(cudaFree(d_pref));
        d_pref_cap = npref * ncol;
        CK(cudaMalloc(&d_pref, d_pref_cap * sizeof(double)));
    }
    for (int j = 0; j < ncol; ++j)
        CK(cudaMemcpy(d_pref + (long)j * npref, buf.col[j].data() + off,
                      npref * sizeof(double), cudaMemcpyHostToDevice));
    const int TPB = 128;
    int blocks = (int)std::min<long>(65535, (npref + TPB - 1) / TPB);
    if (blocks < 1) blocks = 1;
    if (blocks > d_out_cap) {
        if (d_out) CK(cudaFree(d_out));
        d_out_cap = blocks;
        CK(cudaMalloc(&d_out, d_out_cap * sizeof(double)));
    }
    Ctx c = hc; c.nodes = d_nodes;
    bool ok = true;
#define LAUNCH(BB, LL) k_run<BB, CLL(BB, LL)><<<blocks, TPB>>>(c, d_pref, npref, d_out)
#define DISP_L(BB) switch (L) { \
        case 1: LAUNCH(BB,1); break; case 2: LAUNCH(BB,2); break; \
        case 3: LAUNCH(BB,3); break; case 4: LAUNCH(BB,4); break; \
        case 5: LAUNCH(BB,5); break; default: ok = false; }
    switch (b) {
        case 3: DISP_L(3) break;
        case 4: DISP_L(4) break;
        case 5: DISP_L(5) break;
        case 6: DISP_L(6) break;
        case 7: DISP_L(7) break;
        default: ok = false;
    }
#undef DISP_L
#undef LAUNCH
    if (!ok) { fprintf(stderr, "unsupported (b=%d,L=%d)\n", b, L); exit(1); }
    CK(cudaGetLastError());
    std::vector<double> h(blocks);
    CK(cudaMemcpy(h.data(), d_out, blocks * sizeof(double),
                  cudaMemcpyDeviceToHost));
    return kahan_sum(h);
}

// 分块提交, 限制单次 kernel 时长 / chunked submission bounds kernel time
static double gpu_batch(int b, int L, const Ctx& c, PrefBuf& buf, long chunk)
{
    const long tot = (long)buf.size();
    double s = 0.0, comp = 0.0;
    for (long off = 0; off < tot; off += chunk) {
        double xi = gpu_chunk(b, L, c, buf, off, std::min(chunk, tot - off));
        double y = xi - comp, t = s + y; comp = (t - s) - y; s = t;
    }
    buf.clear();
    return s;
}
#endif

static double cpu_batch(int b, int L, const Ctx& c, PrefBuf& buf)
{
    const long npref = (long)buf.size();
    if (npref == 0) return 0.0;
    const double* col[8];
    for (int j = 0; j < L + 3; ++j) col[j] = buf.col[j].data();
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    std::vector<double> part(nth, 0.0);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double acc = 0.0;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4096) nowait
#endif
        for (long t = 0; t < npref; ++t) {
            double pv[8];
            for (int j = 0; j < L; ++j) pv[j] = col[j][t];
            const double W = col[L][t], mx = col[L + 1][t], mn = col[L + 2][t];
            double r = 0.0;
#define CALL(BB, LL) r = eval_prefix<BB, CLL(BB, LL)>(c, pv, W, mx, mn)
#define DISP_L(BB) switch (L) { \
            case 1: CALL(BB,1); break; case 2: CALL(BB,2); break; \
            case 3: CALL(BB,3); break; case 4: CALL(BB,4); break; \
            case 5: CALL(BB,5); break; default: r = 0.0; }
            switch (b) {
                case 3: DISP_L(3) break;
                case 4: DISP_L(4) break;
                case 5: DISP_L(5) break;
                case 6: DISP_L(6) break;
                case 7: DISP_L(7) break;
                default: r = 0.0;
            }
#undef DISP_L
#undef CALL
            acc += r;
        }
        part[tid] = acc;
    }
    buf.clear();
    return kahan_sum(part);
}

// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc >= 3 && !strcmp(argv[1], "--trie")) {
        int b = atoi(argv[2]);
        std::vector<uint32_t> nd = build_trie(b);
        std::vector<int> hist(16, 0);
        for (uint32_t w : nd) hist[w & 15u]++;
        printf("b=%d: trie nodes (= terms) = %zu   depth hist:", b, nd.size());
        for (int d = 0; d < 16; ++d) if (hist[d]) printf(" %d:%d", d, hist[d]);
        printf("\n");
        return 0;
    }
    if (argc < 3) {
        fprintf(stderr, "usage: %s B DV [--i0 I][--i1 I][--dev N|--cpu]"
                        "[--sym][--L n][--tag T][--quiet]\n", argv[0]);
        return 1;
    }
    const int b = atoi(argv[1]);
    const double dv = atof(argv[2]);
    int i0 = 0, i1 = -1, dev = 0, Lforce = 0, sym = 0, use_cpu = 0, quiet = 0;
    long flush = 6000000;                 // prefixes per kernel launch
    std::string tag;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--flush")) flush = atol(argv[++i]);
        else if (!strcmp(argv[i], "--i0")) i0 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--i1")) i1 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dev")) dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--L")) Lforce = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tag")) tag = argv[++i];
        else if (!strcmp(argv[i], "--cpu")) use_cpu = 1;
        else if (!strcmp(argv[i], "--sym")) sym = 1;
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
    }
    if (b < 3 || b > 7) { fprintf(stderr, "b must be 3..7\n"); return 1; }

    // 中点网格 (与 numpy arange(-2+dv/2, 2, dv) 逐位一致)
    const double start = -2.0 + dv / 2.0;
    const int n = (int)ceil((2.0 - start) / dv);
    if (i1 < 0 || i1 > n) i1 = n;

    std::vector<uint32_t> nodes = build_trie(b);
    Ctx c; c.n = n; c.start = start; c.dv = dv;
    c.nodes = nodes.data(); c.nnodes = (int)nodes.size();

    // 前缀层级 L 的选择 / choose prefix level L:
    // count(L) ≈ (L+1)/dv^L ; want enough parallelism, bounded host work
    int L = 1;
    if (Lforce > 0) L = Lforce;
    else {
        const double TARGET = 4e6, CAP = 2.5e8;
        double cnt = 1.0;
        for (int k = 1; k <= b - 2; ++k) {
            double ck = (k + 1) / pow(dv, (double)k);
            if (ck > CAP && k > 1) break;
            L = k; cnt = ck;
            if (cnt >= TARGET) break;
        }
    }
    if (L > b - 2) L = b - 2;
    if (L < 1) L = 1;

    const char* devname = "cpu";
#ifdef __CUDACC__
    char gname[128] = "cpu";
    if (!use_cpu) {
        CK(cudaSetDevice(dev));
        cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
        snprintf(gname, sizeof(gname), "%s", pr.name);
        devname = gname;
        CK(cudaMalloc(&d_nodes, nodes.size() * sizeof(uint32_t)));
        CK(cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(uint32_t),
                      cudaMemcpyHostToDevice));
    }
#endif
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    if (!quiet)
        printf("[fast-ladder] b=%d dv=%g grid=%d^%d terms=%zu L=%d "
               "slices [%d,%d)%s dev=%s%s\n",
               b, dv, n, b - 1, nodes.size(), L, i0, i1,
               sym ? " [sym x2]" : "", devname,
               use_cpu ? (" threads=" + std::to_string(nth)).c_str() : "");
    fflush(stdout);

    // 对称性 v -> -v: 被积函数不变, 中点网格在 4/dv ∈ Z 时精确对称
    // symmetry: integrand invariant under global negation; midpoint grid
    // is exactly symmetric when 4/dv is an integer -> half the slices x2
    double symfac = 1.0;
    if (sym) {
        double q = 4.0 / dv;
        if (fabs(q - floor(q + 0.5)) > 1e-9) {
            fprintf(stderr, "--sym needs 4/dv integral\n"); return 1;
        }
        int half = n / 2;
        if (i1 > half) i1 = half;
        symfac = 2.0;
    }

    const size_t FLUSH = (size_t)flush;
    PrefBuf buf; buf.init(L, FLUSH + 4000000);
    double acc = 0.0, ccomp = 0.0;          // Kahan over slices
    const double wgt = pow(dv, (double)(b - 1)) * symfac;
    auto t0 = std::chrono::steady_clock::now();
    double batch = 0.0;
    long done_pts = 0;

    for (int i = i0; i < i1; ++i) {
        const double v0 = start + i * dv;
        if (fabs(v0) < 1.0 + 1e-12) {       // |v0|>=1 => walk spread>=1 => 0
            double v[8]; v[0] = v0;
            gen_prefix(c, L, 1, v, v0, fmax(0.0, v0), fmin(0.0, v0), buf);
        }
        const bool last = (i == i1 - 1);
        if (buf.size() >= FLUSH || (last && buf.size())) {
            done_pts += (long)buf.size();
#ifdef __CUDACC__
            batch += use_cpu ? cpu_batch(b, L, c, buf)
                             : gpu_batch(b, L, c, buf, (long)FLUSH);
#else
            batch += cpu_batch(b, L, c, buf);
#endif
        }
        if (buf.size() == 0) {              // slice boundary + flushed
            double y = batch * wgt - ccomp, t = acc + y;
            ccomp = (t - acc) - y; acc = t; batch = 0.0;
            if (!quiet && (((i - i0) & 31) == 0 || last)) {
                double el = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();
                double frac = (double)(i - i0 + 1) / (i1 - i0);
                printf("  slice %d/%d  partial %+.10f  (%.0fs, ETA %.1f min)\n",
                       i + 1, n, acc, el, el * (1 / frac - 1) / 60.0);
                fflush(stdout);
            }
        }
    }
    double el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
    printf("[RESULT] b=%d dv=%g slices=[%d,%d) integral=%+.12f "
           "prefixes=%ld wall=%.1fs dev=%s\n",
           b, dv, i0, i1, acc, done_pts, el, devname);
    if (!tag.empty()) {
        FILE* f = fopen(tag.c_str(), "a");
        if (f) {
            fprintf(f, "%d %.10g %d %d %.15e %.1f %s\n",
                    b, dv, i0, i1, acc, el, devname);
            fclose(f);
        }
    }
    return 0;
}
