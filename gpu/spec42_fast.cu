// -*- coding: utf-8 -*-
// =====================================================================
// spec42_fast.cu — {4,2} 旁观对变体的 CUDA/OpenMP 引擎
// CUDA/OpenMP engine for the {4,2} spectator-pair variant (paper §5.5)
//
// 规格来自 gpu/spectator_42_reference.py，逐条对应：
// Spec is gpu/spectator_42_reference.py, followed point by point:
//
//   自由变量 4D: v, c1, c2, c3；c4 = -(c1+c2+c3)
//   中点网格   : g = arange(-2+dv/2, 2, dv)
//   被积函数   : ovV(walk_d) · C4(c1,c2,c3,c4) · C2(v),  C2 = min(|v|,1)
//   C4         : 26 项分拆-循环号叠和 = 纯圈引擎 compile_terms(4)
//   三条走步   : d=1 p=[0,v,0,     s1,s2,s3]
//                d=2 p=[0,v,v+s1, s1,s2,s3]
//                d=3 p=[0,v,v+s1, v+s2,s2,s3]      (s_k = c1+..+ck)
//   U_d = Σ_grid ovV_d·C4·C2 · dv⁴
//   {4,2} = 6·U₁ + 6·U₂ + 3·U₃
//
// 与参考实现**数学等价**，只改执行方式（业务逻辑不变）：
//
//  (1) 支撑集裁剪 / support pruning。ovV_d 在走步跨度 ≥ 1 时恒为 0，
//      该格点贡献恰好 0.0。逐层看，第 LEV 层为每个 d 新加入的走步点
//      是 s' 和/或 v+s'，二者都必须落在 [mx_d−1, mn_d+1] 内 ——
//      每个 d 得到一个**连续区间**，三个 d 取**包络 (hull)**。
//      包络是超集 ⟹ 一个 ovV>0 的点都不会丢；落在包络内但三个 d 全零
//      的点照常算出 0。于是三个 U_d 在同一遍枚举里出，C4 只算一次。
//
//  (2) 项树 + 子树裁剪 / term trie。26 个 (分拆,循环序) 项与链
//      0∈S₁⊊…⊊S_{m−1}⊊[4] 一一对应，符号 (−1)^{m−1}；共享前缀的链
//      共享 max/min，一次 DFS 走完。跨度单调 ⟹ 某节点重叠已为零时
//      整个子树也为零，可整体跳过（同样只丢弃精确的零）。
//
//  (3) 融合内核 / fusion。一个线程一个格点，走步、位置、项树栈
//      全部驻留寄存器。
//
//  (4) 全局反号对称 / global negation symmetry。规格已声明：
//      (v,c1,c2,c3) → −(v,c1,c2,c3) 下走步反射、C2/C4 为偶 ⟹ 被积函数
//      不变。4/dv ∈ Z 时中点网格精确自对称，故只算一半 v-分片再乘 2。
//
// build: nvcc -O3 -std=c++14 -Xcompiler "-O3 -fopenmp -march=native" \
//             -gencode arch=compute_70,code=sm_70 \
//             -gencode arch=compute_86,code=sm_86 \
//             -o spec42_fast spec42_fast.cu
// usage: ./spec42_fast DV [--dev N|--cpu] [--i0 I --i1 I] [--sym]
//                         [--tag FILE] [--flush N] [--quiet]
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

static const double BIG = 1e300;

// ---------------------------------------------------------------------
// 项树 (b=4) / term trie — node word: [subtree:21][delta_mask:7][depth:4]
// ---------------------------------------------------------------------
static int trie_dfs(int b, unsigned U, unsigned delta, int depth,
                    std::vector<uint32_t>& nodes)
{
    const unsigned full = (1u << b) - 1u;
    const size_t me = nodes.size();
    nodes.push_back(0u);
    const unsigned comp = full & ~U;
    for (unsigned s = comp; s; s = (s - 1u) & comp) {
        if (depth == 0 && !(s & 1u)) continue;   // 首块必含 0
        unsigned U2 = U | s;
        if (U2 == full) continue;                // 链保持真子集
        trie_dfs(b, U2, s, depth + 1, nodes);
    }
    const int sz = (int)(nodes.size() - me);
    nodes[me] = ((uint32_t)sz << 11) | ((delta & 127u) << 4) | (uint32_t)depth;
    return sz;
}

template<int D, bool DONE> struct Walk4;
template<int D> struct Walk4<D, true> {
    HD static void go(int&, const uint32_t*, int, const double*,
                      double, double, double, double&) {}
};
template<int D> struct Walk4<D, false> {
    HD static void go(int& i, const uint32_t* nodes, int nn, const double* w,
                      double pp, double pmx, double pmn, double& sub)
    {
        while (i < nn) {
            const uint32_t nd = nodes[i];
            const int d = (int)(nd & 15u);
            if (d != D) return;
            const unsigned msk = (nd >> 4) & 127u;
            double p = pp;
            for (int j = 0; j < 4; ++j) if ((msk >> j) & 1u) p += w[j];
            const double mx = fmax(pmx, p), mn = fmin(pmn, p);
            const double t = 1.0 - (mx - mn);
            if (t <= 0.0) { i += (int)(nd >> 11); }   // 精确为零的子树
            else {
                sub += (D & 1) ? -t : t;
                ++i;
                Walk4<D + 1, (D + 1 >= 4)>::go(i, nodes, nn, w, p, mx, mn, sub);
            }
        }
    }
};

// C4(w) —— 与参考实现 C4_of() 同一个量
HD inline double C4_of(const double* w, const uint32_t* nodes, int nn)
{
    double sub = 1.0;                 // 空链 (m=1), ov = 1
    int i = 1;
    Walk4<1, false>::go(i, nodes, nn, w, 0.0, 0.0, 0.0, sub);
    return sub;
}

// ---------------------------------------------------------------------
// 第 LEV 层为第 d 条走步新加入的点: s' (PLAIN) 和/或 v+s' (SHIFT)
//   LEV=0 (选 c1, s'=s1): d1{s1}      d2{s1, v+s1}  d3{v+s1}
//   LEV=1 (选 c2, s'=s2): d1{s2}      d2{s2}        d3{s2, v+s2}
//   LEV=2 (选 c3, s'=s3): d1{s3}      d2{s3}        d3{s3}
// ---------------------------------------------------------------------
HD inline bool p42_plain(int lev, int d) { return !(lev == 0 && d == 2); }
HD inline bool p42_shift(int lev, int d)
{
    return (lev == 0 && (d == 1 || d == 2)) || (lev == 1 && d == 2);
}

struct Ctx42 {
    int n; double start, dv;
    const uint32_t* nodes; int nnodes;
};

// ---------------------------------------------------------------------
// 尾部枚举 / tail enumeration over the remaining c-coordinates
// ---------------------------------------------------------------------
template<int LEV, bool LAST> struct T42;

template<int LEV> struct T42<LEV, true> {          // LEV == 3 → 完整格点
    HD static void go(const Ctx42& c, double v, double* cc, double,
                      const double* mx, const double* mn, double* acc)
    {
        double o[3];
        bool any = false;
        for (int d = 0; d < 3; ++d) {
            const double t = 1.0 - (mx[d] - mn[d]);
            o[d] = t > 0.0 ? t : 0.0;
            if (t > 0.0) any = true;
        }
        if (!any) return;
        const double w[4] = { cc[0], cc[1], cc[2], -(cc[0] + cc[1] + cc[2]) };
        const double C4 = C4_of(w, c.nodes, c.nnodes);
        double av = fabs(v); if (av > 1.0) av = 1.0;      // C2(v) = min(|v|,1)
        const double f = C4 * av;
        for (int d = 0; d < 3; ++d) acc[d] += o[d] * f;
    }
};

template<int LEV> struct T42<LEV, false> {
    HD static void go(const Ctx42& c, double v, double* cc, double s,
                      const double* mx, const double* mn, double* acc)
    {
        // 每个 d 的允许区间 → 取包络 (超集, 不丢点)
        double lo = BIG, hi = -BIG;
        for (int d = 0; d < 3; ++d) {
            if (mx[d] - mn[d] >= 1.0) continue;          // 该 d 已死
            double a = -BIG, b = BIG;
            if (p42_plain(LEV, d)) { a = fmax(a, mx[d] - 1.0);
                                     b = fmin(b, mn[d] + 1.0); }
            if (p42_shift(LEV, d)) { a = fmax(a, mx[d] - 1.0 - v);
                                     b = fmin(b, mn[d] + 1.0 - v); }
            if (a <= b) { lo = fmin(lo, a); hi = fmax(hi, b); }
        }
        if (lo > hi) return;
        int i0 = (int)ceil((lo - s - 1e-12 - c.start) / c.dv);
        int i1 = (int)floor((hi - s + 1e-12 - c.start) / c.dv);
        if (i0 < 0) i0 = 0;
        if (i1 > c.n - 1) i1 = c.n - 1;
        for (int i = i0; i <= i1; ++i) {
            const double x = c.start + i * c.dv;
            cc[LEV] = x;
            const double s2 = s + x;
            double nmx[3], nmn[3];
            for (int d = 0; d < 3; ++d) {
                double a = mx[d], b = mn[d];
                if (p42_plain(LEV, d)) { a = fmax(a, s2);     b = fmin(b, s2); }
                if (p42_shift(LEV, d)) { a = fmax(a, v + s2); b = fmin(b, v + s2); }
                nmx[d] = a; nmn[d] = b;
            }
            T42<LEV + 1, (LEV + 1 >= 3)>::go(c, v, cc, s2, nmx, nmn, acc);
        }
    }
};

// 前缀 = (v, c1)；内核枚举 c2, c3
HD inline void eval42_prefix(const Ctx42& c, double v, double c1, double* acc)
{
    double mx[3], mn[3], cc[3];
    for (int d = 0; d < 3; ++d) { mx[d] = fmax(0.0, v); mn[d] = fmin(0.0, v); }
    cc[0] = c1;
    const double s1 = c1;
    for (int d = 0; d < 3; ++d) {
        if (p42_plain(0, d)) { mx[d] = fmax(mx[d], s1);     mn[d] = fmin(mn[d], s1); }
        if (p42_shift(0, d)) { mx[d] = fmax(mx[d], v + s1); mn[d] = fmin(mn[d], v + s1); }
    }
    T42<1, false>::go(c, v, cc, s1, mx, mn, acc);
}

// ---------------------------------------------------------------------
#ifdef __CUDACC__
__global__ void k42(Ctx42 c, const double* __restrict__ pv,
                    const double* __restrict__ pc1, long npref, double* out)
{
    const long stride = (long)gridDim.x * blockDim.x;
    double acc[3] = { 0.0, 0.0, 0.0 };
    for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
         t < npref; t += stride)
        eval42_prefix(c, pv[t], pc1[t], acc);
    __shared__ double sh[128 * 3];
    for (int d = 0; d < 3; ++d) sh[threadIdx.x * 3 + d] = acc[d];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            for (int d = 0; d < 3; ++d)
                sh[threadIdx.x * 3 + d] += sh[(threadIdx.x + s) * 3 + d];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int d = 0; d < 3; ++d) out[blockIdx.x * 3 + d] = sh[d];
}
#endif

// ---------------------------------------------------------------------
// 前缀枚举 (host)：对每个 v-分片，按 d=0 层的包络列出合法 c1
// ---------------------------------------------------------------------
static void gen42(const Ctx42& c, double v,
                  std::vector<double>& ov, std::vector<double>& oc1)
{
    double mx[3], mn[3];
    for (int d = 0; d < 3; ++d) { mx[d] = fmax(0.0, v); mn[d] = fmin(0.0, v); }
    double lo = BIG, hi = -BIG;
    for (int d = 0; d < 3; ++d) {
        if (mx[d] - mn[d] >= 1.0) continue;
        double a = -BIG, b = BIG;
        if (p42_plain(0, d)) { a = fmax(a, mx[d] - 1.0);     b = fmin(b, mn[d] + 1.0); }
        if (p42_shift(0, d)) { a = fmax(a, mx[d] - 1.0 - v); b = fmin(b, mn[d] + 1.0 - v); }
        if (a <= b) { lo = fmin(lo, a); hi = fmax(hi, b); }
    }
    if (lo > hi) return;
    int i0 = (int)ceil((lo - 1e-12 - c.start) / c.dv);
    int i1 = (int)floor((hi + 1e-12 - c.start) / c.dv);
    if (i0 < 0) i0 = 0;
    if (i1 > c.n - 1) i1 = c.n - 1;
    for (int i = i0; i <= i1; ++i) {
        ov.push_back(v);
        oc1.push_back(c.start + i * c.dv);
    }
}

static void kahan3(double* s, double* comp, const double* x)
{
    for (int d = 0; d < 3; ++d) {
        double y = x[d] - comp[d], t = s[d] + y;
        comp[d] = (t - s[d]) - y; s[d] = t;
    }
}

// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s DV [--dev N|--cpu][--i0 I][--i1 I]"
                        "[--sym][--tag F][--flush N][--quiet]\n", argv[0]);
        return 1;
    }
    const double dv = atof(argv[1]);
    int i0 = 0, i1 = -1, dev = 0, sym = 0, use_cpu = 0, quiet = 0;
    long flush = 4000000;
    std::string tag;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--i0")) i0 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--i1")) i1 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dev")) dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tag")) tag = argv[++i];
        else if (!strcmp(argv[i], "--flush")) flush = atol(argv[++i]);
        else if (!strcmp(argv[i], "--cpu")) use_cpu = 1;
        else if (!strcmp(argv[i], "--sym")) sym = 1;
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
    }
    const double start = -2.0 + dv / 2.0;
    const int n = (int)ceil((2.0 - start) / dv);
    if (i1 < 0 || i1 > n) i1 = n;

    std::vector<uint32_t> nodes;
    trie_dfs(4, 0u, 0u, 0, nodes);
    Ctx42 hc; hc.n = n; hc.start = start; hc.dv = dv;
    hc.nodes = nodes.data(); hc.nnodes = (int)nodes.size();

    const char* devname = "cpu";
#ifdef __CUDACC__
    char gname[160] = "cpu";
    uint32_t* d_nodes = nullptr;
    double *d_pv = nullptr, *d_pc1 = nullptr, *d_out = nullptr;
    int d_out_cap = 0; long d_cap = 0;
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s @%d: %s\n",#x,__LINE__, \
    cudaGetErrorString(e)); exit(1);} } while(0)
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
    double symfac = 1.0;
    if (sym) {
        double q = 4.0 / dv;
        if (fabs(q - floor(q + 0.5)) > 1e-9) {
            fprintf(stderr, "--sym needs 4/dv integral\n"); return 1;
        }
        if (i1 > n / 2) i1 = n / 2;
        symfac = 2.0;
    }
    if (!quiet)
        printf("[spec42] dv=%g grid=%d^4 terms(C4)=%zu slices [%d,%d)%s dev=%s\n",
               dv, n, nodes.size(), i0, i1, sym ? " [sym x2]" : "", devname);
    fflush(stdout);

    const double wgt = pow(dv, 4.0) * symfac;
    double U[3] = {0,0,0}, comp[3] = {0,0,0};
    long npts = 0;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<double> pv, pc1;
    pv.reserve(flush + 200000); pc1.reserve(flush + 200000);

    for (int i = i0; i < i1; ++i) {
        const double v = start + i * dv;
        if (fabs(v) < 1.0 + 1e-12) gen42(hc, v, pv, pc1);
        const bool last = (i == i1 - 1);
        if ((long)pv.size() >= flush || (last && pv.size())) {
            const long np = (long)pv.size();
            npts += np;
            double batch[3] = {0,0,0};
#ifdef __CUDACC__
            if (!use_cpu) {
                if (np > d_cap) {
                    if (d_pv) { CK(cudaFree(d_pv)); CK(cudaFree(d_pc1)); }
                    d_cap = np;
                    CK(cudaMalloc(&d_pv,  d_cap * sizeof(double)));
                    CK(cudaMalloc(&d_pc1, d_cap * sizeof(double)));
                }
                CK(cudaMemcpy(d_pv,  pv.data(),  np*sizeof(double), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d_pc1, pc1.data(), np*sizeof(double), cudaMemcpyHostToDevice));
                const int TPB = 128;
                int blocks = (int)std::min<long>(32768, (np + TPB - 1) / TPB);
                if (blocks < 1) blocks = 1;
                if (blocks > d_out_cap) {
                    if (d_out) CK(cudaFree(d_out));
                    d_out_cap = blocks;
                    CK(cudaMalloc(&d_out, 3 * d_out_cap * sizeof(double)));
                }
                Ctx42 c = hc; c.nodes = d_nodes;
                k42<<<blocks, TPB>>>(c, d_pv, d_pc1, np, d_out);
                CK(cudaGetLastError());
                std::vector<double> h(3 * blocks);
                CK(cudaMemcpy(h.data(), d_out, 3*blocks*sizeof(double),
                              cudaMemcpyDeviceToHost));
                double bs[3] = {0,0,0}, bc[3] = {0,0,0};
                for (int b = 0; b < blocks; ++b) kahan3(bs, bc, &h[3*b]);
                for (int d = 0; d < 3; ++d) batch[d] = bs[d];
            } else
#endif
            {
                int nth = 1;
#ifdef _OPENMP
                nth = omp_get_max_threads();
#endif
                std::vector<double> part(3 * nth, 0.0);
#ifdef _OPENMP
#pragma omp parallel
#endif
                {
                    int tid = 0;
#ifdef _OPENMP
                    tid = omp_get_thread_num();
#endif
                    double a[3] = {0,0,0};
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 512) nowait
#endif
                    for (long t = 0; t < np; ++t)
                        eval42_prefix(hc, pv[t], pc1[t], a);
                    for (int d = 0; d < 3; ++d) part[3*tid + d] = a[d];
                }
                double bs[3] = {0,0,0}, bc[3] = {0,0,0};
                for (int t = 0; t < nth; ++t) kahan3(bs, bc, &part[3*t]);
                for (int d = 0; d < 3; ++d) batch[d] = bs[d];
            }
            double scaled[3];
            for (int d = 0; d < 3; ++d) scaled[d] = batch[d] * wgt;
            kahan3(U, comp, scaled);
            pv.clear(); pc1.clear();
        }
    }
    const double total = 6.0*U[0] + 6.0*U[1] + 3.0*U[2];
    double el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
    printf("[RESULT] dv=%g slices=[%d,%d) U1=%+.12f U2=%+.12f U3=%+.12f "
           "{4,2}=6U1+6U2+3U3=%+.12f prefixes=%ld wall=%.1fs dev=%s\n",
           dv, i0, i1, U[0], U[1], U[2], total, npts, el, devname);
    if (!tag.empty()) {
        FILE* f = fopen(tag.c_str(), "a");
        if (f) {
            fprintf(f, "42 %.10g %d %d %.15e %.15e %.15e %.15e %.1f %s\n",
                    dv, i0, i1, U[0], U[1], U[2], total, el, devname);
            fclose(f);
        }
    }
    return 0;
}
