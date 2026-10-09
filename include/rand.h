/* rand.h - small, fast, deterministic PRNG (xoshiro256** style) */
#ifndef GOAI_RAND_H
#define GOAI_RAND_H

#include <math.h>
#include <stdint.h>

typedef struct { uint64_t s[4]; } Rng;

static inline uint64_t rng_splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline uint64_t rng_rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline void rng_seed(Rng *r, uint64_t seed) {
    uint64_t x = seed ? seed : 0x123456789ABCDEFULL;
    for (int i = 0; i < 4; i++) r->s[i] = rng_splitmix64(&x);
    if (!(r->s[0] | r->s[1] | r->s[2] | r->s[3])) r->s[0] = 1;
}

static inline uint64_t rng_next(Rng *r) {
    uint64_t *s = r->s;
    uint64_t result = rng_rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = rng_rotl(s[3], 45);
    return result;
}

/* uniform in [0,1) */
static inline double rng_double(Rng *r) {
    return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* uniform integer in [0,n) —— Lemire 无偏取模。
   只用 64 位运算是为了实现跨平台（GCC 的 __uint128_t 在 MSVC 上不存在），
   结果与 128 位版本一致（仅极少数拒绝采样会让随机流多走一步）。 */
static inline uint32_t rng_below(Rng *r, uint32_t n) {
    if (n == 0) return 0;
    uint64_t x = rng_next(r);
    const uint32_t threshold = (uint32_t)(-(int64_t)n) % n;   /* (2^32 - n) mod n */
    for (;;) {
        const uint64_t a = (x >> 32) * (uint64_t)n;                 /* 高 32 位乘 n */
        const uint64_t b = (x & 0xFFFFFFFFULL) * (uint64_t)n;       /* 低 32 位乘 n */
        const uint32_t hi = (uint32_t)(a >> 32) +
                            (uint32_t)(((a & 0xFFFFFFFFULL) + (b >> 32)) >> 32);
        const uint32_t lo = (uint32_t)b;
        if (lo >= threshold) return hi;
        x = rng_next(r);
    }
}

/* standard normal via Box-Muller */
static inline double rng_normal(Rng *r) {
    double u1 = rng_double(r), u2 = rng_double(r);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* gamma(alpha,1) via Marsaglia-Tsang (alpha > 0) */
static inline double rng_gamma(Rng *r, double alpha) {
    if (alpha < 1.0) {
        double u = rng_double(r);
        if (u < 1e-300) u = 1e-300;
        return rng_gamma(r, alpha + 1.0) * pow(u, 1.0 / alpha);
    }
    double d = alpha - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x = rng_normal(r);
        double v = 1.0 + c * x;
        if (v <= 0.0) continue;
        v = v * v * v;
        double u = rng_double(r);
        if (u < 1e-300) u = 1e-300;
        if (log(u) < 0.5 * x * x + d - d * v + d * log(v)) return d * v;
    }
}

#endif /* GOAI_RAND_H */
