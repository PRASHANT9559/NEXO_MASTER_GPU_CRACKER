
/*
 * NEXO MASTER GPU CRACKER v4.0
 * Optimized High-Performance Hash Cracker
 * Created by: PRASHANT
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <signal.h>
#include <nvml.h>

// --- CUDA Error Checking Infrastructure ---
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s at %s:%d - %s\n", \
                cudaGetErrorString(err), __FILE__, __LINE__, #call); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define KERNEL_CHECK() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[KERNEL ERROR] %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// --- Constant Memory ---
__constant__ uint8_t c_target[32];
__constant__ char c_charset[70];
__constant__ char c_salt[64]; // Salt for salted hashes
__constant__ char c_mask_pattern[64]; // Mask pattern (e.g., "?l?l?d?d")
__constant__ char c_mask_charsets[10][128]; // Character sets for each mask position
// Updated: Move K256 to constant memory for faster access
__constant__ uint32_t K256[64] = {
    0x428a2f98,0x71374498,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
__device__ int c_charset_len;
__device__ int c_target_bytes;
__device__ int c_salt_len;
__device__ int c_mask_len;
__device__ int c_mask_charset_sizes[10];

// --- Global Memory for Dictionary (supports larger wordlists) ---
__device__ char* d_wordlist;
__device__ uint32_t* d_word_indices; // Pre-computed word start indices
__device__ int d_wordlist_size;
__device__ int d_wordlist_count;

// --- SHA-256 Engine ---
__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ uint32_t ep0(uint32_t x) { return rotr(x,2) ^ rotr(x,13) ^ rotr(x,22); }
__device__ __forceinline__ uint32_t ep1(uint32_t x) { return rotr(x,6) ^ rotr(x,11) ^ rotr(x,25); }
__device__ __forceinline__ uint32_t sig0(uint32_t x) { return rotr(x,7) ^ rotr(x,18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t sig1(uint32_t x) { return rotr(x,17) ^ rotr(x,19) ^ (x >> 10); }

// Replaced: Old K256 moved to constant memory above
// __device__ const uint32_t K256[64] = { ... };

__device__ void sha256_transform(uint32_t *state, const uint8_t *chunk) {
    uint32_t W[16]; // Only keep 16 words at a time, compute on-the-fly
    uint32_t a,b,c,d,e,f,g,h,i,T1,T2;

    // Load first 16 words with warp shuffle optimization (FIXED: Commented out to prevent 32x slowdown)
    // int lane_id = threadIdx.x % 32;
    for(i=0;i<16;i++) {
        uint32_t word = (chunk[i*4]<<24)|(chunk[i*4+1]<<16)|(chunk[i*4+2]<<8)|chunk[i*4+3];
        // word = __shfl_sync(0xFFFFFFFF, word, 0); // DISABLED: Each thread has a different candidate!
        W[i] = word;
    }

    a=state[0];b=state[1];c=state[2];d=state[3];e=state[4];f=state[5];g=state[6];h=state[7];

    for(i=0;i<64;i++){
        // Compute W[i] on-the-fly if needed
        if(i >= 16) {
            uint32_t w0 = W[(i-3)&0xF];
            uint32_t w1 = W[(i-8)&0xF];
            uint32_t w2 = W[(i-14)&0xF];
            uint32_t w3 = W[(i-16)&0xF];
            W[i&0xF] = sig1(w0) + w1 + sig0(w2) + w3;
        }

        T1 = h + ep1(e) + ch(e,f,g) + K256[i] + W[i&0xF];
        T2 = ep0(a) + maj(a,b,c);
        h=g;g=f;f=e;e=d+T1;d=c;c=b;b=a;a=T1+T2;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

__device__ void sha256_hash(const char *input, int len, uint8_t *output) {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint64_t total_bits = (uint64_t)len * 8;
    uint64_t chunk_count = (len + 8 + 64) / 64; // +8 for 0x80, +56 for length

    for (uint64_t chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
        uint8_t chunk[64] = {0};
        uint64_t chunk_offset = chunk_idx * 64;

        // Fill chunk with input data
        for (int i = 0; i < 64; i++) {
            uint64_t input_pos = chunk_offset + i;
            if (input_pos < (uint64_t)len) {
                chunk[i] = (uint8_t)input[input_pos];
            } else if (input_pos == (uint64_t)len) {
                chunk[i] = 0x80; // Append 1 bit
                break;
            }
        }

        // Last chunk: add length in bits (big-endian, 8 bytes at end)
        if (chunk_idx == chunk_count - 1) {
            chunk[56] = (total_bits >> 56) & 0xFF;
            chunk[57] = (total_bits >> 48) & 0xFF;
            chunk[58] = (total_bits >> 40) & 0xFF;
            chunk[59] = (total_bits >> 32) & 0xFF;
            chunk[60] = (total_bits >> 24) & 0xFF;
            chunk[61] = (total_bits >> 16) & 0xFF;
            chunk[62] = (total_bits >> 8) & 0xFF;
            chunk[63] = total_bits & 0xFF;
        }

        sha256_transform(h, chunk);
    }

    for(int i=0;i<8;i++) {
        output[i*4] = (h[i] >> 24) & 0xFF; output[i*4+1] = (h[i] >> 16) & 0xFF;
        output[i*4+2] = (h[i] >> 8) & 0xFF; output[i*4+3] = h[i] & 0xFF;
    }
}

// --- MD5 Engine ---
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))
#define ROTL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

__device__ void md5_hash(const char *input, int len, uint8_t *output) {
    uint32_t a = 0x67452301, b = 0xefcdab89, c = 0x98badcfe, d = 0x10325476;
    uint32_t W[16] = {0};
    for(int i=0; i<len; i++) ((uint8_t*)W)[i] = (uint8_t)input[i];
    ((uint8_t*)W)[len] = 0x80;
    W[14] = (uint32_t)(len * 8);

    uint32_t k[] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
    };
    uint32_t s[] = { 7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21 };

    uint32_t tempA = a, tempB = b, tempC = c, tempD = d;
    for(int i=0; i<64; i++) {
        uint32_t f, g;
        if(i<16) { f = F(tempB, tempC, tempD); g = i; }
        else if(i<32) { f = G(tempB, tempC, tempD); g = (5*i + 1) % 16; }
        else if(i<48) { f = H(tempB, tempC, tempD); g = (3*i + 5) % 16; }
        else { f = I(tempB, tempC, tempD); g = (7*i) % 16; }
        uint32_t temp = tempD; tempD = tempC; tempC = tempB;
        tempB = tempB + ROTL(tempA + f + k[i] + W[g], s[(i/16)*4 + (i%4)]);
        tempA = temp;
    }
    a += tempA; b += tempB; c += tempC; d += tempD;
    uint32_t* out32 = (uint32_t*)output;
    out32[0] = a; out32[1] = b; out32[2] = c; out32[3] = d;
}

// --- SHA-1 Engine ---
__device__ void sha1_hash(const char *input, int len, uint8_t *output) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0;
    uint32_t w[80] = {0};
    for (int i = 0; i < len; i++) ((uint8_t*)w)[i ^ 3] = (uint8_t)input[i]; // Big-endian adjust
    ((uint8_t*)w)[len ^ 3] = 0x80;
    w[15] = (uint32_t)(len * 8);

    for (int i = 16; i < 80; i++) w[i] = ROTL(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) { f = (b & c) | ((~b) & d); k = 0x5A827999; }
        else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
        else { f = b ^ c ^ d; k = 0xCA62C1D6; }
        uint32_t temp = ROTL(a, 5) + f + e + k + w[i];
        e = d; d = c; c = ROTL(b, 30); b = a; a = temp;
    }
    h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    uint32_t* out32 = (uint32_t*)output;
    out32[0] = __byte_perm(h0, 0, 0x0123); out32[1] = __byte_perm(h1, 0, 0x0123);
    out32[2] = __byte_perm(h2, 0, 0x0123); out32[3] = __byte_perm(h3, 0, 0x0123);
    out32[4] = __byte_perm(h4, 0, 0x0123);
}

// --- NTLM Engine (Proper MD4) ---
__device__ uint32_t md4_F(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
__device__ uint32_t md4_G(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (x & z) | (y & z); }
__device__ uint32_t md4_H(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }

// ROTL macro from MD5 engine is reused here

__device__ void md4_transform(uint32_t *state, const uint8_t *chunk) {
    uint32_t W[16];
    for(int i=0; i<16; i++) W[i] = chunk[i*4] | (chunk[i*4+1] << 8) | (chunk[i*4+2] << 16) | (chunk[i*4+3] << 24);

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];

    // Round 1
    a = ROTL((a + md4_F(b, c, d) + W[0]), 3);  d = ROTL((d + md4_F(a, b, c) + W[1]), 7);
    c = ROTL((c + md4_F(d, a, b) + W[2]), 11); b = ROTL((b + md4_F(c, d, a) + W[3]), 19);
    a = ROTL((a + md4_F(b, c, d) + W[4]), 3);  d = ROTL((d + md4_F(a, b, c) + W[5]), 7);
    c = ROTL((c + md4_F(d, a, b) + W[6]), 11); b = ROTL((b + md4_F(c, d, a) + W[7]), 19);
    a = ROTL((a + md4_F(b, c, d) + W[8]), 3);  d = ROTL((d + md4_F(a, b, c) + W[9]), 7);
    c = ROTL((c + md4_F(d, a, b) + W[10]), 11); b = ROTL((b + md4_F(c, d, a) + W[11]), 19);
    a = ROTL((a + md4_F(b, c, d) + W[12]), 3);  d = ROTL((d + md4_F(a, b, c) + W[13]), 7);
    c = ROTL((c + md4_F(d, a, b) + W[14]), 11); b = ROTL((b + md4_F(c, d, a) + W[15]), 19);

    // Round 2
    a = ROTL((a + md4_G(b, c, d) + W[0] + 0x5a827999), 3);  d = ROTL((d + md4_G(a, b, c) + W[4] + 0x5a827999), 5);
    c = ROTL((c + md4_G(d, a, b) + W[8] + 0x5a827999), 9);  b = ROTL((b + md4_G(c, d, a) + W[12] + 0x5a827999), 13);
    a = ROTL((a + md4_G(b, c, d) + W[1] + 0x5a827999), 3);  d = ROTL((d + md4_G(a, b, c) + W[5] + 0x5a827999), 5);
    c = ROTL((c + md4_G(d, a, b) + W[9] + 0x5a827999), 9);  b = ROTL((b + md4_G(c, d, a) + W[13] + 0x5a827999), 13);
    a = ROTL((a + md4_G(b, c, d) + W[2] + 0x5a827999), 3);  d = ROTL((d + md4_G(a, b, c) + W[6] + 0x5a827999), 5);
    c = ROTL((c + md4_G(d, a, b) + W[10] + 0x5a827999), 9); b = ROTL((b + md4_G(c, d, a) + W[14] + 0x5a827999), 13);
    a = ROTL((a + md4_G(b, c, d) + W[3] + 0x5a827999), 3);  d = ROTL((d + md4_G(a, b, c) + W[7] + 0x5a827999), 5);
    c = ROTL((c + md4_G(d, a, b) + W[11] + 0x5a827999), 9); b = ROTL((b + md4_G(c, d, a) + W[15] + 0x5a827999), 13);

    // Round 3
    a = ROTL((a + md4_H(b, c, d) + W[0] + 0x6ed9eba1), 3);  d = ROTL((d + md4_H(a, b, c) + W[8] + 0x6ed9eba1), 9);
    c = ROTL((c + md4_H(d, a, b) + W[4] + 0x6ed9eba1), 11); b = ROTL((b + md4_H(c, d, a) + W[12] + 0x6ed9eba1), 15);
    a = ROTL((a + md4_H(b, c, d) + W[2] + 0x6ed9eba1), 3);  d = ROTL((d + md4_H(a, b, c) + W[10] + 0x6ed9eba1), 9);
    c = ROTL((c + md4_H(d, a, b) + W[6] + 0x6ed9eba1), 11); b = ROTL((b + md4_H(c, d, a) + W[14] + 0x6ed9eba1), 15);
    a = ROTL((a + md4_H(b, c, d) + W[1] + 0x6ed9eba1), 3);  d = ROTL((d + md4_H(a, b, c) + W[9] + 0x6ed9eba1), 9);
    c = ROTL((c + md4_H(d, a, b) + W[5] + 0x6ed9eba1), 11); b = ROTL((b + md4_H(c, d, a) + W[13] + 0x6ed9eba1), 15);
    a = ROTL((a + md4_H(b, c, d) + W[3] + 0x6ed9eba1), 3);  d = ROTL((d + md4_H(a, b, c) + W[11] + 0x6ed9eba1), 9);
    c = ROTL((c + md4_H(d, a, b) + W[7] + 0x6ed9eba1), 11); b = ROTL((b + md4_H(c, d, a) + W[15] + 0x6ed9eba1), 15);

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
}

__device__ void md4_hash(const char *input, int len, uint8_t *output) {
    uint32_t h[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};

    // Convert to UTF-16LE for NTLM
    uint8_t utf16[128] = {0};
    for(int i=0; i<len; i++) {
        utf16[i*2] = input[i];
        utf16[i*2+1] = 0;
    }
    int utf16_len = len * 2;

    uint64_t total_bits = (uint64_t)utf16_len * 8;
    uint64_t chunk_count = (utf16_len + 8 + 64) / 64;

    for (uint64_t chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
        uint8_t chunk[64] = {0};
        uint64_t chunk_offset = chunk_idx * 64;

        for (int i = 0; i < 64; i++) {
            uint64_t input_pos = chunk_offset + i;
            if (input_pos < (uint64_t)utf16_len) {
                chunk[i] = utf16[input_pos];
            } else if (input_pos == (uint64_t)utf16_len) {
                chunk[i] = 0x80;
                break;
            }
        }

        if (chunk_idx == chunk_count - 1) {
            chunk[56] = (total_bits >> 56) & 0xFF;
            chunk[57] = (total_bits >> 48) & 0xFF;
            chunk[58] = (total_bits >> 40) & 0xFF;
            chunk[59] = (total_bits >> 32) & 0xFF;
            chunk[60] = (total_bits >> 24) & 0xFF;
            chunk[61] = (total_bits >> 16) & 0xFF;
            chunk[62] = (total_bits >> 8) & 0xFF;
            chunk[63] = total_bits & 0xFF;
        }

        md4_transform(h, chunk);
    }

    for(int i=0; i<4; i++) {
        output[i*4] = h[i] & 0xFF;
        output[i*4+1] = (h[i] >> 8) & 0xFF;
        output[i*4+2] = (h[i] >> 16) & 0xFF;
        output[i*4+3] = (h[i] >> 24) & 0xFF;
    }
}

__device__ void ntlm_hash(const char *input, int len, uint8_t *output) {
    md4_hash(input, len, output);
}

// --- Salted Hash Functions ---
__device__ void md5_salt_pass_hash(const char *input, int len, uint8_t *output) {
    char combined[128];
    int salt_len = c_salt_len;
    for(int i=0; i<len; i++) combined[i] = input[i];
    for(int i=0; i<salt_len; i++) combined[len+i] = c_salt[i];
    md5_hash(combined, len + salt_len, output);
}

__device__ void sha256_salt_pass_hash(const char *input, int len, uint8_t *output) {
    char combined[128];
    int salt_len = c_salt_len;
    for(int i=0; i<salt_len; i++) combined[i] = c_salt[i];
    for(int i=0; i<len; i++) combined[salt_len+i] = input[i];
    sha256_hash(combined, len + salt_len, output);
}

__device__ void sha256_pass_salt_hash(const char *input, int len, uint8_t *output) {
    char combined[128];
    int salt_len = c_salt_len;
    for(int i=0; i<len; i++) combined[i] = input[i];
    for(int i=0; i<salt_len; i++) combined[len+i] = c_salt[i];
    sha256_hash(combined, len + salt_len, output);
}

// --- Global Result State ---
__device__ int d_found = 0;
__device__ char d_result[32];

// --- Checkpoint State ---
typedef struct {
    int hash_choice;
    int attack_mode;
    int min_len;
    int max_len;
    int current_len;
    uint64_t offset;
    uint64_t total_scanned;
    uint64_t fixed_limit;
    time_t start_time;
    char hex_input[128];
    char salt_input[64];
    char wordlist_path[256];
    int salt_len;
} CheckpointState;

// --- Real-time Stats ---
// StatsState struct is defined later in the file

// --- NVML GPU Monitoring ---
typedef struct {
    unsigned int temperature;
    unsigned int power_draw;
    unsigned int fan_speed;
    unsigned int utilization;
    char name[256];
} GPUStats;

bool initNVML() {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "[NVML ERROR] Failed to initialize: %s\n", nvmlErrorString(result));
        return false;
    }
    return true;
}

GPUStats getGPUStats(int device) {
    nvmlDevice_t nvml_dev;
    GPUStats stats = {0};

    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(device, &nvml_dev);
    if (result != NVML_SUCCESS) {
        return stats;
    }

    nvmlDeviceGetTemperature(nvml_dev, NVML_TEMPERATURE_GPU, &stats.temperature);
    nvmlDeviceGetPowerUsage(nvml_dev, &stats.power_draw);
    nvmlDeviceGetFanSpeed(nvml_dev, &stats.fan_speed);
    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(nvml_dev, &util);
    stats.utilization = util.gpu;

    char name[256];
    nvmlDeviceGetName(nvml_dev, name, sizeof(name));
    strcpy(stats.name, name);

    return stats;
}

void shutdownNVML() {
    nvmlShutdown();
}

// --- GPU Auto-Tuning ---
typedef struct {
    int compute_major;
    int compute_minor;
    int multi_processor_count;
    int max_threads_per_mp;
    size_t total_global_mem;
    char name[256];
} GPUInfo;

GPUInfo getGPUInfo(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    GPUInfo info = {
        .compute_major = prop.major,
        .compute_minor = prop.minor,
        .multi_processor_count = prop.multiProcessorCount,
        .max_threads_per_mp = prop.maxThreadsPerMultiProcessor,
        .total_global_mem = prop.totalGlobalMem
    };
    strcpy(info.name, prop.name);
    return info;
}

void autoTuneGPU(int device, int* blocks, int* threads) {
    GPUInfo info = getGPUInfo(device);
    int compute_cap = info.compute_major * 10 + info.compute_minor;

    // Auto-tune based on architecture
    switch(compute_cap) {
        case 52: // Maxwell
            *blocks = 1024;
            *threads = 128;
            break;
        case 61: // Pascal
        case 62:
            *blocks = 2048;
            *threads = 128;
            break;
        case 70: // Volta
            *blocks = 2048;
            *threads = 256;
            break;
        case 75: // Turing
            *blocks = 2048;
            *threads = 256;
            break;
        case 80: // Ampere
        case 86:
            *blocks = 4096;
            *threads = 256;
            break;
        case 89: // Ada Lovelace
            *blocks = 4096;
            *threads = 512;
            break;
        case 90: // Hopper
            *blocks = 8192;
            *threads = 512;
            break;
        default: // Fallback for unknown architectures
            *blocks = 2048;
            *threads = 256;
            break;
    }

    printf("🔧 Auto-tuned GPU %d (%s): %d blocks, %d threads\n",
           device, info.name, *blocks, *threads);
}

void saveCheckpoint(const char* filename, CheckpointState* state) {
    FILE* fp = fopen(filename, "wb");
    if (fp) {
        fwrite(state, sizeof(CheckpointState), 1, fp);
        fclose(fp);
        printf("\n💾 Checkpoint saved to %s\n", filename);
    }
}

int loadCheckpoint(const char* filename, CheckpointState* state) {
    FILE* fp = fopen(filename, "rb");
    if (fp) {
        if (fread(state, sizeof(CheckpointState), 1, fp) != 1) {
            // Error reading checkpoint
        }
        fclose(fp);
        printf("\n📂 Checkpoint loaded from %s\n", filename);
        return 1;
    }
    return 0;
}

int checkPotfile(const char* potfile_path, const char* target_hash, char* found_password) {
    FILE* fp = fopen(potfile_path, "r");
    if (!fp) return 0;

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        char hash[128];
        char password[128];
        if (sscanf(line, "%127s:%127s", hash, password) == 2) {
            if (strcmp(hash, target_hash) == 0) {
                strcpy(found_password, password);
                fclose(fp);
                return 1;
            }
        }
    }
    fclose(fp);
    return 0;
}

void addToPotfile(const char* potfile_path, const char* hash, const char* password) {
    FILE* fp = fopen(potfile_path, "a");
    if (fp) {
        fprintf(fp, "%s:%s\n", hash, password);
        fclose(fp);
        printf("\n💾 Saved to potfile: %s\n", potfile_path);
    }
}

// --- Improved Real-time Stats ---
typedef struct {
    uint64_t current_hashes;
    uint64_t total_hashes;
    uint64_t hashes_per_second;
    uint64_t avg_hashes_per_second;
    uint64_t peak_hashes_per_second;
    double progress_percent;
    double eta_seconds;
    time_t start_time;
    time_t last_update;
    uint64_t last_hashes;

    // Moving average for smooth H/s
    double hps_history[10];
    int hps_history_idx;

    // Progress bar state
    int bar_width;
} StatsState;

void initStats(StatsState* stats, uint64_t total_hashes, time_t start_time) {
    memset(stats, 0, sizeof(StatsState));
    stats->total_hashes = total_hashes;
    stats->start_time = start_time;
    stats->last_update = start_time;
    stats->last_hashes = 0;
    stats->bar_width = 40;

    for (int i = 0; i < 10; i++) stats->hps_history[i] = 0.0;
    stats->hps_history_idx = 0;
}

void updateStats(StatsState* stats, uint64_t hashes_processed) {
    time_t current_time = time(NULL);
    double delta_time = difftime(current_time, stats->last_update);

    stats->current_hashes = hashes_processed;

    uint64_t hashes_delta = hashes_processed - stats->last_hashes;
    uint64_t instant_hps = 0;

    if (delta_time > 0) {
        instant_hps = (uint64_t)(hashes_delta / delta_time);
    }

    stats->hps_history[stats->hps_history_idx] = (double)instant_hps;
    stats->hps_history_idx = (stats->hps_history_idx + 1) % 10;

    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < 10; i++) {
        if (stats->hps_history[i] > 0) {
            sum += stats->hps_history[i];
            count++;
        }
    }
    stats->hashes_per_second = (count > 0) ? (uint64_t)(sum / count) : instant_hps;

    if (stats->hashes_per_second > stats->peak_hashes_per_second) {
        stats->peak_hashes_per_second = stats->hashes_per_second;
    }

    double total_elapsed = difftime(current_time, stats->start_time);
    if (total_elapsed > 0) {
        stats->avg_hashes_per_second = (uint64_t)(hashes_processed / total_elapsed);
    }

    if (stats->total_hashes > 0) {
        stats->progress_percent = (double)hashes_processed / stats->total_hashes * 100.0;
    }

    if (stats->hashes_per_second > 0 && stats->total_hashes > hashes_processed) {
        uint64_t remaining = stats->total_hashes - hashes_processed;
        double weighted_hps = (stats->hashes_per_second * 0.7) + (stats->avg_hashes_per_second * 0.3);
        stats->eta_seconds = remaining / weighted_hps;

        if (stats->eta_seconds < 0) stats->eta_seconds = 0;
        if (stats->eta_seconds > 31536000) stats->eta_seconds = 31536000;
    } else {
        stats->eta_seconds = 0;
    }

    stats->last_update = current_time;
    stats->last_hashes = hashes_processed;
}

void formatNumber(uint64_t num, char* buffer, int buffer_size) {
    if (num >= 1e12) {
        snprintf(buffer, buffer_size, "%.2fT", num / 1e12);
    } else if (num >= 1e9) {
        snprintf(buffer, buffer_size, "%.2fG", num / 1e9);
    } else if (num >= 1e6) {
        snprintf(buffer, buffer_size, "%.2fM", num / 1e6);
    } else if (num >= 1e3) {
        snprintf(buffer, buffer_size, "%.2fK", num / 1e3);
    } else {
        snprintf(buffer, buffer_size, "%lu", num);
    }
}

void formatTime(double seconds, char* buffer, int buffer_size) {
    if (seconds < 60) {
        snprintf(buffer, buffer_size, "%.0fs", seconds);
    } else if (seconds < 3600) {
        int mins = (int)(seconds / 60);
        int secs = (int)(seconds) % 60;
        snprintf(buffer, buffer_size, "%dm %ds", mins, secs);
    } else if (seconds < 86400) {
        int hours = (int)(seconds / 3600);
        int mins = (int)((long long)(seconds / 60) % 60);
        snprintf(buffer, buffer_size, "%dh %dm", hours, mins);
    } else {
        int days = (int)(seconds / 86400);
        int hours = (int)((long long)(seconds / 3600) % 24);
        snprintf(buffer, buffer_size, "%dd %dh", days, hours);
    }
}

void displayStats(StatsState* stats) {
    char current_h_str[32], total_h_str[32], hps_str[32], avg_hps_str[32], peak_hps_str[32];
    char eta_str[32], elapsed_str[32];

    formatNumber(stats->current_hashes, current_h_str, sizeof(current_h_str));
    formatNumber(stats->total_hashes, total_h_str, sizeof(total_h_str));
    formatNumber(stats->hashes_per_second, hps_str, sizeof(hps_str));
    formatNumber(stats->avg_hashes_per_second, avg_hps_str, sizeof(avg_hps_str));
    formatNumber(stats->peak_hashes_per_second, peak_hps_str, sizeof(peak_hps_str));

    double elapsed = difftime(time(NULL), stats->start_time);
    formatTime(elapsed, elapsed_str, sizeof(elapsed_str));
    formatTime(stats->eta_seconds, eta_str, sizeof(eta_str));

    int filled = (int)(stats->progress_percent / 100.0 * stats->bar_width);
    if (filled > stats->bar_width) filled = stats->bar_width;

    printf("\r");
    printf("[");
    for (int i = 0; i < stats->bar_width; i++) {
        if (i < filled) printf("█");
        else printf("░");
    }
    printf("] %.2f%%  ", stats->progress_percent);

    printf("⚡ %sH/s | 📈 %sH/s (avg) | 🔝 %sH/s (peak)  ", hps_str, avg_hps_str, peak_hps_str);
    printf("📊 %s/%s  ", current_h_str, total_h_str);
    printf("⏱️ %s elapsed | ⏳ %s ETA", elapsed_str, eta_str);

    fflush(stdout);
}

__global__ void crackKernel(uint64_t offset, int len, int iterations, int type);

void runBenchmark() {
    printf("\n========================================\n");
    printf("   📊 NEXO REAL GPU BENCHMARK\n");
    printf("========================================\n\n");

    const char* hash_names[] = {
        "SHA256 (64 hex)", "SHA256 (32 hex)", "MD5 (32 hex)",
        "SHA-1 (40 hex)", "NTLM (32 hex)", "MySQL41 (40 hex)",
        "MD5($pass.$salt)", "SHA256($salt.$pass)", "SHA256($pass.$salt)"
    };

    int threads = 256, blocks = 2048, iterations = 10000;
    uint64_t batch_size = (uint64_t)blocks * threads * iterations;

    printf("Testing with %d threads, %d blocks, %d iterations per batch\n", threads, blocks, iterations);
    printf("Batch size: %lu hashes per kernel launch\n\n", batch_size);

    // Warmup kernel
    crackKernel<<<blocks, threads>>>(0, 6, 100, 1);
    cudaDeviceSynchronize();
    cudaDeviceSynchronize();

    printf("Running 3 iterations per hash type for accuracy...\n\n");

    for (int type = 1; type <= 9; type++) {
        int h_found = 0;
        cudaMemcpyToSymbol(d_found, &h_found, sizeof(int));

        double total_hashes = 0;
        double total_time = 0;

        for (int run = 0; run < 3; run++) {
            int h_found_run = 0;
            cudaMemcpyToSymbol(d_found, &h_found_run, sizeof(int));

            clock_t start = clock();
            crackKernel<<<blocks, threads>>>(0, 6, iterations, type);
            cudaDeviceSynchronize();
            clock_t end = clock();

            double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
            total_time += elapsed;
            total_hashes += batch_size;
        }

        double avg_hashes_per_sec = total_hashes / total_time;
        printf("%-20s %12.0f H/s measured on this GPU\n", hash_names[type-1], avg_hashes_per_sec);
    }

    printf("\n✅ Real benchmark complete!\n");
}

int parseMaskPattern(const char* pattern, char charsets[10][128], int charset_sizes[10], const char* custom_charsets[4]) {
    const char* lowercase = "abcdefghijklmnopqrstuvwxyz";
    const char* uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const char* digits = "0123456789";
    const char* special = "!@#$%^&*()_+-=[]{}|;:,.<>?";
    const char* all = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?";

    int mask_len = strlen(pattern);
    int pos = 0;

    for (int i = 0; i < mask_len && pos < 10; i++) {
        if (pattern[i] == '?') {
            i++;
            if (i >= mask_len) break;

            switch(pattern[i]) {
                case 'l': // lowercase
                    strcpy(charsets[pos], lowercase);
                    charset_sizes[pos] = 26;
                    break;
                case 'u': // uppercase
                    strcpy(charsets[pos], uppercase);
                    charset_sizes[pos] = 26;
                    break;
                case 'd': // digits
                    strcpy(charsets[pos], digits);
                    charset_sizes[pos] = 10;
                    break;
                case 's': // special
                    strcpy(charsets[pos], special);
                    charset_sizes[pos] = strlen(special);
                    break;
                case 'a': // all
                    strcpy(charsets[pos], all);
                    charset_sizes[pos] = strlen(all);
                    break;
                case '1': // custom charset 1
                    if (custom_charsets[0] != NULL) {
                        strcpy(charsets[pos], custom_charsets[0]);
                        charset_sizes[pos] = strlen(custom_charsets[0]);
                    } else {
                        strcpy(charsets[pos], all);
                        charset_sizes[pos] = strlen(all);
                    }
                    break;
                case '2': // custom charset 2
                    if (custom_charsets[1] != NULL) {
                        strcpy(charsets[pos], custom_charsets[1]);
                        charset_sizes[pos] = strlen(custom_charsets[1]);
                    } else {
                        strcpy(charsets[pos], all);
                        charset_sizes[pos] = strlen(all);
                    }
                    break;
                case '3': // custom charset 3
                    if (custom_charsets[2] != NULL) {
                        strcpy(charsets[pos], custom_charsets[2]);
                        charset_sizes[pos] = strlen(custom_charsets[2]);
                    } else {
                        strcpy(charsets[pos], all);
                        charset_sizes[pos] = strlen(all);
                    }
                    break;
                case '4': // custom charset 4
                    if (custom_charsets[3] != NULL) {
                        strcpy(charsets[pos], custom_charsets[3]);
                        charset_sizes[pos] = strlen(custom_charsets[3]);
                    } else {
                        strcpy(charsets[pos], all);
                        charset_sizes[pos] = strlen(all);
                    }
                    break;
                default:
                    strcpy(charsets[pos], all);
                    charset_sizes[pos] = strlen(all);
                    break;
            }
        } else {
            // Static character
            charsets[pos][0] = pattern[i];
            charsets[pos][1] = '\0';
            charset_sizes[pos] = 1;
        }
        pos++;
    }

    return pos;
}

__device__ void indexToPassword(uint64_t n, int len, char* out) {
    for (int i = len - 1; i >= 0; i--) {
        out[i] = c_charset[n % c_charset_len];
        n /= c_charset_len;
    }
    out[len] = '\0';
}

// --- Specialized Kernels for Each Hash Type (No Divergence) ---
__global__ void crackSHA256(uint64_t offset, int len, int iterations) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_n = offset + (idx * iterations);
    char candidate[64];
    uint8_t hash[32];

    for (int i = 0; i < iterations; i++) {
        if (d_found) return;
        indexToPassword(start_n + i, len, candidate);
        sha256_hash(candidate, len, hash);

        bool match = true;
        for (int k = 0; k < c_target_bytes; k++) {
            if (hash[k] != c_target[k]) { match = false; break; }
        }

        if (match) {
            if (atomicExch(&d_found, 1) == 0) {
                for(int k=0; k<=len; k++) d_result[k] = candidate[k];
            }
            return;
        }
    }
}

__global__ void crackMD5(uint64_t offset, int len, int iterations) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_n = offset + (idx * iterations);
    char candidate[64];
    uint8_t hash[32];

    for (int i = 0; i < iterations; i++) {
        if (d_found) return;
        indexToPassword(start_n + i, len, candidate);
        md5_hash(candidate, len, hash);

        bool match = true;
        for (int k = 0; k < c_target_bytes; k++) {
            if (hash[k] != c_target[k]) { match = false; break; }
        }

        if (match) {
            if (atomicExch(&d_found, 1) == 0) {
                for(int k=0; k<=len; k++) d_result[k] = candidate[k];
            }
            return;
        }
    }
}

__global__ void crackSHA1(uint64_t offset, int len, int iterations) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_n = offset + (idx * iterations);
    char candidate[64];
    uint8_t hash[32];

    for (int i = 0; i < iterations; i++) {
        if (d_found) return;
        indexToPassword(start_n + i, len, candidate);
        sha1_hash(candidate, len, hash);

        bool match = true;
        for (int k = 0; k < c_target_bytes; k++) {
            if (hash[k] != c_target[k]) { match = false; break; }
        }

        if (match) {
            if (atomicExch(&d_found, 1) == 0) {
                for(int k=0; k<=len; k++) d_result[k] = candidate[k];
            }
            return;
        }
    }
}

__global__ void crackNTLM(uint64_t offset, int len, int iterations) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_n = offset + (idx * iterations);
    char candidate[64];
    uint8_t hash[32];

    for (int i = 0; i < iterations; i++) {
        if (d_found) return;
        indexToPassword(start_n + i, len, candidate);
        ntlm_hash(candidate, len, hash);

        bool match = true;
        for (int k = 0; k < c_target_bytes; k++) {
            if (hash[k] != c_target[k]) { match = false; break; }
        }

        if (match) {
            if (atomicExch(&d_found, 1) == 0) {
                for(int k=0; k<=len; k++) d_result[k] = candidate[k];
            }
            return;
        }
    }
}

__global__ void crackMySQL41(uint64_t offset, int len, int iterations) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_n = offset + (idx * iterations);
    char candidate[64];
    uint8_t hash[32];

    for (int i = 0; i < iterations; i++) {
        if (d_found) return;
        indexToPassword(start_n + i, len, candidate);
        uint8_t tmp[20];
        sha1_hash(candidate, len, tmp);
        sha1_hash((char*)tmp, 20, hash);

        bool match = true;
        for (int k = 0; k < c_target_bytes; k++) {
            if (hash[k] != c_target[k]) { match = false; break; }
        }

        if (match) {
            if (atomicExch(&d_found, 1) == 0) {
                for(int k=0; k<=len; k++) d_result[k] = candidate[k];
            }
            return;
        }
    }
}

// Generic kernel for salted hashes (keep switch for these less common cases)
__global__ void crackKernel(uint64_t offset, int len, int iterations, int type) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_n = offset + (idx * iterations);
    char candidate[64];
    uint8_t hash[32];

    for (int i = 0; i < iterations; i++) {
        if (d_found) return;
        indexToPassword(start_n + i, len, candidate);

        switch(type) {
            case 7: md5_salt_pass_hash(candidate, len, hash); break;
            case 8: sha256_salt_pass_hash(candidate, len, hash); break;
            case 9: sha256_pass_salt_hash(candidate, len, hash); break;
            default: return; // Should not reach here
        }

        bool match = true;
        for (int k = 0; k < c_target_bytes; k++) {
            if (hash[k] != c_target[k]) { match = false; break; }
        }

        if (match) {
            if (atomicExch(&d_found, 1) == 0) {
                for(int k=0; k<=len; k++) d_result[k] = candidate[k];
            }
            return;
        }
    }
}

__global__ void dictionaryKernel(int type, int apply_rules, int rule_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_wordlist_count) return;

    // Coalesced memory access: load 4 bytes at a time using uint32_t*
    uint32_t* d_wordlist_32 = (uint32_t*)d_wordlist;

    // O(1) word access using pre-computed indices
    uint32_t word_start = d_word_indices[idx];
    uint32_t word_end = (idx < d_wordlist_count - 1) ? d_word_indices[idx + 1] : d_wordlist_size;

    char candidate[64];
    int len = 0;

    // FIXED: Reverted to safe character-by-character read to prevent stack overflow/garbage reads
    // (The previous "coalesced" attempt was reading out of bounds of the local uint32_t chunk)
    for (uint32_t i = word_start; i < word_end && d_wordlist[i] != '\n' && d_wordlist[i] != '\0'; i++) {
        if (len < 63) candidate[len++] = d_wordlist[i];
    }
    candidate[len] = '\0';

    if (len == 0) return;

    // Apply rules if enabled (GPU-side rule processing)
    if (apply_rules && rule_count > 0) {
        // Simple rule application: try each rule variant
        // For production, would need full rule engine in GPU memory
        char temp_candidate[64];
        for (int r = 0; r < rule_count && r < 10; r++) {
            // Copy original word
            for (int j = 0; j < len; j++) temp_candidate[j] = candidate[j];
            temp_candidate[len] = '\0';
            int temp_len = len;

            // Apply rule transformations (simplified for GPU)
            // Rule 0: lowercase
            if (r == 0) {
                for (int j = 0; j < temp_len; j++) {
                    if (temp_candidate[j] >= 'A' && temp_candidate[j] <= 'Z') {
                        temp_candidate[j] += 32;
                    }
                }
            }
            // Rule 1: uppercase
            else if (r == 1) {
                for (int j = 0; j < temp_len; j++) {
                    if (temp_candidate[j] >= 'a' && temp_candidate[j] <= 'z') {
                        temp_candidate[j] -= 32;
                    }
                }
            }
            // Rule 2: capitalize first letter
            else if (r == 2) {
                if (temp_candidate[0] >= 'a' && temp_candidate[0] <= 'z') {
                    temp_candidate[0] -= 32;
                }
            }
            // Rule 3: append 's'
            else if (r == 3 && temp_len < 63) {
                temp_candidate[temp_len] = 's';
                temp_candidate[temp_len + 1] = '\0';
                temp_len++;
            }
            // Rule 4: reverse
            else if (r == 4) {
                for (int j = 0; j < temp_len / 2; j++) {
                    char tmp = temp_candidate[j];
                    temp_candidate[j] = temp_candidate[temp_len - 1 - j];
                    temp_candidate[temp_len - 1 - j] = tmp;
                }
            }

            // Hash the transformed word
            uint8_t hash[32];
            switch(type) {
                case 1: case 2: sha256_hash(temp_candidate, temp_len, hash); break;
                case 3: md5_hash(temp_candidate, temp_len, hash); break;
                case 4: sha1_hash(temp_candidate, temp_len, hash); break;
                case 5: ntlm_hash(temp_candidate, temp_len, hash); break;
                case 6: { uint8_t tmp[20]; sha1_hash(temp_candidate, temp_len, tmp); sha1_hash((char*)tmp, 20, hash); } break;
                case 7: md5_salt_pass_hash(temp_candidate, temp_len, hash); break;
                case 8: sha256_salt_pass_hash(temp_candidate, temp_len, hash); break;
                case 9: sha256_pass_salt_hash(temp_candidate, temp_len, hash); break;
            }

            bool match = true;
            for (int k = 0; k < c_target_bytes; k++) {
                if (hash[k] != c_target[k]) { match = false; break; }
            }

            if (match) {
                if (atomicExch(&d_found, 1) == 0) {
                    for(int k=0; k<=temp_len; k++) d_result[k] = temp_candidate[k];
                }
                return;
            }
        }
    }

    // Hash original word
    uint8_t hash[32];
    switch(type) {
        case 1: case 2: sha256_hash(candidate, len, hash); break;
        case 3: md5_hash(candidate, len, hash); break;
        case 4: sha1_hash(candidate, len, hash); break;
        case 5: ntlm_hash(candidate, len, hash); break;
        case 6: { uint8_t tmp[20]; sha1_hash(candidate, len, tmp); sha1_hash((char*)tmp, 20, hash); } break;
        case 7: md5_salt_pass_hash(candidate, len, hash); break;
        case 8: sha256_salt_pass_hash(candidate, len, hash); break;
        case 9: sha256_pass_salt_hash(candidate, len, hash); break;
    }

    bool match = true;
    for (int k = 0; k < c_target_bytes; k++) {
        if (hash[k] != c_target[k]) { match = false; break; }
    }

    if (match) {
        if (atomicExch(&d_found, 1) == 0) {
            for(int k=0; k<=len; k++) d_result[k] = candidate[k];
        }
    }
}

__global__ void maskKernel(uint64_t offset, int type) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_n = offset + idx;
    char candidate[64];

    // Generate password from mask pattern
    uint64_t temp = start_n;
    for (int pos = c_mask_len - 1; pos >= 0; pos--) {
        int charset_size = c_mask_charset_sizes[pos];
        candidate[pos] = c_mask_charsets[pos][temp % charset_size];
        temp /= charset_size;
    }
    candidate[c_mask_len] = '\0';

    uint8_t hash[32];
    switch(type) {
        case 1: case 2: sha256_hash(candidate, c_mask_len, hash); break;
        case 3: md5_hash(candidate, c_mask_len, hash); break;
        case 4: sha1_hash(candidate, c_mask_len, hash); break;
        case 5: ntlm_hash(candidate, c_mask_len, hash); break;
        case 6: { uint8_t tmp[20]; sha1_hash(candidate, c_mask_len, tmp); sha1_hash((char*)tmp, 20, hash); } break;
        case 7: md5_salt_pass_hash(candidate, c_mask_len, hash); break;
        case 8: sha256_salt_pass_hash(candidate, c_mask_len, hash); break;
        case 9: sha256_pass_salt_hash(candidate, c_mask_len, hash); break;
    }

    bool match = true;
    for (int k = 0; k < c_target_bytes; k++) {
        if (hash[k] != c_target[k]) { match = false; break; }
    }

    if (match) {
        if (atomicExch(&d_found, 1) == 0) {
            for(int k=0; k<=c_mask_len; k++) d_result[k] = candidate[k];
        }
    }
}

// --- Enhanced Mask Kernel with Iteration Loop ---
__global__ void maskKernelIter(uint64_t offset, int iterations, int type) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_n = offset + (idx * iterations);
    char candidate[64];

    for (int i = 0; i < iterations; i++) {
        if (d_found) return;

        uint64_t temp = start_n + i;
        for (int pos = c_mask_len - 1; pos >= 0; pos--) {
            int charset_size = c_mask_charset_sizes[pos];
            candidate[pos] = c_mask_charsets[pos][temp % charset_size];
            temp /= charset_size;
        }
        candidate[c_mask_len] = '\0';

        uint8_t hash[32];
        switch(type) {
            case 1: case 2: sha256_hash(candidate, c_mask_len, hash); break;
            case 3: md5_hash(candidate, c_mask_len, hash); break;
            case 4: sha1_hash(candidate, c_mask_len, hash); break;
            case 5: ntlm_hash(candidate, c_mask_len, hash); break;
            case 6: { uint8_t tmp[20]; sha1_hash(candidate, c_mask_len, tmp); sha1_hash((char*)tmp, 20, hash); } break;
            case 7: md5_salt_pass_hash(candidate, c_mask_len, hash); break;
            case 8: sha256_salt_pass_hash(candidate, c_mask_len, hash); break;
            case 9: sha256_pass_salt_hash(candidate, c_mask_len, hash); break;
        }

        bool match = true;
        for (int k = 0; k < c_target_bytes; k++) {
            if (hash[k] != c_target[k]) { match = false; break; }
        }

        if (match) {
            if (atomicExch(&d_found, 1) == 0) {
                for(int k=0; k<=c_mask_len; k++) d_result[k] = candidate[k];
            }
            return;
        }
    }
}

// --- Hybrid Attack Kernel (Dictionary + Mask) ---
__global__ void hybridKernel(int word_idx, uint64_t mask_offset, int mask_iterations, int type) {
    if (word_idx >= d_wordlist_count) return;

    // Get dictionary word
    uint32_t word_start = d_word_indices[word_idx];
    uint32_t word_end = (word_idx < d_wordlist_count - 1) ? d_word_indices[word_idx + 1] : d_wordlist_size;

    char dict_word[64];
    int dict_len = 0;
    for (uint32_t i = word_start; i < word_end && d_wordlist[i] != '\n' && d_wordlist[i] != '\0'; i++) {
        dict_word[dict_len++] = d_wordlist[i];
    }
    dict_word[dict_len] = '\0';

    if (dict_len == 0) return;

    // Generate mask combinations and append to dictionary word
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_n = mask_offset + (idx * mask_iterations);

    for (int i = 0; i < mask_iterations; i++) {
        if (d_found) return;

        char candidate[128];
        uint64_t temp = start_n + i;

        // Copy dictionary word first
        for (int j = 0; j < dict_len; j++) {
            candidate[j] = dict_word[j];
        }

        // Append mask pattern
        for (int pos = c_mask_len - 1; pos >= 0; pos--) {
            int charset_size = c_mask_charset_sizes[pos];
            candidate[dict_len + pos] = c_mask_charsets[pos][temp % charset_size];
            temp /= charset_size;
        }
        candidate[dict_len + c_mask_len] = '\0';

        uint8_t hash[32];
        int total_len = dict_len + c_mask_len;
        switch(type) {
            case 1: case 2: sha256_hash(candidate, total_len, hash); break;
            case 3: md5_hash(candidate, total_len, hash); break;
            case 4: sha1_hash(candidate, total_len, hash); break;
            case 5: ntlm_hash(candidate, total_len, hash); break;
            case 6: { uint8_t tmp[20]; sha1_hash(candidate, total_len, tmp); sha1_hash((char*)tmp, 20, hash); } break;
            case 7: md5_salt_pass_hash(candidate, total_len, hash); break;
            case 8: sha256_salt_pass_hash(candidate, total_len, hash); break;
            case 9: sha256_pass_salt_hash(candidate, total_len, hash); break;
        }

        bool match = true;
        for (int k = 0; k < c_target_bytes; k++) {
            if (hash[k] != c_target[k]) { match = false; break; }
        }

        if (match) {
            if (atomicExch(&d_found, 1) == 0) {
                for(int k=0; k<=total_len; k++) d_result[k] = candidate[k];
            }
            return;
        }
    }
}

// --- Rule Engine (Hashcat-Compatible) ---
#define MAX_RULES 1000
#define MAX_WORD_LENGTH 128
#define MAX_RULE_LINE 256

typedef struct {
    char rule[16];
    int param;
} Rule;

Rule rules[MAX_RULES];
int rule_count = 0;

void applyRule(char* word, const Rule* rule) {
    int len = strlen(word);
    if (len == 0) return;

    switch(rule->rule[0]) {
        case ':': // No op
            break;
        case 'l': // Lowercase all
            for (int i = 0; i < len; i++) {
                if (word[i] >= 'A' && word[i] <= 'Z') {
                    word[i] += 32;
                }
            }
            break;
        case 'u': // Uppercase all
            for (int i = 0; i < len; i++) {
                if (word[i] >= 'a' && word[i] <= 'z') {
                    word[i] -= 32;
                }
            }
            break;
        case 'c': // Capitalize first letter
            if (word[0] >= 'a' && word[0] <= 'z') {
                word[0] -= 32;
            }
            break;
        case 'C': // Lowercase first, uppercase rest
            if (word[0] >= 'A' && word[0] <= 'Z') {
                word[0] += 32;
            }
            for (int i = 1; i < len; i++) {
                if (word[i] >= 'a' && word[i] <= 'z') {
                    word[i] -= 32;
                }
            }
            break;
        case 't': // Toggle case
            for (int i = 0; i < len; i++) {
                if (word[i] >= 'a' && word[i] <= 'z') {
                    word[i] -= 32;
                } else if (word[i] >= 'A' && word[i] <= 'Z') {
                    word[i] += 32;
                }
            }
            break;
        case 'r': // Reverse word
            for (int i = 0; i < len / 2; i++) {
                char temp = word[i];
                word[i] = word[len - 1 - i];
                word[len - 1 - i] = temp;
            }
            break;
        case 'd': // Duplicate word
            if (len * 2 < MAX_WORD_LENGTH - 1) {
                memmove(word + len, word, len);
                word[len * 2] = '\0';
            }
            break;
        case 'p': // Pluralize (add 's')
            if (len < MAX_WORD_LENGTH - 1) {
                word[len] = 's';
                word[len + 1] = '\0';
            }
            break;
        case '$': // Append character
            if (len < MAX_WORD_LENGTH - 1) {
                word[len] = (char)rule->param;
                word[len + 1] = '\0';
            }
            break;
        case '^': // Prepend character
            if (len < MAX_WORD_LENGTH - 1) {
                memmove(word + 1, word, len + 1);
                word[0] = (char)rule->param;
            }
            break;
        case '@': // Remove all occurrences
            {
                char target = (char)rule->param;
                int j = 0;
                for (int i = 0; word[i]; i++) {
                    if (word[i] != target) {
                        word[j++] = word[i];
                    }
                }
                word[j] = '\0';
            }
            break;
        case '!': // Reject if contains
            {
                char target = (char)rule->param;
                for (int i = 0; word[i]; i++) {
                    if (word[i] == target) {
                        word[0] = '\0'; // Mark as rejected
                        return;
                    }
                }
            }
            break;
        case '+': // Increment char at position
            {
                int pos = rule->param;
                if (pos >= 0 && pos < len) {
                    if (word[pos] >= 'a' && word[pos] <= 'z') {
                        word[pos]++;
                        if (word[pos] > 'z') word[pos] = 'a';
                    } else if (word[pos] >= 'A' && word[pos] <= 'Z') {
                        word[pos]++;
                        if (word[pos] > 'Z') word[pos] = 'A';
                    } else if (word[pos] >= '0' && word[pos] <= '9') {
                        word[pos]++;
                        if (word[pos] > '9') word[pos] = '0';
                    }
                }
            }
            break;
        case '-': // Decrement char at position
            {
                int pos = rule->param;
                if (pos >= 0 && pos < len) {
                    if (word[pos] >= 'a' && word[pos] <= 'z') {
                        word[pos]--;
                        if (word[pos] < 'a') word[pos] = 'z';
                    } else if (word[pos] >= 'A' && word[pos] <= 'Z') {
                        word[pos]--;
                        if (word[pos] < 'A') word[pos] = 'Z';
                    } else if (word[pos] >= '0' && word[pos] <= '9') {
                        word[pos]--;
                        if (word[pos] < '0') word[pos] = '9';
                    }
                }
            }
            break;
    }
}

void estimateHashRate() {
    printf("\n========================================\n");
    printf("   📈 HASH RATE ESTIMATOR\n");
    printf("========================================\n\n");

    int device_count;
    cudaGetDeviceCount(&device_count);

    printf("Detected %d GPU(s). Running quick tests...\n\n", device_count);

    // Quick test logic (similar to benchmark but shorter)
    int threads = 256, blocks = 2048, iterations = 5000;
    uint64_t batch_size = (uint64_t)blocks * threads * iterations;

    for (int type = 1; type <= 5; type++) {
        clock_t start = clock();
        crackKernel<<<blocks, threads>>>(0, 6, iterations, type);
        cudaDeviceSynchronize();
        clock_t end = clock();

        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double hps = batch_size / elapsed;

        char hps_str[32];
        formatNumber((uint64_t)hps, hps_str, sizeof(hps_str));
        printf("Type %d Estimated Speed: %sH/s\n", type, hps_str);
    }
    printf("\n✅ Estimation complete!\n");
}

int loadRules(const char* rule_file) {
    FILE* fp = fopen(rule_file, "r");
    if (!fp) return 0;

    char line[MAX_RULE_LINE];
    rule_count = 0;

    while (fgets(line, sizeof(line), fp) && rule_count < MAX_RULES) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        // Parse rule
        int len = strlen(line);
        int i = 0;

        while (i < len && rule_count < MAX_RULES) {
            // Skip whitespace
            while (i < len && (line[i] == ' ' || line[i] == '\t' || line[i] == '\n' || line[i] == '\r')) i++;
            if (i >= len) break;

            rules[rule_count].rule[0] = line[i];
            rules[rule_count].rule[1] = '\0';

            // Check if rule has parameter
            if (i + 1 < len && (line[i+1] >= '0' && line[i+1] <= '9')) {
                rules[rule_count].param = line[i+1] - '0';
                i += 2;
            } else if (i + 1 < len && line[i+1] != ' ' && line[i+1] != '\t') {
                rules[rule_count].param = line[i+1];
                i += 2;
            } else {
                rules[rule_count].param = 0;
                i++;
            }

            rule_count++;
        }
    }

    fclose(fp);
    return rule_count;
}

// --- Signal Handler for Graceful Shutdown ---
static volatile int keep_running = 1;
static CheckpointState* g_checkpoint = NULL;
static const char* g_checkpoint_path = "nexo_checkpoint.bin";

void signal_handler(int sig) {
    printf("\n\n⚠️  Interrupted! Saving checkpoint...\n");
    if (g_checkpoint != NULL) {
        saveCheckpoint(g_checkpoint_path, g_checkpoint);
        printf("✅ Checkpoint saved. Resume with mode 4.\n");
    }
    exit(0);
}

int main() {
    char hex_input[128];
    char wordlist_path[256];
    char salt_input[64];
    char resume_choice = 'n';
    int hash_choice, attack_mode, min_len, max_len, limit_choice;
    uint64_t fixed_limit = 0;
    CheckpointState checkpoint = {0};

    printf("\n========================================\n");
    printf("   🚀 NEXO MASTER GPU CRACKER v4.0\n");
    printf("   👤 CREATED BY: PRASHANT\n");
    printf("========================================\n");

    // Register signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    g_checkpoint = &checkpoint;

    printf("\n[0] Select Mode:\n");
    printf("    1. Crack Hash      2. Benchmark\n");
    printf("    3. Hash Rate Estimate  4. Resume from Checkpoint\n");
    printf("    Choice: "); int mode_choice; if(scanf("%d", &mode_choice)) {};

    if (mode_choice == 2) {
        runBenchmark();
        return 0;
    }

    if (mode_choice == 3) {
        estimateHashRate();
        return 0;
    }

    if (mode_choice == 4) {
        resume_choice = 'y';
    } else {
        printf("\n[1] Resume from checkpoint? (y/n): "); if(scanf(" %c", &resume_choice)) {};
    }
    int is_resume = 0;
    if (resume_choice == 'y' || resume_choice == 'Y') {
        if (loadCheckpoint("nexo_checkpoint.bin", &checkpoint)) {
            hash_choice = checkpoint.hash_choice;
            attack_mode = checkpoint.attack_mode;
            min_len = checkpoint.min_len;
            max_len = checkpoint.max_len;
            fixed_limit = checkpoint.fixed_limit;
            strcpy(hex_input, checkpoint.hex_input);
            strcpy(salt_input, checkpoint.salt_input);
            strcpy(wordlist_path, checkpoint.wordlist_path);
            is_resume = 1;
            printf("📋 Resuming from length %d, offset %lu\n", checkpoint.current_len, checkpoint.offset);
        } else {
            printf("❌ No checkpoint found. Starting fresh.\n");
        }
    }

    if (!is_resume) {
        printf("\n[1] Enter Target Hash: "); if(scanf("%127s", hex_input)) {};
        printf("\n[2] Select Hash Type:\n");
        printf("    1. SHA256 (64 hex)  2. SHA256 (32 hex)\n");
        printf("    3. MD5 (32 hex)     4. SHA-1 (40 hex)\n");
        printf("    5. NTLM (32 hex)    6. MySQL41 (40 hex)\n");
        printf("    7. MD5($pass.$salt)  8. SHA256($salt.$pass)\n");
        printf("    9. SHA256($pass.$salt)\n");
        printf("    Choice: "); if(scanf("%d", &hash_choice)) {};
    }

    int target_bytes = 32;
    if (hash_choice == 2 || hash_choice == 3 || hash_choice == 5) target_bytes = 16;
    if (hash_choice == 4 || hash_choice == 6) target_bytes = 20;

    // Handle salt input for salted hash types
    if (hash_choice >= 7 && hash_choice <= 9) {
        printf("\n[3] Enter Salt: "); if(scanf("%63s", salt_input)) {};
        int salt_len = strlen(salt_input);
        cudaMemcpyToSymbol(c_salt, salt_input, salt_len + 1);
        cudaMemcpyToSymbol(c_salt_len, &salt_len, sizeof(int));
    }

    printf("\n[4] Select Attack Mode:\n");
    printf("    1. Brute-Force      2. Dictionary Attack\n");
    printf("    3. Mask Attack      4. Hybrid Attack (Dictionary + Mask)\n");
    printf("    Choice: "); if(scanf("%d", &attack_mode)) {};

    uint8_t h_target[32] = {0};
    for (int i = 0; i < target_bytes; i++) sscanf(hex_input + 2 * i, "%2hhx", &h_target[i]);
    cudaMemcpyToSymbol(c_target, h_target, 32);
    cudaMemcpyToSymbol(c_target_bytes, &target_bytes, sizeof(int));

    // Check potfile for already cracked hash
    char potfile_password[128];
    if (checkPotfile("nexo.potfile", hex_input, potfile_password)) {
        printf("\n✅ Hash already cracked! Password: %s\n", potfile_password);
        return 0;
    }

    int h_found = 0;
    cudaMemcpyToSymbol(d_found, &h_found, sizeof(int));

    if (attack_mode == 3) {
        char mask_pattern[64];
        printf("\n[5] Enter Mask Pattern (e.g., ?l?l?l?d?d): "); if(scanf("%63s", mask_pattern)) {};

        // Custom charsets for ?1, ?2, ?3, ?4
        const char* custom_charsets[4] = {NULL, NULL, NULL, NULL};
        printf("\n[5a] Enter Custom Charsets (optional, press Enter to skip):\n");
        printf("    ?1 charset: "); char cs1[128]; if(scanf(" %127[^\n]", cs1)) {};
        if (strlen(cs1) > 0) custom_charsets[0] = cs1;
        printf("    ?2 charset: "); char cs2[128]; if(scanf(" %127[^\n]", cs2)) {};
        if (strlen(cs2) > 0) custom_charsets[1] = cs2;
        printf("    ?3 charset: "); char cs3[128]; if(scanf(" %127[^\n]", cs3)) {};
        if (strlen(cs3) > 0) custom_charsets[2] = cs3;
        printf("    ?4 charset: "); char cs4[128]; if(scanf(" %127[^\n]", cs4)) {};
        if (strlen(cs4) > 0) custom_charsets[3] = cs4;

        char h_charsets[10][128];
        int h_charset_sizes[10];
        int mask_len = parseMaskPattern(mask_pattern, h_charsets, h_charset_sizes, custom_charsets);

        cudaMemcpyToSymbol(c_mask_pattern, mask_pattern, strlen(mask_pattern) + 1);
        cudaMemcpyToSymbol(c_mask_len, &mask_len, sizeof(int));

        for (int i = 0; i < mask_len; i++) {
            cudaMemcpyToSymbol(c_mask_charsets[i], h_charsets[i], 128);
            cudaMemcpyToSymbol(&c_mask_charset_sizes[i], &h_charset_sizes[i], sizeof(int));
        }

        // Calculate total combinations
        uint64_t total_combinations = 1;
        for (int i = 0; i < mask_len; i++) {
            total_combinations *= h_charset_sizes[i];
        }

        printf("\n🎭 Mask: %s\n", mask_pattern);
        printf("📊 Total combinations: %.2e\n", (double)total_combinations);

        // Auto-tune GPU
        int threads = 256, blocks = 2048;
        autoTuneGPU(0, &blocks, &threads);

        int mask_iterations = 1000;
        uint64_t batch_size = (uint64_t)blocks * threads * mask_iterations;
        uint64_t offset = 0;
        time_t mask_start = time(NULL);
        StatsState mask_stats;
        initStats(&mask_stats, total_combinations, mask_start);

        // Initialize NVML
        bool nvml_available = initNVML();

        while (offset < total_combinations) {
            uint64_t remaining = total_combinations - offset;
            uint64_t current_iterations = (remaining < batch_size) ?
                                          (remaining / (blocks * threads)) : mask_iterations;
            if (current_iterations == 0) current_iterations = 1;

            maskKernelIter<<<blocks, threads>>>(offset, current_iterations, hash_choice);
            KERNEL_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpyFromSymbol(&h_found, d_found, sizeof(int)));

            updateStats(&mask_stats, offset);

            // Get GPU stats if NVML available
            if (nvml_available && offset % (batch_size * 10) == 0) {
                GPUStats gpu_stats = getGPUStats(0);
                printf("\r🌡️ %d°C | ⚡ %uW | 🌀 %u%%  ",
                       gpu_stats.temperature, gpu_stats.power_draw / 1000, gpu_stats.utilization);
                fflush(stdout);
            }

            if (offset % (batch_size * 10) == 0) displayStats(&mask_stats);

            if (h_found) {
                char res[64];
                CUDA_CHECK(cudaMemcpyFromSymbol(res, d_result, 64));
                printf("\n\n🎉 FOUND! Password: %s\n", res);
                addToPotfile("nexo.potfile", hex_input, res);
                if (nvml_available) shutdownNVML();
                return 0;
            }

            offset += batch_size;
        }
        printf("\n❌ Password not found in mask space.\n");
        if (nvml_available) shutdownNVML();
        return 0;
    }

    if (attack_mode == 2) {
        printf("\n[6] Enter Wordlist Path: "); if(scanf("%255s", wordlist_path)) {};

        int apply_rules = 0;
        printf("\n[7] Apply basic mutation rules to wordlist? (1=Yes, 0=No): "); if(scanf("%d", &apply_rules)) {};

        FILE* fp = fopen(wordlist_path, "r");
        if (!fp) {
            printf("\n❌ Error: Cannot open wordlist file: %s\n", wordlist_path);
            return 1;
        }

        struct stat st;
        stat(wordlist_path, &st);
        long file_size = st.st_size;
        const size_t CHUNK_SIZE = 50 * 1024 * 1024; // 50MB chunks
        size_t total_chunks = (file_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        size_t chunk_words_processed = 0;

        printf("\n📚 Wordlist size: %.2f MB (%zu chunks)\n", (double)file_size / (1024 * 1024), total_chunks);
        printf("🔍 Starting Chunked Dictionary Attack...\n");

        // Auto-tune GPU
        int threads = 256, blocks = 2048;
        autoTuneGPU(0, &blocks, &threads);

        time_t dict_start = time(NULL);
        StatsState dict_stats;
        initStats(&dict_stats, file_size, dict_start);

        // Initialize NVML
        bool nvml_available = initNVML();

        // Create CUDA stream for async operations
        cudaStream_t dict_stream;
        CUDA_CHECK(cudaStreamCreate(&dict_stream));

        // Process wordlist in chunks with async memory transfers
        for (size_t chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
            size_t chunk_offset = chunk_idx * CHUNK_SIZE;
            size_t chunk_bytes = (chunk_idx == total_chunks - 1) ?
                               (file_size - chunk_offset) : CHUNK_SIZE;

            // Allocate chunk buffer
            char* chunk_buffer = (char*)malloc(chunk_bytes + 1);
            if (!chunk_buffer) {
                printf("\n❌ Error: Memory allocation failed for chunk\n");
                fclose(fp);
                if (nvml_available) shutdownNVML();
                cudaStreamDestroy(dict_stream);
                return 1;
            }

            // Read chunk
            fseek(fp, chunk_offset, SEEK_SET);
            size_t bytes_read = fread(chunk_buffer, 1, chunk_bytes, fp);
            chunk_buffer[bytes_read] = '\0';

            // Count words in chunk
            int chunk_word_count = 0;
            for (size_t i = 0; i < bytes_read; i++) {
                if (chunk_buffer[i] == '\n') chunk_word_count++;
            }

            if (chunk_word_count == 0) {
                free(chunk_buffer);
                continue;
            }

            // Pre-compute word indices for chunk
            uint32_t* chunk_word_indices = (uint32_t*)malloc((chunk_word_count + 1) * sizeof(uint32_t));
            int current_word = 0;
            chunk_word_indices[0] = 0;
            for (size_t i = 0; i < bytes_read && current_word < chunk_word_count; i++) {
                if (chunk_buffer[i] == '\n') {
                    chunk_word_indices[++current_word] = i + 1;
                }
            }

            // Allocate GPU memory for this chunk
            char* d_chunk_ptr;
            uint32_t* d_chunk_indices_ptr;
            CUDA_CHECK(cudaMalloc(&d_chunk_ptr, chunk_bytes + 1));
            CUDA_CHECK(cudaMalloc(&d_chunk_indices_ptr, (chunk_word_count + 1) * sizeof(uint32_t)));

            // Async copy chunk to GPU (overlaps with previous kernel execution)
            CUDA_CHECK(cudaMemcpyAsync(d_chunk_ptr, chunk_buffer, chunk_bytes + 1, cudaMemcpyHostToDevice, dict_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_chunk_indices_ptr, chunk_word_indices, (chunk_word_count + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, dict_stream));

            // Synchronize to ensure data is ready
            CUDA_CHECK(cudaStreamSynchronize(dict_stream));

            // Set device pointers
            CUDA_CHECK(cudaMemcpyToSymbol(d_wordlist, &d_chunk_ptr, sizeof(char*)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_word_indices, &d_chunk_indices_ptr, sizeof(uint32_t*)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_wordlist_size, (int*)&chunk_bytes, sizeof(int)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_wordlist_count, &chunk_word_count, sizeof(int)));

            // Launch kernel for this chunk
            int blocks = (chunk_word_count + threads - 1) / threads;
            int rules_to_apply = apply_rules ? 5 : 0;
            dictionaryKernel<<<blocks, threads, 0, dict_stream>>>(hash_choice, apply_rules, rules_to_apply);
            KERNEL_CHECK();
            CUDA_CHECK(cudaStreamSynchronize(dict_stream));

            // Check if found
            CUDA_CHECK(cudaMemcpyFromSymbol(&h_found, d_found, sizeof(int)));

            // Cleanup GPU memory for this chunk
            CUDA_CHECK(cudaFree(d_chunk_ptr));
            CUDA_CHECK(cudaFree(d_chunk_indices_ptr));

            free(chunk_buffer);
            free(chunk_word_indices);

            // Update progress
            chunk_words_processed += chunk_word_count;
            updateStats(&dict_stats, chunk_offset + bytes_read);

            // Get GPU stats if NVML available
            if (nvml_available && chunk_idx % 10 == 0) {
                GPUStats gpu_stats = getGPUStats(0);
                printf("\r🌡️ %d°C | ⚡ %uW | 🌀 %u%%  ",
                       gpu_stats.temperature, gpu_stats.power_draw / 1000, gpu_stats.utilization);
                fflush(stdout);
            }

            if (chunk_idx % 10 == 0 || chunk_idx == total_chunks - 1) {
                displayStats(&dict_stats);
            }

            if (h_found) {
                char res[64];
                CUDA_CHECK(cudaMemcpyFromSymbol(res, d_result, 64));
                printf("\n🎉 FOUND! Password: %s\n", res);
                addToPotfile("nexo.potfile", hex_input, res);
                fclose(fp);
                if (nvml_available) shutdownNVML();
                return 0;
            }
        }

        fclose(fp);
        cudaStreamDestroy(dict_stream);
        printf("\n❌ Password not found in wordlist.\n");
        if (nvml_available) shutdownNVML();
        return 0;
    }

    if (attack_mode == 4) {
        // Hybrid Attack (Dictionary + Mask)
        printf("\n[6] Enter Wordlist Path: "); if(scanf("%255s", wordlist_path)) {};

        FILE* fp = fopen(wordlist_path, "r");
        if (!fp) {
            printf("\n❌ Error: Cannot open wordlist file: %s\n", wordlist_path);
            return 1;
        }

        char mask_pattern[64];
        printf("\n[7] Enter Mask Pattern (e.g., ?d?d): "); if(scanf("%63s", mask_pattern)) {};

        // Custom charsets for ?1, ?2, ?3, ?4
        const char* custom_charsets[4] = {NULL, NULL, NULL, NULL};
        printf("\n[7a] Enter Custom Charsets (optional, press Enter to skip):\n");
        printf("    ?1 charset: "); char cs1[128]; if(scanf(" %127[^\n]", cs1)) {};
        if (strlen(cs1) > 0) custom_charsets[0] = cs1;
        printf("    ?2 charset: "); char cs2[128]; if(scanf(" %127[^\n]", cs2)) {};
        if (strlen(cs2) > 0) custom_charsets[1] = cs2;
        printf("    ?3 charset: "); char cs3[128]; if(scanf(" %127[^\n]", cs3)) {};
        if (strlen(cs3) > 0) custom_charsets[2] = cs3;
        printf("    ?4 charset: "); char cs4[128]; if(scanf(" %127[^\n]", cs4)) {};
        if (strlen(cs4) > 0) custom_charsets[3] = cs4;

        char h_charsets[10][128];
        int h_charset_sizes[10];
        int mask_len = parseMaskPattern(mask_pattern, h_charsets, h_charset_sizes, custom_charsets);

        cudaMemcpyToSymbol(c_mask_pattern, mask_pattern, strlen(mask_pattern) + 1);
        cudaMemcpyToSymbol(c_mask_len, &mask_len, sizeof(int));

        for (int i = 0; i < mask_len; i++) {
            cudaMemcpyToSymbol(c_mask_charsets[i], h_charsets[i], 128);
            cudaMemcpyToSymbol(&c_mask_charset_sizes[i], &h_charset_sizes[i], sizeof(int));
        }

        // Calculate total mask combinations
        uint64_t mask_combinations = 1;
        for (int i = 0; i < mask_len; i++) {
            mask_combinations *= h_charset_sizes[i];
        }

        printf("\n🎭 Mask: %s\n", mask_pattern);
        printf("📊 Mask combinations: %.2e\n", (double)mask_combinations);

        // Get file size for stats
        struct stat st;
        stat(wordlist_path, &st);
        long file_size = st.st_size;

        // Auto-tune GPU
        int threads = 256, blocks = 2048;
        autoTuneGPU(0, &blocks, &threads);

        int mask_iterations = 1000;
        uint64_t mask_batch_size = (uint64_t)blocks * threads * mask_iterations;
        const size_t CHUNK_SIZE = 50 * 1024 * 1024; // 50MB chunks
        size_t total_chunks = (file_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

        printf("\n📚 Wordlist size: %.2f MB (%zu chunks)\n", (double)file_size / (1024 * 1024), total_chunks);
        printf("🔍 Starting Hybrid Attack (Dictionary + Mask)...\n");

        time_t hybrid_start = time(NULL);
        StatsState hybrid_stats;
        initStats(&hybrid_stats, file_size, hybrid_start);

        // Initialize NVML
        bool nvml_available = initNVML();

        // Process wordlist in chunks
        for (size_t chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
            size_t chunk_offset = chunk_idx * CHUNK_SIZE;
            size_t chunk_bytes = (chunk_idx == total_chunks - 1) ?
                               (file_size - chunk_offset) : CHUNK_SIZE;

            // Allocate chunk buffer
            char* chunk_buffer = (char*)malloc(chunk_bytes + 1);
            if (!chunk_buffer) {
                printf("\n❌ Error: Memory allocation failed for chunk\n");
                fclose(fp);
                if (nvml_available) shutdownNVML();
                return 1;
            }

            // Read chunk
            fseek(fp, chunk_offset, SEEK_SET);
            size_t bytes_read = fread(chunk_buffer, 1, chunk_bytes, fp);
            chunk_buffer[bytes_read] = '\0';

            // Count words in chunk
            int chunk_word_count = 0;
            for (size_t i = 0; i < bytes_read; i++) {
                if (chunk_buffer[i] == '\n') chunk_word_count++;
            }

            if (chunk_word_count == 0) {
                free(chunk_buffer);
                continue;
            }

            // Pre-compute word indices for chunk
            uint32_t* chunk_word_indices = (uint32_t*)malloc((chunk_word_count + 1) * sizeof(uint32_t));
            int current_word = 0;
            chunk_word_indices[0] = 0;
            for (size_t i = 0; i < bytes_read && current_word < chunk_word_count; i++) {
                if (chunk_buffer[i] == '\n') {
                    chunk_word_indices[++current_word] = i + 1;
                }
            }

            // Allocate GPU memory for this chunk
            char* d_chunk_ptr;
            uint32_t* d_chunk_indices_ptr;
            CUDA_CHECK(cudaMalloc(&d_chunk_ptr, chunk_bytes + 1));
            CUDA_CHECK(cudaMalloc(&d_chunk_indices_ptr, (chunk_word_count + 1) * sizeof(uint32_t)));

            // Copy chunk to GPU
            CUDA_CHECK(cudaMemcpy(d_chunk_ptr, chunk_buffer, chunk_bytes + 1, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_chunk_indices_ptr, chunk_word_indices, (chunk_word_count + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

            // Set device pointers
            CUDA_CHECK(cudaMemcpyToSymbol(d_wordlist, &d_chunk_ptr, sizeof(char*)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_word_indices, &d_chunk_indices_ptr, sizeof(uint32_t*)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_wordlist_size, (int*)&chunk_bytes, sizeof(int)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_wordlist_count, &chunk_word_count, sizeof(int)));

            // Process each word with mask combinations
            for (int word_idx = 0; word_idx < chunk_word_count; word_idx++) {
                if (h_found) break;

                uint64_t mask_offset = 0;
                while (mask_offset < mask_combinations) {
                    uint64_t remaining_mask = mask_combinations - mask_offset;
                    uint64_t current_mask_iterations = (remaining_mask < mask_batch_size) ?
                                                       (remaining_mask / (blocks * threads)) : mask_iterations;
                    if (current_mask_iterations == 0) current_mask_iterations = 1;

                    hybridKernel<<<blocks, threads>>>(word_idx, mask_offset, current_mask_iterations, hash_choice);
                    KERNEL_CHECK();
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaMemcpyFromSymbol(&h_found, d_found, sizeof(int)));

                    if (h_found) {
                        char res[128];
                        CUDA_CHECK(cudaMemcpyFromSymbol(res, d_result, 128));
                        printf("\n🎉 FOUND! Password: %s\n", res);
                        addToPotfile("nexo.potfile", hex_input, res);

                        // Cleanup
                        CUDA_CHECK(cudaFree(d_chunk_ptr));
                        CUDA_CHECK(cudaFree(d_chunk_indices_ptr));
                        free(chunk_buffer);
                        free(chunk_word_indices);
                        fclose(fp);
                        if (nvml_available) shutdownNVML();
                        return 0;
                    }

                    mask_offset += mask_batch_size;
                }
            }

            // Cleanup GPU memory for this chunk
            CUDA_CHECK(cudaFree(d_chunk_ptr));
            CUDA_CHECK(cudaFree(d_chunk_indices_ptr));

            free(chunk_buffer);
            free(chunk_word_indices);

            // Update progress
            updateStats(&hybrid_stats, chunk_offset + bytes_read);

            // Get GPU stats if NVML available
            if (nvml_available && chunk_idx % 5 == 0) {
                GPUStats gpu_stats = getGPUStats(0);
                printf("\r🌡️ %d°C | ⚡ %uW | 🌀 %u%%  ",
                       gpu_stats.temperature, gpu_stats.power_draw / 1000, gpu_stats.utilization);
                fflush(stdout);
            }

            if (chunk_idx % 5 == 0 || chunk_idx == total_chunks - 1) {
                displayStats(&hybrid_stats);
            }
        }

        fclose(fp);
        printf("\n❌ Password not found in hybrid attack.\n");
        if (nvml_available) shutdownNVML();
        return 0;
    } else {
        printf("\n[6] Enter Length Range (min max): "); if(scanf("%d %d", &min_len, &max_len)) {};
        printf("\n[7] Select Run Mode (1: 12h, 2: Fixed B): "); if(scanf("%d", &limit_choice)) {};
        if(limit_choice == 2) { printf("    Enter Billions: "); double b; if(scanf("%lf", &b)) {}; fixed_limit = (uint64_t)(b * 1000000000ULL); }

        const char* h_charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%&*";
        int h_charset_len = strlen(h_charset);
        cudaMemcpyToSymbol(c_charset, h_charset, h_charset_len + 1);
        cudaMemcpyToSymbol(c_charset_len, &h_charset_len, sizeof(int));

        // Auto-tune GPU for each device
        int threads = 256, blocks = 2048, iterations = 5000;
        int device_count;
        cudaGetDeviceCount(&device_count);
        printf("\n🖥️  Detected %d GPU(s)\n", device_count);

        if (device_count > 1) {
            printf("⚡ Using Multi-GPU mode with load balancing\n");
        }

        // Auto-tune based on primary GPU (assumes homogeneous GPUs for balanced stride)
        autoTuneGPU(0, &blocks, &threads);

        uint64_t batch_size = (uint64_t)blocks * threads * iterations;
        time_t wall_start = time(NULL); uint64_t total_scanned = 0;
        time_t last_checkpoint = time(NULL);
        uint64_t total_to_scan = 0;
        for (int l = min_len; l <= max_len; l++) {
            uint64_t comb = 1; for(int i=0; i<l; i++) comb *= h_charset_len;
            total_to_scan += comb;
        }
        StatsState stats;
        initStats(&stats, total_to_scan, wall_start);

        // Initialize NVML
        bool nvml_available = initNVML();

        // Create CUDA streams for each GPU (asynchronous multi-GPU)
        cudaStream_t* streams = (cudaStream_t*)malloc(device_count * sizeof(cudaStream_t));
        for (int dev = 0; dev < device_count; dev++) {
            cudaSetDevice(dev);
            CUDA_CHECK(cudaStreamCreate(&streams[dev]));
        }

        int start_len = is_resume ? checkpoint.current_len : min_len;
        uint64_t start_offset = is_resume ? checkpoint.offset : 0;
        uint64_t start_total = is_resume ? checkpoint.total_scanned : 0;

        for (int len = start_len; len <= max_len; len++) {
            uint64_t max_idx = 1; for(int i=0; i<len; i++) max_idx *= h_charset_len;
            printf("\n--- Length %d | Total: %.2f T ---\n", len, (double)max_idx/1e12);
            uint64_t offset = (len == start_len) ? start_offset : 0;
            total_scanned = (len == start_len) ? start_total : total_scanned;

            while (offset < max_idx) {
                if (difftime(time(NULL), wall_start) > 43200 || (fixed_limit > 0 && total_scanned >= fixed_limit)) break;

                // Distribute workload across GPUs with asynchronous streams
                uint64_t gpu_stride = batch_size * device_count;
                for (int dev = 0; dev < device_count; dev++) {
                    cudaSetDevice(dev);
                    uint64_t dev_offset = offset + (dev * batch_size);

                    if (dev_offset >= max_idx) break;

                    uint64_t remaining = max_idx - dev_offset;
                    uint64_t dev_iterations = (remaining < batch_size) ?
                                          (remaining / (blocks * threads)) : iterations;
                    if (dev_iterations == 0) dev_iterations = 1;

                    // Use specialized kernels with asynchronous streams
                    switch(hash_choice) {
                        case 1: case 2: crackSHA256<<<blocks, threads, 0, streams[dev]>>>(dev_offset, len, dev_iterations); break;
                        case 3: crackMD5<<<blocks, threads, 0, streams[dev]>>>(dev_offset, len, dev_iterations); break;
                        case 4: crackSHA1<<<blocks, threads, 0, streams[dev]>>>(dev_offset, len, dev_iterations); break;
                        case 5: crackNTLM<<<blocks, threads, 0, streams[dev]>>>(dev_offset, len, dev_iterations); break;
                        case 6: crackMySQL41<<<blocks, threads, 0, streams[dev]>>>(dev_offset, len, dev_iterations); break;
                        default: crackKernel<<<blocks, threads, 0, streams[dev]>>>(dev_offset, len, dev_iterations, hash_choice); break;
                    }
                    KERNEL_CHECK();
                }

                // Synchronize all GPU streams
                for (int dev = 0; dev < device_count; dev++) {
                    cudaSetDevice(dev);
                    CUDA_CHECK(cudaStreamSynchronize(streams[dev]));
                    cudaMemcpyFromSymbol(&h_found, d_found, sizeof(int));
                    if (h_found) {
                        char res[32];
                        cudaMemcpyFromSymbol(res, d_result, 32);
                        printf("\n\n🎉 FOUND! Password: %s (GPU %d)\n", res, dev);
                        addToPotfile("nexo.potfile", hex_input, res);

                        // Cleanup streams
                        for (int d = 0; d < device_count; d++) {
                            cudaSetDevice(d);
                            cudaStreamDestroy(streams[d]);
                        }
                        free(streams);
                        if (nvml_available) shutdownNVML();
                        return 0;
                    }
                }

                // Update and display stats
                updateStats(&stats, total_scanned);

                // Get GPU stats if NVML available
                if (nvml_available && offset % (batch_size * 5) == 0) {
                    GPUStats gpu_stats = getGPUStats(0);
                    printf("\r🌡️ %d°C | ⚡ %uW | 🌀 %u%%  ",
                           gpu_stats.temperature, gpu_stats.power_draw / 1000, gpu_stats.utilization);
                    fflush(stdout);
                }

                if (offset % (batch_size * 5) == 0) displayStats(&stats);

                // Save checkpoint every 5 minutes
                if (difftime(time(NULL), last_checkpoint) > 300) {
                    checkpoint.hash_choice = hash_choice;
                    checkpoint.attack_mode = attack_mode;
                    checkpoint.min_len = min_len;
                    checkpoint.max_len = max_len;
                    checkpoint.current_len = len;
                    checkpoint.offset = offset;
                    checkpoint.total_scanned = total_scanned;
                    checkpoint.fixed_limit = fixed_limit;
                    checkpoint.start_time = wall_start;
                    strcpy(checkpoint.hex_input, hex_input);
                    strcpy(checkpoint.salt_input, salt_input);
                    strcpy(checkpoint.wordlist_path, wordlist_path);
                    checkpoint.salt_len = strlen(salt_input);
                    saveCheckpoint("nexo_checkpoint.bin", &checkpoint);
                    last_checkpoint = time(NULL);
                }

                offset += gpu_stride; total_scanned += gpu_stride;
            }
        }

        // Cleanup streams
        for (int dev = 0; dev < device_count; dev++) {
            cudaSetDevice(dev);
            cudaStreamDestroy(streams[dev]);
        }
        free(streams);

        printf("\n❌ Not found.\n");
        if (nvml_available) shutdownNVML();
        return 0;
    }
}
