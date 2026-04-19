
+++ NEXO_MASTER_GPU_CRACKER.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "\n❌ CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define DEFAULT_CHUNK_SIZE (50 * 1024 * 1024)

// --- Constant Memory ---
__constant__ uint8_t c_target[32];
__constant__ char c_charset[70];
__constant__ char c_salt[64]; // Salt for salted hashes
__constant__ char c_mask_pattern[64]; // Mask pattern (e.g., "?l?l?d?d")
__constant__ char c_mask_charsets[10][128]; // Character sets for each mask position
__constant__ int c_charset_len;
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

__device__ const uint32_t K256[64] = {
    0x428a2f98,0x71374498,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ void sha256_transform(uint32_t *state, const uint8_t *chunk) {
    uint32_t W[64], a,b,c,d,e,f,g,h,i,T1,T2;
    for(i=0;i<16;i++) W[i] = (chunk[i*4]<<24)|(chunk[i*4+1]<<16)|(chunk[i*4+2]<<8)|chunk[i*4+3];
    for(i=16;i<64;i++) W[i] = sig1(W[i-2]) + W[i-7] + sig0(W[i-15]) + W[i-16];
    a=state[0];b=state[1];c=state[2];d=state[3];e=state[4];f=state[5];g=state[6];h=state[7];
    for(i=0;i<64;i++){
        T1 = h + ep1(e) + ch(e,f,g) + K256[i] + W[i];
        T2 = ep0(a) + maj(a,b,c);
        h=g;g=f;f=e;e=d+T1;d=c;c=b;b=a;a=T1+T2;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

__device__ void sha256_hash(const char *input, int len, uint8_t *output) {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    int offset = 0;

    // Process all full 64-byte chunks first.
    while (len - offset >= 64) {
        sha256_transform(h, (const uint8_t*)(input + offset));
        offset += 64;
    }

    // FIPS 180-4 compliant padding for remaining bytes.
    uint8_t final_blocks[128] = {0};
    int remaining = len - offset;
    for (int i = 0; i < remaining; i++) {
        final_blocks[i] = (uint8_t)input[offset + i];
    }
    final_blocks[remaining] = 0x80;

    int final_block_count = (remaining <= 55) ? 1 : 2;
    uint64_t bits = (uint64_t)len * 8;
    int len_pos = final_block_count * 64 - 8;
    for (int i = 0; i < 8; i++) {
        final_blocks[len_pos + 7 - i] = (uint8_t)((bits >> (i * 8)) & 0xFF);
    }

    sha256_transform(h, final_blocks);
    if (final_block_count == 2) sha256_transform(h, final_blocks + 64);

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

// --- MD4 Engine for NTLM ---
__device__ void md4_hash(const char *input, int len, uint8_t *output) {
    uint32_t a = 0x67452301, b = 0xefcdab89, c = 0x98badcfe, d = 0x10325476;
    uint32_t W[16] = {0};

    // Copy input and add padding
    for(int i=0; i<len; i++) ((uint8_t*)W)[i] = (uint8_t)input[i];
    ((uint8_t*)W)[len] = 0x80;

    // Pad to 56 bytes (448 bits), then add 64-bit length
    if (len < 56) {
        for(int i=len+1; i<56; i++) ((uint8_t*)W)[i] = 0;
    } else {
        // Need two blocks - process first block then second
        for(int i=len+1; i<64; i++) ((uint8_t*)W)[i] = 0;

        // Save original state
        uint32_t AA = a, BB = b, CC = c, DD = d;

        // Round 1
        #define MD4_F(x,y,z) (((x) & (y)) | ((~x) & (z)))
        #define MD4_ROTL(x,n) (((x) << (n)) | ((x) >> (32-(n))))

        a = AA; b = BB; c = CC; d = DD;
        a = MD4_ROTL(a + MD4_F(b,c,d) + W[0], 3);
        d = MD4_ROTL(d + MD4_F(a,b,c) + W[1], 7);
        c = MD4_ROTL(c + MD4_F(d,a,b) + W[2], 11);
        b = MD4_ROTL(b + MD4_F(c,d,a) + W[3], 19);
        a = MD4_ROTL(a + MD4_F(b,c,d) + W[4], 3);
        d = MD4_ROTL(d + MD4_F(a,b,c) + W[5], 7);
        c = MD4_ROTL(c + MD4_F(d,a,b) + W[6], 11);
        b = MD4_ROTL(b + MD4_F(c,d,a) + W[7], 19);
        a = MD4_ROTL(a + MD4_F(b,c,d) + W[8], 3);
        d = MD4_ROTL(d + MD4_F(a,b,c) + W[9], 7);
        c = MD4_ROTL(c + MD4_F(d,a,b) + W[10], 11);
        b = MD4_ROTL(b + MD4_F(c,d,a) + W[11], 19);
        a = MD4_ROTL(a + MD4_F(b,c,d) + W[12], 3);
        d = MD4_ROTL(d + MD4_F(a,b,c) + W[13], 7);
        c = MD4_ROTL(c + MD4_F(d,a,b) + W[14], 11);
        b = MD4_ROTL(b + MD4_F(c,d,a) + W[15], 19);

        // Round 2
        #define MD4_G(x,y,z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
        a = MD4_ROTL(a + MD4_G(b,c,d) + W[0] + 0x5a827999, 3);
        d = MD4_ROTL(d + MD4_G(a,b,c) + W[4] + 0x5a827999, 5);
        c = MD4_ROTL(c + MD4_G(d,a,b) + W[8] + 0x5a827999, 9);
        b = MD4_ROTL(b + MD4_G(c,d,a) + W[12] + 0x5a827999, 13);
        a = MD4_ROTL(a + MD4_G(b,c,d) + W[1] + 0x5a827999, 3);
        d = MD4_ROTL(d + MD4_G(a,b,c) + W[5] + 0x5a827999, 5);
        c = MD4_ROTL(c + MD4_G(d,a,b) + W[9] + 0x5a827999, 9);
        b = MD4_ROTL(b + MD4_G(c,d,a) + W[13] + 0x5a827999, 13);
        a = MD4_ROTL(a + MD4_G(b,c,d) + W[2] + 0x5a827999, 3);
        d = MD4_ROTL(d + MD4_G(a,b,c) + W[6] + 0x5a827999, 5);
        c = MD4_ROTL(c + MD4_G(d,a,b) + W[10] + 0x5a827999, 9);
        b = MD4_ROTL(b + MD4_G(c,d,a) + W[14] + 0x5a827999, 13);
        a = MD4_ROTL(a + MD4_G(b,c,d) + W[3] + 0x5a827999, 3);
        d = MD4_ROTL(d + MD4_G(a,b,c) + W[7] + 0x5a827999, 5);
        c = MD4_ROTL(c + MD4_G(d,a,b) + W[11] + 0x5a827999, 9);
        b = MD4_ROTL(b + MD4_G(c,d,a) + W[15] + 0x5a827999, 13);

        // Round 3
        #define MD4_H(x,y,z) ((x) ^ (y) ^ (z))
        a = MD4_ROTL(a + MD4_H(b,c,d) + W[0] + 0x6ed9eba1, 3);
        d = MD4_ROTL(d + MD4_H(a,b,c) + W[8] + 0x6ed9eba1, 9);
        c = MD4_ROTL(c + MD4_H(d,a,b) + W[4] + 0x6ed9eba1, 11);
        b = MD4_ROTL(b + MD4_H(c,d,a) + W[12] + 0x6ed9eba1, 15);
        a = MD4_ROTL(a + MD4_H(b,c,d) + W[2] + 0x6ed9eba1, 3);
        d = MD4_ROTL(d + MD4_H(a,b,c) + W[10] + 0x6ed9eba1, 9);
        c = MD4_ROTL(c + MD4_H(d,a,b) + W[6] + 0x6ed9eba1, 11);
        b = MD4_ROTL(b + MD4_H(c,d,a) + W[14] + 0x6ed9eba1, 15);
        a = MD4_ROTL(a + MD4_H(b,c,d) + W[1] + 0x6ed9eba1, 3);
        d = MD4_ROTL(d + MD4_H(a,b,c) + W[9] + 0x6ed9eba1, 9);
        c = MD4_ROTL(c + MD4_H(d,a,b) + W[5] + 0x6ed9eba1, 11);
        b = MD4_ROTL(b + MD4_H(c,d,a) + W[13] + 0x6ed9eba1, 15);
        a = MD4_ROTL(a + MD4_H(b,c,d) + W[3] + 0x6ed9eba1, 3);
        d = MD4_ROTL(d + MD4_H(a,b,c) + W[11] + 0x6ed9eba1, 9);
        c = MD4_ROTL(c + MD4_H(d,a,b) + W[7] + 0x6ed9eba1, 11);
        b = MD4_ROTL(b + MD4_H(c,d,a) + W[15] + 0x6ed9eba1, 15);

        AA += a; BB += b; CC += c; DD += d;

        // Second block - clear and set up
        for(int i=0; i<16; i++) W[i] = 0;
        int remaining = len - 55;
        for(int i=0; i<remaining; i++) ((uint8_t*)W)[i] = ((uint8_t*)input)[55 + i];
        ((uint8_t*)W)[remaining] = 0x80;
        ((uint32_t*)W)[14] = (uint32_t)(len * 8);

        // Process second block with same rounds
        a = AA; b = BB; c = CC; d = DD;
        a = MD4_ROTL(a + MD4_F(b,c,d) + W[0], 3);
        d = MD4_ROTL(d + MD4_F(a,b,c) + W[1], 7);
        c = MD4_ROTL(c + MD4_F(d,a,b) + W[2], 11);
        b = MD4_ROTL(b + MD4_F(c,d,a) + W[3], 19);
        a = MD4_ROTL(a + MD4_F(b,c,d) + W[4], 3);
        d = MD4_ROTL(d + MD4_F(a,b,c) + W[5], 7);
        c = MD4_ROTL(c + MD4_F(d,a,b) + W[6], 11);
        b = MD4_ROTL(b + MD4_F(c,d,a) + W[7], 19);
        a = MD4_ROTL(a + MD4_F(b,c,d) + W[8], 3);
        d = MD4_ROTL(d + MD4_F(a,b,c) + W[9], 7);
        c = MD4_ROTL(c + MD4_F(d,a,b) + W[10], 11);
        b = MD4_ROTL(b + MD4_F(c,d,a) + W[11], 19);
        a = MD4_ROTL(a + MD4_F(b,c,d) + W[12], 3);
        d = MD4_ROTL(d + MD4_F(a,b,c) + W[13], 7);
        c = MD4_ROTL(c + MD4_F(d,a,b) + W[14], 11);
        b = MD4_ROTL(b + MD4_F(c,d,a) + W[15], 19);

        a = MD4_ROTL(a + MD4_G(b,c,d) + W[0] + 0x5a827999, 3);
        d = MD4_ROTL(d + MD4_G(a,b,c) + W[4] + 0x5a827999, 5);
        c = MD4_ROTL(c + MD4_G(d,a,b) + W[8] + 0x5a827999, 9);
        b = MD4_ROTL(b + MD4_G(c,d,a) + W[12] + 0x5a827999, 13);
        a = MD4_ROTL(a + MD4_G(b,c,d) + W[1] + 0x5a827999, 3);
        d = MD4_ROTL(d + MD4_G(a,b,c) + W[5] + 0x5a827999, 5);
        c = MD4_ROTL(c + MD4_G(d,a,b) + W[9] + 0x5a827999, 9);
        b = MD4_ROTL(b + MD4_G(c,d,a) + W[13] + 0x5a827999, 13);
        a = MD4_ROTL(a + MD4_G(b,c,d) + W[2] + 0x5a827999, 3);
        d = MD4_ROTL(d + MD4_G(a,b,c) + W[6] + 0x5a827999, 5);
        c = MD4_ROTL(c + MD4_G(d,a,b) + W[10] + 0x5a827999, 9);
        b = MD4_ROTL(b + MD4_G(c,d,a) + W[14] + 0x5a827999, 13);
        a = MD4_ROTL(a + MD4_G(b,c,d) + W[3] + 0x5a827999, 3);
        d = MD4_ROTL(d + MD4_G(a,b,c) + W[7] + 0x5a827999, 5);
        c = MD4_ROTL(c + MD4_G(d,a,b) + W[11] + 0x5a827999, 9);
        b = MD4_ROTL(b + MD4_G(c,d,a) + W[15] + 0x5a827999, 13);

        a = MD4_ROTL(a + MD4_H(b,c,d) + W[0] + 0x6ed9eba1, 3);
        d = MD4_ROTL(d + MD4_H(a,b,c) + W[8] + 0x6ed9eba1, 9);
        c = MD4_ROTL(c + MD4_H(d,a,b) + W[4] + 0x6ed9eba1, 11);
        b = MD4_ROTL(b + MD4_H(c,d,a) + W[12] + 0x6ed9eba1, 15);
        a = MD4_ROTL(a + MD4_H(b,c,d) + W[2] + 0x6ed9eba1, 3);
        d = MD4_ROTL(d + MD4_H(a,b,c) + W[10] + 0x6ed9eba1, 9);
        c = MD4_ROTL(c + MD4_H(d,a,b) + W[6] + 0x6ed9eba1, 11);
        b = MD4_ROTL(b + MD4_H(c,d,a) + W[14] + 0x6ed9eba1, 15);
        a = MD4_ROTL(a + MD4_H(b,c,d) + W[1] + 0x6ed9eba1, 3);
        d = MD4_ROTL(d + MD4_H(a,b,c) + W[9] + 0x6ed9eba1, 9);
        c = MD4_ROTL(c + MD4_H(d,a,b) + W[5] + 0x6ed9eba1, 11);
        b = MD4_ROTL(b + MD4_H(c,d,a) + W[13] + 0x6ed9eba1, 15);
        a = MD4_ROTL(a + MD4_H(b,c,d) + W[3] + 0x6ed9eba1, 3);
        d = MD4_ROTL(d + MD4_H(a,b,c) + W[11] + 0x6ed9eba1, 9);
        c = MD4_ROTL(c + MD4_H(d,a,b) + W[7] + 0x6ed9eba1, 11);
        b = MD4_ROTL(b + MD4_H(c,d,a) + W[15] + 0x6ed9eba1, 15);

        AA += a; BB += b; CC += c; DD += d;
        a = AA; b = BB; c = CC; d = DD;
    }

    // Round 1
    #define MD4_F(x,y,z) (((x) & (y)) | ((~x) & (z)))
    #define MD4_ROTL(x,n) (((x) << (n)) | ((x) >> (32-(n))))

    a = MD4_ROTL(a + MD4_F(b,c,d) + W[0], 3);
    d = MD4_ROTL(d + MD4_F(a,b,c) + W[1], 7);
    c = MD4_ROTL(c + MD4_F(d,a,b) + W[2], 11);
    b = MD4_ROTL(b + MD4_F(c,d,a) + W[3], 19);
    a = MD4_ROTL(a + MD4_F(b,c,d) + W[4], 3);
    d = MD4_ROTL(d + MD4_F(a,b,c) + W[5], 7);
    c = MD4_ROTL(c + MD4_F(d,a,b) + W[6], 11);
    b = MD4_ROTL(b + MD4_F(c,d,a) + W[7], 19);
    a = MD4_ROTL(a + MD4_F(b,c,d) + W[8], 3);
    d = MD4_ROTL(d + MD4_F(a,b,c) + W[9], 7);
    c = MD4_ROTL(c + MD4_F(d,a,b) + W[10], 11);
    b = MD4_ROTL(b + MD4_F(c,d,a) + W[11], 19);
    a = MD4_ROTL(a + MD4_F(b,c,d) + W[12], 3);
    d = MD4_ROTL(d + MD4_F(a,b,c) + W[13], 7);
    c = MD4_ROTL(c + MD4_F(d,a,b) + W[14], 11);
    b = MD4_ROTL(b + MD4_F(c,d,a) + W[15], 19);

    // Round 2
    #define MD4_G(x,y,z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
    a = MD4_ROTL(a + MD4_G(b,c,d) + W[0] + 0x5a827999, 3);
    d = MD4_ROTL(d + MD4_G(a,b,c) + W[4] + 0x5a827999, 5);
    c = MD4_ROTL(c + MD4_G(d,a,b) + W[8] + 0x5a827999, 9);
    b = MD4_ROTL(b + MD4_G(c,d,a) + W[12] + 0x5a827999, 13);
    a = MD4_ROTL(a + MD4_G(b,c,d) + W[1] + 0x5a827999, 3);
    d = MD4_ROTL(d + MD4_G(a,b,c) + W[5] + 0x5a827999, 5);
    c = MD4_ROTL(c + MD4_G(d,a,b) + W[9] + 0x5a827999, 9);
    b = MD4_ROTL(b + MD4_G(c,d,a) + W[13] + 0x5a827999, 13);
    a = MD4_ROTL(a + MD4_G(b,c,d) + W[2] + 0x5a827999, 3);
    d = MD4_ROTL(d + MD4_G(a,b,c) + W[6] + 0x5a827999, 5);
    c = MD4_ROTL(c + MD4_G(d,a,b) + W[10] + 0x5a827999, 9);
    b = MD4_ROTL(b + MD4_G(c,d,a) + W[14] + 0x5a827999, 13);
    a = MD4_ROTL(a + MD4_G(b,c,d) + W[3] + 0x5a827999, 3);
    d = MD4_ROTL(d + MD4_G(a,b,c) + W[7] + 0x5a827999, 5);
    c = MD4_ROTL(c + MD4_G(d,a,b) + W[11] + 0x5a827999, 9);
    b = MD4_ROTL(b + MD4_G(c,d,a) + W[15] + 0x5a827999, 13);

    // Round 3
    #define MD4_H(x,y,z) ((x) ^ (y) ^ (z))
    a = MD4_ROTL(a + MD4_H(b,c,d) + W[0] + 0x6ed9eba1, 3);
    d = MD4_ROTL(d + MD4_H(a,b,c) + W[8] + 0x6ed9eba1, 9);
    c = MD4_ROTL(c + MD4_H(d,a,b) + W[4] + 0x6ed9eba1, 11);
    b = MD4_ROTL(b + MD4_H(c,d,a) + W[12] + 0x6ed9eba1, 15);
    a = MD4_ROTL(a + MD4_H(b,c,d) + W[2] + 0x6ed9eba1, 3);
    d = MD4_ROTL(d + MD4_H(a,b,c) + W[10] + 0x6ed9eba1, 9);
    c = MD4_ROTL(c + MD4_H(d,a,b) + W[6] + 0x6ed9eba1, 11);
    b = MD4_ROTL(b + MD4_H(c,d,a) + W[14] + 0x6ed9eba1, 15);
    a = MD4_ROTL(a + MD4_H(b,c,d) + W[1] + 0x6ed9eba1, 3);
    d = MD4_ROTL(d + MD4_H(a,b,c) + W[9] + 0x6ed9eba1, 9);
    c = MD4_ROTL(c + MD4_H(d,a,b) + W[5] + 0x6ed9eba1, 11);
    b = MD4_ROTL(b + MD4_H(c,d,a) + W[13] + 0x6ed9eba1, 15);
    a = MD4_ROTL(a + MD4_H(b,c,d) + W[3] + 0x6ed9eba1, 3);
    d = MD4_ROTL(d + MD4_H(a,b,c) + W[11] + 0x6ed9eba1, 9);
    c = MD4_ROTL(c + MD4_H(d,a,b) + W[7] + 0x6ed9eba1, 11);
    b = MD4_ROTL(b + MD4_H(c,d,a) + W[15] + 0x6ed9eba1, 15);

    uint32_t* out32 = (uint32_t*)output;
    out32[0] = a; out32[1] = b; out32[2] = c; out32[3] = d;
}

// --- NTLM Engine (uses MD4) ---
__device__ void ntlm_hash(const char *input, int len, uint8_t *output) {
    uint8_t unicode[256] = {0};
    for(int i=0; i<len && i<128; i++) { unicode[i*2] = input[i]; unicode[i*2+1] = 0; }
    md4_hash((char*)unicode, len*2, output);
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

// --- Global Result State (per-device for multi-GPU) ---
__device__ int d_found = 0;
__device__ char d_result[32];

// Host-side flag for multi-GPU synchronization
static int h_found_flag = 0;
static volatile sig_atomic_t g_stop_requested = 0;

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

void saveCheckpoint(const char* filename, CheckpointState* state) {
    FILE* fp = fopen(filename, "wb");
    if (fp) {
        fwrite(state, sizeof(CheckpointState), 1, fp);
        fclose(fp);
        printf("\n💾 Checkpoint saved to %s\n", filename);
    }
}

void handleSigint(int signo) {
    (void)signo;
    g_stop_requested = 1;
}

void updateCheckpointState(
    CheckpointState* checkpoint,
    int hash_choice,
    int attack_mode,
    int min_len,
    int max_len,
    int current_len,
    uint64_t offset,
    uint64_t total_scanned,
    uint64_t fixed_limit,
    time_t wall_start,
    const char* hex_input,
    const char* salt_input,
    const char* wordlist_path
) {
    checkpoint->hash_choice = hash_choice;
    checkpoint->attack_mode = attack_mode;
    checkpoint->min_len = min_len;
    checkpoint->max_len = max_len;
    checkpoint->current_len = current_len;
    checkpoint->offset = offset;
    checkpoint->total_scanned = total_scanned;
    checkpoint->fixed_limit = fixed_limit;
    checkpoint->start_time = wall_start;
    strcpy(checkpoint->hex_input, hex_input);
    strcpy(checkpoint->salt_input, salt_input);
    strcpy(checkpoint->wordlist_path, wordlist_path);
    checkpoint->salt_len = strlen(salt_input);
}

int loadCheckpoint(const char* filename, CheckpointState* state) {
    FILE* fp = fopen(filename, "rb");
    if (fp) {
        fread(state, sizeof(CheckpointState), 1, fp);
        fclose(fp);
        printf("\n📂 Checkpoint loaded from %s\n", filename);
        return 1;
    }
    return 0;
}

int parseDeviceList(const char* input, int max_devices, int* out_devices, int* out_count) {
    if (!input || !out_devices || !out_count) return 0;
    if (strcmp(input, "-1") == 0) {
        *out_count = 0;
        return 1;
    }

    char temp[128];
    strncpy(temp, input, sizeof(temp) - 1);
    temp[sizeof(temp) - 1] = '\0';

    int count = 0;
    char* token = strtok(temp, ",");
    while (token) {
        int dev = atoi(token);
        if (dev < 0 || dev >= max_devices) return 0;

        int duplicate = 0;
        for (int i = 0; i < count; i++) {
            if (out_devices[i] == dev) { duplicate = 1; break; }
        }
        if (!duplicate) out_devices[count++] = dev;
        token = strtok(NULL, ",");
    }
    *out_count = count;
    return count > 0;
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
        int mins = (int)((seconds / 60) % 60);
        snprintf(buffer, buffer_size, "%dh %dm", hours, mins);
    } else {
        int days = (int)(seconds / 86400);
        int hours = (int)((seconds / 3600) % 24);
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

int parseMaskPattern(const char* pattern, char charsets[10][128], int charset_sizes[10]) {
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

__global__ void crackKernel(uint64_t offset, int len, int iterations, int type) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_n = offset + (idx * iterations);
    char candidate[16];
    uint8_t hash[32];

    for (int i = 0; i < iterations; i++) {
        if (d_found) return;
        indexToPassword(start_n + i, len, candidate);

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
            return;
        }
    }
}

__global__ void dictionaryKernel(int type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_wordlist_count) return;

    // O(1) word access using pre-computed indices
    uint32_t word_start = d_word_indices[idx];
    uint32_t word_end = (idx < d_wordlist_count - 1) ? d_word_indices[idx + 1] : d_wordlist_size;

    char candidate[64];
    int len = 0;
    for (uint32_t i = word_start; i < word_end && d_wordlist[i] != '\n' && d_wordlist[i] != '\0'; i++) {
        candidate[len++] = d_wordlist[i];
    }
    candidate[len] = '\0';

    if (len == 0) return;

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

int main() {
    signal(SIGINT, handleSigint);

    char hex_input[128];
    char wordlist_path[256];
    char salt_input[64];
    char resume_choice = 'n';
    int hash_choice, attack_mode, min_len, max_len, limit_choice;
    uint64_t fixed_limit = 0;
    CheckpointState checkpoint = {0};

    printf("\n========================================\n");
    printf("   🚀 NEXO MASTER GPU CRACKER v4.0\n");
    printf("========================================\n");

    printf("\n[0] Select Mode:\n");
    printf("    1. Crack Hash      2. Benchmark\n");
    printf("    3. Hash Rate Estimate  4. Resume from Checkpoint\n");
    printf("    Choice: "); int mode_choice; scanf("%d", &mode_choice);

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
        printf("\n[1] Resume from checkpoint? (y/n): "); scanf(" %c", &resume_choice);
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
        printf("\n[1] Enter Target Hash: "); scanf("%s", hex_input);
        printf("\n[2] Select Hash Type:\n");
        printf("    1. SHA256 (64 hex)  2. SHA256 (32 hex)\n");
        printf("    3. MD5 (32 hex)     4. SHA-1 (40 hex)\n");
        printf("    5. NTLM (32 hex)    6. MySQL41 (40 hex)\n");
        printf("    7. MD5($pass.$salt)  8. SHA256($salt.$pass)\n");
        printf("    9. SHA256($pass.$salt)\n");
        printf("    Choice: "); scanf("%d", &hash_choice);
    }

    int target_bytes = 32;
    if (hash_choice == 2 || hash_choice == 3 || hash_choice == 5) target_bytes = 16;
    if (hash_choice == 4 || hash_choice == 6) target_bytes = 20;

    // Handle salt input for salted hash types
    if (hash_choice >= 7 && hash_choice <= 9) {
        printf("\n[3] Enter Salt: "); scanf("%s", salt_input);
        int salt_len = strlen(salt_input);
        cudaMemcpyToSymbol(c_salt, salt_input, salt_len + 1);
        cudaMemcpyToSymbol(c_salt_len, &salt_len, sizeof(int));
    }

    printf("\n[4] Select Attack Mode:\n");
    printf("    1. Brute-Force      2. Dictionary Attack\n");
    printf("    3. Mask Attack\n");
    printf("    Choice: "); scanf("%d", &attack_mode);

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
        printf("\n[5] Enter Mask Pattern (e.g., ?l?l?l?d?d): "); scanf("%s", mask_pattern);

        char h_charsets[10][128];
        int h_charset_sizes[10];
        int mask_len = parseMaskPattern(mask_pattern, h_charsets, h_charset_sizes);

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

        int threads = 256;
        int blocks = 2048;
        uint64_t batch_size = threads * blocks;
        uint64_t offset = 0;
        time_t mask_start = time(NULL);
        StatsState mask_stats;
        initStats(&mask_stats, total_combinations, mask_start);

        while (offset < total_combinations) {
            maskKernel<<<blocks, threads>>>(offset, hash_choice);
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(&h_found, d_found, sizeof(int));

            updateStats(&mask_stats, offset);
            if (offset % (batch_size * 10) == 0) displayStats(&mask_stats);

            if (h_found) {
                char res[64];
                cudaMemcpyFromSymbol(res, d_result, 64);
                printf("\n\n🎉 FOUND! Password: %s\n", res);
                addToPotfile("nexo.potfile", hex_input, res);
                return 0;
            }

            offset += batch_size;
        }
        printf("\n❌ Password not found in mask space.\n");
        return 0;
    }

    if (attack_mode == 2) {
        printf("\n[6] Enter Wordlist Path: "); scanf("%s", wordlist_path);

        FILE* fp = fopen(wordlist_path, "r");
        if (!fp) {
            printf("\n❌ Error: Cannot open wordlist file: %s\n", wordlist_path);
            return 1;
        }

        struct stat st;
        stat(wordlist_path, &st);
        long file_size = st.st_size;
        size_t chunk_size = DEFAULT_CHUNK_SIZE;
        size_t total_chunks = (file_size > 0) ? (size_t)((file_size + (long)chunk_size - 1) / (long)chunk_size) : 1;

        // Allocate reusable host/GPU buffers for chunked dictionary processing.
        char* chunk_buffer = (char*)malloc(chunk_size + 512);
        char carry_over[256] = {0};
        int carry_len = 0;

        if (!chunk_buffer) {
            fclose(fp);
            printf("\n❌ Error: Failed to allocate dictionary chunk buffer.\n");
            return 1;
        }

        char* d_wordlist_ptr = NULL;
        uint32_t* d_word_indices_ptr = NULL;
        CUDA_CHECK(cudaMalloc(&d_wordlist_ptr, chunk_size + 512));
        CUDA_CHECK(cudaMalloc(&d_word_indices_ptr, (chunk_size + 2) * sizeof(uint32_t)));

        int h_found = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(d_found, &h_found, sizeof(int)));

        time_t dict_start = time(NULL);
        StatsState dict_stats;
        initStats(&dict_stats, 0, dict_start);

        printf("\n📚 Starting Chunked Dictionary Attack (chunk size: %.2f MB)\n", (double)chunk_size / (1024 * 1024));

        uint64_t total_words_processed = 0;
        size_t chunk_idx = 0;

        while (!feof(fp) || carry_len > 0) {
            if (g_stop_requested) {
                printf("\n\n🛑 SIGINT received during dictionary mode. Stopping safely...\n");
                break;
            }

            // Bring carry-over to beginning of buffer before reading next chunk.
            if (carry_len > 0) memcpy(chunk_buffer, carry_over, carry_len);
            size_t bytes_read = fread(chunk_buffer + carry_len, 1, chunk_size - carry_len, fp);
            size_t total_bytes = bytes_read + carry_len;
            if (total_bytes == 0) break;

            chunk_idx++;
            size_t process_bytes = total_bytes;

            // If we are not at EOF, keep partial trailing word for the next chunk.
            if (!feof(fp)) {
                ssize_t last_nl = -1;
                for (ssize_t i = (ssize_t)total_bytes - 1; i >= 0; i--) {
                    if (chunk_buffer[i] == '\n') { last_nl = i; break; }
                }
                if (last_nl >= 0) {
                    process_bytes = (size_t)last_nl + 1;
                    carry_len = (int)(total_bytes - process_bytes);
                    if (carry_len >= (int)sizeof(carry_over)) {
                        // Fallback for unusually long single trailing token.
                        carry_len = 0;
                    } else if (carry_len > 0) {
                        memcpy(carry_over, chunk_buffer + process_bytes, carry_len);
                    }
                } else {
                    // Extremely long single line; process it as-is to avoid stalling.
                    carry_len = 0;
                }
            } else {
                carry_len = 0;
            }

            if (process_bytes == 0) continue;

            // Ensure final line in this processed chunk is terminated.
            if (chunk_buffer[process_bytes - 1] != '\n') {
                chunk_buffer[process_bytes++] = '\n';
            }
            chunk_buffer[process_bytes] = '\0';

            int word_count = 0;
            for (size_t i = 0; i < process_bytes; i++) {
                if (chunk_buffer[i] == '\n') word_count++;
            }
            if (word_count == 0) continue;

            uint32_t* word_indices = (uint32_t*)malloc((word_count + 1) * sizeof(uint32_t));
            if (!word_indices) {
                printf("\n❌ Error: Failed to allocate word indices.\n");
                break;
            }

            int current_word = 0;
            word_indices[0] = 0;
            for (size_t i = 0; i < process_bytes && current_word < word_count; i++) {
                if (chunk_buffer[i] == '\n') word_indices[++current_word] = (uint32_t)i + 1;
            }

            CUDA_CHECK(cudaMemcpy(d_wordlist_ptr, chunk_buffer, process_bytes + 1, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_word_indices_ptr, word_indices, (word_count + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyToSymbol(d_wordlist, &d_wordlist_ptr, sizeof(char*)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_word_indices, &d_word_indices_ptr, sizeof(uint32_t*)));
            int process_bytes_i = (int)process_bytes;
            CUDA_CHECK(cudaMemcpyToSymbol(d_wordlist_size, &process_bytes_i, sizeof(int)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_wordlist_count, &word_count, sizeof(int)));

            int threads = 256;
            int blocks = (word_count + threads - 1) / threads;
            dictionaryKernel<<<blocks, threads>>>(hash_choice);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            total_words_processed += word_count;
            dict_stats.total_hashes = total_words_processed;
            updateStats(&dict_stats, total_words_processed);
            printf("\r📦 Processing chunk %zu/%zu | words: %d | total words: %lu",
                   chunk_idx, total_chunks, word_count, total_words_processed);
            fflush(stdout);

            CUDA_CHECK(cudaMemcpyFromSymbol(&h_found, d_found, sizeof(int)));
            free(word_indices);
            if (h_found) break;
        }

        printf("\n");
        displayStats(&dict_stats);

        fclose(fp);
        free(chunk_buffer);
        CUDA_CHECK(cudaFree(d_wordlist_ptr));
        CUDA_CHECK(cudaFree(d_word_indices_ptr));

        if (h_found) {
            char res[64];
            CUDA_CHECK(cudaMemcpyFromSymbol(res, d_result, 64));
            printf("\n🎉 FOUND! Password: %s\n", res);
            addToPotfile("nexo.potfile", hex_input, res);
            return 0;
        } else {
            printf("\n❌ Password not found in wordlist.\n");
            return 0;
        }
    } else {
        printf("\n[6] Enter Length Range (min max): "); scanf("%d %d", &min_len, &max_len);
        printf("\n[7] Select Run Mode (1: 12h, 2: Fixed B): "); scanf("%d", &limit_choice);
        if(limit_choice == 2) { printf("    Enter Billions: "); double b; scanf("%lf", &b); fixed_limit = (uint64_t)(b * 1000000000ULL); }

        const char* h_charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%&*";
        int h_charset_len = strlen(h_charset);
        CUDA_CHECK(cudaMemcpyToSymbol(c_charset, h_charset, h_charset_len + 1));
        CUDA_CHECK(cudaMemcpyToSymbol(c_charset_len, &h_charset_len, sizeof(int)));

        int threads = 256, blocks = 2048, iterations = 5000;
        uint64_t batch_size = (uint64_t)blocks * threads * iterations;
        time_t wall_start = time(NULL); uint64_t total_scanned = 0;
        time_t last_checkpoint = time(NULL);

        // Multi-GPU Support
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count < 1) {
            fprintf(stderr, "\n❌ No CUDA-capable GPU found.\n");
            return 1;
        }
        printf("\n🖥️  Detected %d GPU(s)\n", device_count);
        for (int dev = 0; dev < device_count; dev++) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
            printf("    [%d] %s | %.2f GB VRAM\n", dev, prop.name, (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        }

        int selected_devices[32];
        int selected_count = 0;
        char device_input[128];
        printf("\n[8] Device Selection (-1 = all, e.g. 0,1,3): ");
        scanf("%127s", device_input);
        if (!parseDeviceList(device_input, device_count, selected_devices, &selected_count)) {
            printf("\n❌ Invalid device selection. Falling back to all devices.\n");
            selected_count = 0;
        }
        if (selected_count == 0) {
            for (int i = 0; i < device_count; i++) selected_devices[i] = i;
            selected_count = device_count;
        }
        printf("✅ Using %d GPU(s): ", selected_count);
        for (int i = 0; i < selected_count; i++) printf("%d%s", selected_devices[i], (i == selected_count - 1) ? "\n" : ", ");

        if (selected_count > 1) {
            printf("⚡ Using Multi-GPU mode with load balancing\n");
        }

        int start_len = is_resume ? checkpoint.current_len : min_len;
        uint64_t start_offset = is_resume ? checkpoint.offset : 0;
        uint64_t start_total = is_resume ? checkpoint.total_scanned : 0;

        for (int len = start_len; len <= max_len; len++) {
            uint64_t max_idx = 1; for(int i=0; i<len; i++) max_idx *= h_charset_len;
            printf("\n--- Length %d | Total: %.2f T ---\n", len, (double)max_idx/1e12);
            uint64_t offset = (len == start_len) ? start_offset : 0;
            total_scanned = (len == start_len) ? start_total : total_scanned;

            StatsState stats;
            initStats(&stats, max_idx, wall_start);

            while (offset < max_idx) {
                if (g_stop_requested) {
                    printf("\n\n🛑 SIGINT received. Saving checkpoint and exiting gracefully...\n");
                    updateCheckpointState(
                        &checkpoint, hash_choice, attack_mode, min_len, max_len, len,
                        offset, total_scanned, fixed_limit, wall_start,
                        hex_input, salt_input, wordlist_path
                    );
                    saveCheckpoint("nexo_checkpoint.bin", &checkpoint);
                    return 130;
                }

                if (difftime(time(NULL), wall_start) > 43200 || (fixed_limit > 0 && total_scanned >= fixed_limit)) break;

                // Distribute workload across GPUs
                const int base_blocks = blocks / selected_count;
                const int extra_blocks = blocks % selected_count;
                uint64_t dispatched_work = 0;

                for (int slot = 0; slot < selected_count; slot++) {
                    int dev = selected_devices[slot];
                    CUDA_CHECK(cudaSetDevice(dev));
                    int h_found_reset = 0;
                    CUDA_CHECK(cudaMemcpyToSymbol(d_found, &h_found_reset, sizeof(int)));

                    int dev_blocks = base_blocks + (slot < extra_blocks ? 1 : 0);
                    if (dev_blocks == 0) continue;

                    uint64_t dev_work = (uint64_t)dev_blocks * threads * iterations;
                    uint64_t dev_offset = offset + dispatched_work;
                    crackKernel<<<dev_blocks, threads>>>(dev_offset, len, iterations, hash_choice);
                    CUDA_CHECK(cudaGetLastError());
                    dispatched_work += dev_work;
                }

                // Synchronize all GPUs and check for found using host-side flag
                h_found_flag = 0;
                for (int slot = 0; slot < selected_count; slot++) {
                    int dev = selected_devices[slot];
                    CUDA_CHECK(cudaSetDevice(dev));
                    CUDA_CHECK(cudaDeviceSynchronize());
                    int dev_found = 0;
                    CUDA_CHECK(cudaMemcpyFromSymbol(&dev_found, d_found, sizeof(int)));
                    if (dev_found) {
                        h_found_flag = 1;
                        char res[32];
                        CUDA_CHECK(cudaMemcpyFromSymbol(res, d_result, 32));
                        printf("\n\n🎉 FOUND! Password: %s (GPU %d)\n", res, dev);
                        addToPotfile("nexo.potfile", hex_input, res);
                        return 0;
                    }
                }

                // Update and display stats
                updateStats(&stats, total_scanned);
                if (offset % (batch_size * 5) == 0) displayStats(&stats);

                // Save checkpoint every 5 minutes
                if (difftime(time(NULL), last_checkpoint) > 300) {
                    updateCheckpointState(
                        &checkpoint, hash_choice, attack_mode, min_len, max_len, len,
                        offset, total_scanned, fixed_limit, wall_start,
                        hex_input, salt_input, wordlist_path
                    );
                    saveCheckpoint("nexo_checkpoint.bin", &checkpoint);
                    last_checkpoint = time(NULL);
                }

                offset += dispatched_work;
                total_scanned += dispatched_work;
            }
        }
        printf("\n❌ Not found.\n"); return 0;
    }
}
