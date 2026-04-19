# NEXO MASTER GPU CRACKER v4.0

🚀 **High-Performance GPU Hash Cracker** for CUDA-enabled NVIDIA GPUs

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Compilation](#compilation)
- [Usage](#usage)
- [Attack Modes](#attack-modes)
- [Real-Time Statistics](#real-time-statistics)
- [Configuration](#configuration)
- [Examples](#examples)
- [Checkpoint System](#checkpoint-system)
- [Potfile System](#potfile-system)
- [Troubleshooting](#troubleshooting)
- [Performance Notes](#performance-notes)

---

## 🎯 Overview

NEXO MASTER GPU CRACKER is a CUDA-accelerated password hash cracking tool designed for NVIDIA GPUs. It supports multiple hash algorithms and attack modes with real-time performance monitoring.

### Supported Hash Types:
1. **SHA256 (64 hex)** - Full SHA-256 hash
2. **SHA256 (32 hex)** - Truncated SHA-256
3. **MD5 (32 hex)** - MD5 hash
4. **SHA-1 (40 hex)** - SHA-1 hash
5. **NTLM (32 hex)** - Windows NTLM hash
6. **MySQL41 (40 hex)** - MySQL 4.1+ (double SHA-1)
7. **MD5($pass.$salt)** - Salted MD5 (password + salt)
8. **SHA256($salt.$pass)** - Salted SHA-256 (salt + password)
9. **SHA256($pass.$salt)** - Salted SHA-256 (password + salt)

---

## ✨ Features

- **Multi-GPU Support** - Automatic load balancing across multiple GPUs using asynchronous streams
- **Hardware Monitoring** - Real-time monitoring of GPU temperature, power draw, and utilization (NVML)
- **Auto-Tuning Engine** - Optimized performance for Maxwell, Pascal, Volta, Turing, Ampere, Ada, and Hopper architectures
- **Real-Time Statistics** - Progress bar, hash rate, ETA, peak performance, and total processed
- **Checkpoint System** - Automatic saving every 5 minutes and graceful resumption
- **Potfile Support** - Skip already cracked hashes with persistent caching
- **Four Attack Modes** - Brute-force, Dictionary, Mask, and Hybrid
- **Moving Average** - Smooth hash rate calculations using a 10-point moving window
- **Benchmark Mode** - Comprehensive speed testing for all supported hash types
- **12-Hour Limit** - Built-in safety safeguards for long runs
- **Salted Hashes** - Native support for various salted hash formats

---

## 📦 Requirements

### Hardware:
- **NVIDIA GPU** with CUDA support (Compute Capability 5.0+)
- **Minimum**: GTX 970 or equivalent
- **Recommended**: RTX 20-series or better
- **Optimal**: RTX 30/40-series or B200/H100

### Software:
- **CUDA Toolkit** 11.0 or higher
- **NVIDIA Driver** 470+ (Linux) / 471+ (Windows)
- **GCC** 7.0+ (Linux) or Visual Studio 2019+ (Windows)
- **Make** (Linux) or MSBuild (Windows)

### System:
- **RAM**: 4GB minimum (8GB+ recommended)
- **Disk Space**: 100MB for program + space for wordlists
- **OS**: Linux (Ubuntu 20.04+, CentOS 7+) or Windows 10+

---

## 🔧 Installation

### Linux:

```bash
# 1. Install CUDA Toolkit (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda-toolkit-11-8

# 2. Clone or download the project
cd /path/to/nexo-project

# 3. Compile
nvcc -O3 -arch=sm_70 NEXO_MASTER_GPU_CRACKER.cu -o nexo_cracker

# 4. Run
./nexo_cracker
```

### Windows:

```cmd
REM 1. Install CUDA Toolkit from NVIDIA website
REM Download: https://developer.nvidia.com/cuda-downloads

REM 2. Open Visual Studio Command Prompt

REM 3. Navigate to project directory
cd C:\path\to\nexo-project

REM 4. Compile
nvcc -O3 -arch=sm_70 NEXO_MASTER_GPU_CRACKER.cu -o nexo_cracker.exe

REM 5. Run
nexo_cracker.exe
```

---

## 🔨 Compilation

### Basic Compilation:
```bash
nvcc -O3 -arch=sm_70 NEXO_MASTER_GPU_CRACKER.cu -o nexo_cracker
```

### Advanced Compilation Options:
```bash
# For specific GPU architecture
nvcc -O3 -arch=sm_75 NEXO_MASTER_GPU_CRACKER.cu -o nexo_cracker  # RTX 20-series
nvcc -O3 -arch=sm_80 NEXO_MASTER_GPU_CRACKER.cu -o nexo_cracker  # RTX 30-series
nvcc -O3 -arch=sm_89 NEXO_MASTER_GPU_CRACKER.cu -o nexo_cracker  # RTX 40-series
nvcc -O3 -arch=sm_90 NEXO_MASTER_GPU_CRACKER.cu -o nexo_cracker  # B200/H100

# With debug symbols
nvcc -O3 -g -G -arch=sm_70 NEXO_MASTER_GPU_CRACKER.cu -o nexo_cracker

# Maximum optimization
nvcc -O3 -maxrregcount=255 -arch=sm_70 NEXO_MASTER_GPU_CRACKER.cu -o nexo_cracker
```

### Architecture Reference:
| GPU | Architecture | SM Version |
|-----|--------------|------------|
| GTX 970/980 | Maxwell | sm_52 |
| GTX 10-series | Pascal | sm_61 |
| RTX 20-series | Turing | sm_75 |
| RTX 30-series | Ampere | sm_80/sm_86 |
| RTX 40-series | Ada | sm_89 |
| B200/H100 | Hopper | sm_90 |

---

## 🚀 Usage

### Basic Usage:
```bash
./nexo_cracker
```

### Interactive Menu:
```
========================================
   🚀 NEXO MASTER GPU CRACKER v4.0
========================================

[0] Select Mode:
    1. Crack Hash      2. Benchmark
    3. Hash Rate Estimate  4. Resume from Checkpoint
    Choice:
```

### Step-by-Step Guide:

#### 1. Select Mode:
```
Choice: 1  # For cracking a hash
```

#### 2. Enter Target Hash:
```
[1] Enter Target Hash: 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8
```

#### 3. Select Hash Type:
```
[2] Select Hash Type:
    1. SHA256 (64 hex)  2. SHA256 (32 hex)
    3. MD5 (32 hex)     4. SHA-1 (40 hex)
    5. NTLM (32 hex)    6. MySQL41 (40 hex)
    7. MD5($pass.$salt)  8. SHA256($salt.$pass)
    9. SHA256($pass.$salt)
    Choice: 1
```

#### 4. Enter Salt (if salted hash):
```
[3] Enter Salt: mysalt123
```
*(Only for hash types 7, 8, 9)*

#### 5. Select Attack Mode:
```
[4] Select Attack Mode:
    1. Brute-Force      2. Dictionary Attack
    3. Mask Attack      4. Hybrid Attack (Dictionary + Mask)
    Choice: 1
```

#### 6. Configure Attack Parameters:

**Brute-Force:**
```
[6] Enter Length Range (min max): 6 8
[7] Select Run Mode (1: 12h, 2: Fixed B): 1
```

**Dictionary:**
```
[6] Enter Wordlist Path: /path/to/wordlist.txt
```

**Mask:**
```
[5] Enter Mask Pattern (e.g., ?l?l?l?d?d): ?l?l?l?d?d
```

---

## ⚔️ Attack Modes

### 1. Brute-Force Attack

Exhaustively searches all combinations within specified length range.

**Syntax:**
```
Length Range: min_len max_len
Run Mode: 1 (12-hour limit) or 2 (fixed billion limit)
```

**Example:**
```
[6] Enter Length Range (min max): 6 8
[7] Select Run Mode (1: 12h, 2: Fixed B): 1
```

**Best For:**
- Short passwords (≤8 characters)
- Unknown password structure
- When no wordlist available

**Performance:**
- 6 chars: ~68 billion combinations (seconds to minutes)
- 7 chars: ~4.2 trillion combinations (minutes to hours)
- 8 chars: ~262 trillion combinations (hours to days)

---

### 2. Dictionary Attack

Tests passwords from a wordlist file.

**Syntax:**
```
Wordlist Path: /path/to/wordlist.txt
```

**Example:**
```
[6] Enter Wordlist Path: /usr/share/wordlists/rockyou.txt
```

**Best For:**
- Common passwords
- Leaked password databases
- Known password patterns

**Wordlist Format:**
```
password
123456
admin
test
qwerty
(one password per line)
```

**Performance:**
- 10 million words: <1 second
- 100 million words: ~5-10 seconds
- 1 billion words: ~1-2 minutes

---

### 3. Mask Attack

Targeted attack using pattern-based generation.

**Mask Characters:**
- `?l` - Lowercase letters (a-z)
- `?u` - Uppercase letters (A-Z)
- `?d` - Digits (0-9)
- `?s` - Special characters (!@#$%^&*()_+-=[]{}|;:,.<>?)
- `?a` - All characters (a-z, A-Z, 0-9, special)
- Static characters - Literal characters

**Syntax:**
```
Mask Pattern: ?l?l?l?d?d
```

**Examples:**
```
?l?l?l?d?d      → 3 lowercase + 2 digits (e.g., abc12)
?u?l?l?l?l!     → 1 uppercase + 4 lowercase + ! (e.g., Zabcd!)
?d?d?d?d        → 4 digits (e.g., 1234)
?a?a?a?a        → 4 any characters (e.g., aB1@)
Test?d?d        → "Test" + 2 digits (e.g., Test12)
```

**Best For:**
- Known password patterns
- Corporate password policies
- Partially known passwords

**Performance:**
- `?l?l?l?d?d`: 67.6 million combinations (~5 seconds)
- `?u?l?l?l?l?d?d`: 3.2 billion combinations (~2-3 minutes)
---

### 4. Hybrid Attack (Dictionary + Mask)

Combines wordlist words with custom mask patterns (e.g., word + digits).

**Syntax:**
```
Wordlist Path: /path/to/wordlist.txt
Mask Pattern: ?d?d
```

**Example:**
```
[6] Enter Wordlist Path: company_names.txt
[7] Enter Mask Pattern (e.g., ?d?d): ?d?d
```
*Result: Testing "Google01", "Apple99", etc.*

**Best For:**
- Passwords following "Word + Year" or "Name + Suffix" patterns
- Targeted attacks on users known to use specific bases
- Significantly faster than pure brute-force for large bases

**Performance:**
- Large Wordlist + `?d?d`: Near wordlist speed
- Medium Wordlist + `?l?l?l`: Dependent on mask complexity

## 📊 Real-Time Statistics

### Display Format:
```
[████████████████████████░░░░░░░░░░░░░░░░░░░░] 45.67%
🌡️ 68°C | ⚡ 320W | 🌀 99%  ⚡ 12.5GH/s | 📈 11.8GH/s (avg) | 🔝 13.2GH/s (peak)  📊 2.18T/4.78T  ⏱️ 3m 45s elapsed | ⏳ 4m 32s ETA
```

### Components Explained:

1. **Progress Bar** `[████░░░░]`
   - 40 characters wide
   - `█` = completed portion
   - `░` = remaining portion
   - Shows percentage completion

2. **Current Speed** `⚡ 12.5GH/s`
   - Instantaneous hash rate
   - Moving average of last 10 readings
   - Smooths out fluctuations

3. **Average Speed** `📈 11.8GH/s (avg)`
   - Overall average since start
   - Long-term performance indicator

4. **Peak Speed** `🔝 13.2GH/s (peak)`
   - Maximum achieved hash rate
   - Best performance recorded

5. **Progress** `📊 2.18T/4.78T`
   - Current hashes processed
   - Total hashes to process
   - Auto-formatted: K, M, G, T

6. **Elapsed Time** `⏱️ 3m 45s elapsed`
   - Time since attack started
   - Format: seconds, minutes, hours, days

7. **ETA** `⏳ 4m 32s ETA`
   - Estimated time remaining
   - Weighted calculation (70% current + 30% average)
   - Adapts to performance changes

### Update Frequency:
- Updates every 5-10 batches
- Smooths performance fluctuations
- Minimal display overhead

---

## ⚙️ Configuration

### Default Settings:
```c
int threads = 256;        // Threads per block
int blocks = 2048;        // Number of blocks
int iterations = 5000;    // Iterations per launch
int bar_width = 40;       // Progress bar width
```

### Adjusting Performance:

**For Faster Performance (More GPU Utilization):**
```c
int threads = 512;        // Increase threads
int blocks = 4096;        // Increase blocks
int iterations = 10000;   // Increase iterations
```

**For Lower GPU Memory Usage:**
```c
int threads = 128;        // Decrease threads
int blocks = 1024;        // Decrease blocks
int iterations = 2500;    // Decrease iterations
```

### Multi-GPU Configuration:
Code automatically detects and uses all available GPUs with load balancing.

---

## 📝 Examples

### Example 1: Brute-Force SHA256 Hash
```bash
./nexo_cracker
Choice: 1
Enter Target Hash: 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8
Hash Type: 1
Attack Mode: 1
Length Range: 6 8
Run Mode: 1
```

### Example 2: Dictionary Attack MD5
```bash
./nexo_cracker
Choice: 1
Enter Target Hash: 5f4dcc3b5aa765d61d8327deb882cf99
Hash Type: 3
Attack Mode: 2
Wordlist Path: /usr/share/wordlists/rockyou.txt
```

### Example 3: Mask Attack NTLM
```bash
./nexo_cracker
Choice: 1
Enter Target Hash: 31d6cfe0d16ae931b73c59d7e0c089c0
Hash Type: 5
Attack Mode: 3
Mask Pattern: ?u?l?l?l?d?d
```

### Example 4: Salted Hash
```bash
./nexo_cracker
Choice: 1
Enter Target Hash: 8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918
Hash Type: 8
Enter Salt: mysalt123
Attack Mode: 1
Length Range: 6 10
Run Mode: 1
```

### Example 5: Benchmark GPU
```bash
./nexo_cracker
Choice: 2
```

### Example 6: Resume from Checkpoint
```bash
./nexo_cracker
Choice: 4
```

---

## 💾 Checkpoint System

### What is Checkpoint?
Checkpoint saves the current attack state to disk, allowing you to resume after interruption.

### Checkpoint File:
- **Filename**: `nexo_checkpoint.bin`
- **Format**: Binary
- **Size**: ~512 bytes

### Saved Information:
- Hash choice
- Attack mode
- Length range
- Current offset
- Total hashes scanned
- Target hash
- Salt (if applicable)
- Wordlist path (if dictionary mode)
- Start time

### Automatic Checkpoint:
- Saves every 5 minutes
- Saves before exit (interrupted by Ctrl+C)
- Saves on 12-hour limit reached

### Manual Resume:
```
[0] Select Mode:
    1. Crack Hash      2. Benchmark
    3. Hash Rate Estimate  4. Resume from Checkpoint
    Choice: 4

📂 Checkpoint loaded from nexo_checkpoint.bin
📋 Resuming from length 7, offset 1234567890
```

### Checkpoint Location:
```
Current directory: ./nexo_checkpoint.bin
```

### Delete Checkpoint:
```bash
rm nexo_checkpoint.bin
```

---

## 🗂️ Potfile System

### What is Potfile?
Potfile (password file) stores cracked hashes in `hash:password` format.

### Potfile Format:
```
5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8:password123
d41d8cd98f00b204e9800998ecf8427e:
098f6bcd4621d373cade4e832627b4f6:test
```

### Benefits:
- **Skip Already Cracked** - Don't re-crack same hash
- **Persistent Storage** - Survives program restarts
- **Standard Format** - Compatible with hashcat, John the Ripper
- **Simple Text File** - Easy to view/edit

### How It Works:

**First Run:**
```
Hash not in potfile → Crack it → Save to potfile
```

**Second Run:**
```
Hash found in potfile → Skip cracking → Display result immediately
```

### Potfile Location:
```
Current directory: ./nexo.potfile
```

### Manual Entry:
```bash
echo "5e884898...:password123" >> nexo.potfile
```

### View Potfile:
```bash
cat nexo.potfile
```

### Delete Potfile:
```bash
rm nexo.potfile
```

---

## 🔧 Troubleshooting

### Issue: "CUDA Error: no capable device found"
**Solution:**
- Check GPU is CUDA-compatible
- Update NVIDIA drivers
- Verify CUDA Toolkit installation
- Check `nvidia-smi` output

### Issue: "Error: Cannot open wordlist file"
**Solution:**
- Verify file path is correct
- Check file permissions
- Ensure file exists
- Use absolute path if needed

### Issue: Low hash rates
**Solution:**
- Check GPU utilization (`nvidia-smi`)
- Increase threads/blocks/iterations
- Verify GPU is not thermal throttling
- Close other GPU-intensive applications

### Issue: Program crashes
**Solution:**
- Reduce threads/blocks/iterations (lower memory usage)
- Check GPU memory availability
- Update CUDA drivers
- Recompile with debug symbols

### Issue: Checkpoint not loading
**Solution:**
- Verify `nexo_checkpoint.bin` exists
- Check file permissions
- Ensure checkpoint is from same hash
- Delete corrupted checkpoint and start fresh

### Issue: Hash not found but should be
**Solution:**
- Verify hash type is correct
- Check character set includes needed characters
- Increase length range
- Try different attack mode

---

## ⚡ Performance Notes

### GPU Performance Estimates (RTX 4090):

| Hash Type | Hash Rate |
|-----------|------------|
| MD5 (32 hex) | 25-30 GH/s |
| SHA-1 (40 hex) | 16-24 GH/s |
| SHA256 (64 hex) | 10-15 GH/s |
| SHA256 (32 hex) | 12-18 GH/s |
| NTLM (32 hex) | 20-30 GH/s |
| MySQL41 (40 hex) | 6-10 GH/s |
| Salted variants | 6-14 GH/s |

### B200/H100 Performance:

| Hash Type | Hash Rate |
|-----------|------------|
| MD5 (32 hex) | 50-60 GH/s |
| SHA-1 (40 hex) | 35-45 GH/s |
| SHA256 (64 hex) | 20-30 GH/s |
| SHA256 (32 hex) | 25-35 GH/s |
| NTLM (32 hex) | 40-50 GH/s |

### Performance Factors:
- **Password Length**: Shorter = faster
- **Character Set**: Smaller = faster
- **Hash Algorithm**: Simpler = faster
- **GPU Architecture**: Newer = faster
- **Multi-GPU**: Near-linear scaling

### Real-World Examples:

**8-character password (62 charset):**
- ~2.18 trillion combinations
- At 12 GH/s: ~3 minutes
- At 24 GH/s: ~1.5 minutes

**12-Hour Capacity (SHA256):**
- At 10 GH/s: 432 trillion hashes
- At 15 GH/s: 648 trillion hashes
- Average: ~540 trillion hashes

---

## 🛡️ Security Notes

### Legal Use Only:
- Use only on hashes you own or have permission to crack
- Unauthorized password cracking is illegal
- Educational and authorized security testing only

### Best Practices:
- Keep potfiles secure (contain cracked passwords)
- Delete sensitive data after use
- Use in isolated environments
- Follow responsible disclosure

---

## 📞 Support

### For Issues:
- Check GPU compatibility
- Verify CUDA installation
- Review troubleshooting section
- Check hash type and format

### Contributing:
- Report bugs with system info
- Suggest performance improvements
- Share optimization tips

---

## 📄 License

This tool is for educational and authorized security testing purposes only.

---

## 🎓 Learning Resources

- **CUDA Programming**: https://developer.nvidia.com/cuda-toolkit
- **Hash Algorithms**: https://en.wikipedia.org/wiki/Cryptographic_hash_function
- **Password Security**: https://www.sans.org/security-resources/password-security/

---

**Version**: 4.0
**Last Updated**: 2026
**Author**: Prashant

---

*Happy Cracking! 🚀*
