# Warp-Level Reduction Optimization

## Summary of Changes

All reduction kernels in `cvdv.cu` have been refactored to use **modern warp-level reduction** instead of the traditional tree-based shared memory reduction. This is the current industry standard for CUDA reductions.

### Kernels Optimized
1. `kernelComputeRegisterNorm` - Computes norm of a single register
2. `kernelExpectX2` - Computes ⟨x²⟩ expectation value
3. `kernelComputeNorm` - Computes total state norm
4. `kernelMeasureMultiple` - Computes joint marginal probabilities for N registers

---

## What Changed?

### **Before: Tree-Based Shared Memory Reduction**
```c
__global__ void kernelComputeNorm(double* partialSums, const cuDoubleComplex* state, size_t totalSize) {
    extern __shared__ double sdataNorm[];  // Dynamic shared memory allocation
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double localSum = 0.0;
    
    // Grid-stride loop
    for (size_t globalIdx = idx; globalIdx < totalSize; globalIdx += blockDim.x * gridDim.x) {
        localSum += absSquare(state[globalIdx]);
    }
    
    // Store in shared memory
    sdataNorm[threadIdx.x] = localSum;
    __syncthreads();
    
    // Tree-based reduction with many syncs
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdataNorm[threadIdx.x] += sdataNorm[threadIdx.x + s];
        }
        __syncthreads();  // Sync at every iteration!
    }
    
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = sdataNorm[0];
    }
}

// Launch with dynamic shared memory
kernelComputeNorm<<<numBlocks, CUDA_BLOCK_SIZE, sharedMemBytes>>>(...)
```

**Problems:**
- Requires `blockDim.x * sizeof(double)` bytes of shared memory per block
- Multiple `__syncthreads()` calls (log₂(blockDim) syncs)
- Potential bank conflicts when accessing shared memory
- Warp divergence in the final iterations
- Shared memory bandwidth bottleneck

---

### **After: Warp-Level Reduction with Shuffle**
```c
__global__ void kernelComputeNorm(double* partialSums, const cuDoubleComplex* state, size_t totalSize) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double localSum = 0.0;
    
    // Grid-stride loop
    for (size_t globalIdx = idx; globalIdx < totalSize; globalIdx += blockDim.x * gridDim.x) {
        localSum += absSquare(state[globalIdx]);
    }
    
    // Modern warp-level reduction (faster, no bank conflicts)
    localSum = blockReduceSum(localSum);
    
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = localSum;
    }
}

// Launch without dynamic shared memory
kernelComputeNorm<<<numBlocks, CUDA_BLOCK_SIZE>>>(...)
```

**Helper Functions:**
```c
// Warp-level reduction using shuffle instructions
__device__ __forceinline__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using warp reduction + minimal shared memory
__device__ __forceinline__ double blockReduceSum(double val) {
    static __shared__ double warpSums[32];  // Only 32 doubles (256 bytes)
    
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    // Reduce within warp using shuffle (no shared memory!)
    val = warpReduceSum(val);
    
    // First thread in each warp writes to shared memory
    if (lane == 0) warpSums[wid] = val;
    __syncthreads();
    
    // First warp reduces the warp sums
    if (threadIdx.x < 32) {
        val = (threadIdx.x < blockDim.x / 32) ? warpSums[threadIdx.x] : 0.0;
        val = warpReduceSum(val);
    }
    
    return val;
}
```

---

## Performance Benefits Explained

### **1. Warp Shuffle Instructions (`__shfl_down_sync`)**

**What it does:**
- Allows threads within a warp (32 threads) to directly exchange register values
- No shared memory access needed
- Hardware-accelerated operation

**Example:**
```
Iteration 1 (offset=16):
Thread 0:  val += val from thread 16
Thread 1:  val += val from thread 17
...
Thread 15: val += val from thread 31

Iteration 2 (offset=8):
Thread 0:  val += val from thread 8
Thread 1:  val += val from thread 9
...

... continues until offset=1
```

After 5 iterations, thread 0 has the sum of all 32 threads in the warp.

**Why it's faster:**
- **Register-to-register transfer** (fastest memory on GPU)
- **No memory transactions** (shared or global)
- **No bank conflicts** (not using shared memory)
- **No explicit synchronization** within a warp (threads execute in lockstep)

---

### **2. Reduced Shared Memory Usage**

**Old approach:**
- 256 threads × 8 bytes = **2,048 bytes** per block

**New approach:**
- 32 warps × 8 bytes = **256 bytes** per block
- **8x less shared memory!**

**Impact:**
- Higher occupancy (more blocks can fit on an SM)
- Less shared memory bank conflicts
- More shared memory available for other kernels

---

### **3. Fewer Synchronization Points**

**Old approach (256 threads):**
```
__syncthreads()  // After storing to shared memory
__syncthreads()  // s = 128
__syncthreads()  // s = 64
__syncthreads()  // s = 32
__syncthreads()  // s = 16
__syncthreads()  // s = 8
__syncthreads()  // s = 4
__syncthreads()  // s = 2
__syncthreads()  // s = 1
Total: 9 synchronization points
```

**New approach:**
```
// Warp reduction: 0 syncs (implicit within warp)
__syncthreads()  // Only 1 sync between warps
Total: 1 synchronization point
```

**Why this matters:**
- `__syncthreads()` is expensive (stalls all threads in block)
- Fewer syncs = less latency
- Better instruction-level parallelism

---

### **4. No Warp Divergence**

**Old approach:**
```c
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {  // Divergence!
        sdataNorm[threadIdx.x] += sdataNorm[threadIdx.x + s];
    }
    __syncthreads();
}
```

When `s = 16`:
- Threads 0-15: Active (execute addition)
- Threads 16-255: Inactive (wait)
- **Warp divergence** in warps containing both active and inactive threads

**New approach:**
```c
val = warpReduceSum(val);  // All threads in warp participate
```

- All 32 threads in each warp execute shuffle instructions
- No divergence within warps
- Better SIMT efficiency

---

### **5. No Bank Conflicts**

**Shared memory banks:**
- 32 banks on modern GPUs
- Bank conflict when multiple threads in a warp access different addresses in the same bank

**Old approach:**
```c
sdataNorm[threadIdx.x] += sdataNorm[threadIdx.x + s];
```

When `s = 32` and `blockDim.x = 256`:
- Thread 0 accesses: `sdata[0]` and `sdata[32]` (bank 0)
- Thread 1 accesses: `sdata[1]` and `sdata[33]` (bank 1)
- ...
- Thread 31 accesses: `sdata[31]` and `sdata[63]` (bank 31)

No conflicts here, but potential conflicts exist in other patterns.

**New approach:**
- Warp shuffle uses **registers only**
- Zero bank conflicts possible

## Further Reading

### **NVIDIA Resources**
1. [Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
2. [Cooperative Groups](https://developer.nvidia.com/blog/cooperative-groups/) - Even more modern approach
3. [CUB Library](https://nvlabs.github.io/cub/) - Production-ready primitives

### **Academic Papers**
- "Optimizing Parallel Reduction in CUDA" - Mark Harris, NVIDIA
- "Modern GPU" - Sean Baxter (covers reduction patterns)

### **Code Examples**
- [CUDA Samples: Reduction](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/reduction)
- [CUB DeviceReduce](https://github.com/NVIDIA/cub/blob/main/cub/device/device_reduce.cuh)
