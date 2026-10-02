# Lab Sheet 9: Introduction to CUDA (Google Colab Edition)

**Course:** SE3082 - Parallel Computing  
**Institution:** SLIIT (BSc (Hons) in Information Technology - Year 3)  
**Semester:** Semester 2, 2026  

---

## Introduction

In this lab, you will write and run your first CUDA C++ programs on a real NVIDIA GPU using Google Colab. Colab gives you a free GPU in your browser, so there is nothing to install and nothing to pay for. All the programs from the lecture are here, followed by two exercises where you write the kernel yourself.

### Learning Objectives
By the end of this lab, you will be able to:
* Switch on a GPU runtime in Colab and confirm it is working with `nvidia-smi`.
* Write, compile (`nvcc`), and run a CUDA program from notebook cells.
* Explain the host/device pattern: **copy in → launch kernel → copy out**.
* Map blocks and threads onto array elements, in one dimension and in two, with a bounds check.

---

## Exercise Overview

| Exercise | Topic |
| :--- | :--- |
| **1** | Set up your CUDA environment in Google Colab |
| **2** | Your first kernel: adding two numbers |
| **3** | Vector addition with 512 blocks and one thread each |
| **4** | Vector addition with one block and 512 threads |
| **5** | Multiplying two 10,000,000-element vectors (blocks and threads) |
| **6** | Multiplying two $10,000 \times 10,000$ matrices element by element (2D grid) |

> **Prerequisites:** A Google account and a web browser. Colab menu names change slightly from time to time; if a menu item is not exactly where this sheet says, look for the closest match.

---

## Exercise 1 - Set up your CUDA environment in Google Colab

### Step 1 - Create a notebook
1. Open [colab.research.google.com](https://colab.research.google.com) and sign in with your Google account.
2. Choose **File → New notebook**.
3. Click the notebook name at the top left and rename it to `SE3082_Lab09_<your student ID>`.

### Step 2 - Switch on the GPU
1. Choose **Runtime → Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**, then click **Save**.
3. Colab connects to a machine with a GPU when you run your first cell.

> **TIP:** A notebook has two kinds of cells: **Text cells** for notes and **Code cells** for code. Use `+ Text` to add a heading such as "Exercise 2" above each exercise so your notebook is easy to read. Run a cell with its play button or `Shift + Enter`. Lines starting with `!` are sent to the Linux shell instead of Python.

### Step 3 - Check that the GPU is visible
In a Colab **CODE** cell, execute:
```bash
!nvidia-smi
```
The output should show a table that includes the GPU name (normally Tesla T4), about 15 GB of memory, and a CUDA Version line. If it says the command was not found or failed to communicate with the driver, your runtime has no GPU. Go back to Step 2.

### Step 4 - Check the CUDA compiler
In a Colab **CODE** cell, execute:
```bash
!nvcc --version
```
You should see a line such as `Cuda compilation tools, release 12.x`. If the shell cannot find `nvcc`, try `!/usr/local/cuda/bin/nvcc --version` and use that full path in the compile commands.

### Step 5 - Find your GPU's architecture code
The compiler needs to know which GPU generation to build for. Ask the driver for the compute capability:
```bash
!nvidia-smi --query-gpu=name,compute_cap --format=csv
```
Remove the dot to get the architecture code (e.g., compute capability `7.5` means `-arch=sm_75`). If the column is not supported, use this table:

| GPU shown by `nvidia-smi` | Compute capability | Compiler flag |
| :--- | :--- | :--- |
| Tesla T4 (free tier, normal case) | 7.5 | `-arch=sm_75` |
| NVIDIA L4 | 8.9 | `-arch=sm_89` |
| NVIDIA A100 | 8.0 | `-arch=sm_80` |
| NVIDIA H100 | 9.0 | `-arch=sm_90` |

*This sheet uses `-arch=sm_75` everywhere. If your GPU is different, change it in every compile cell.*

### Step 6 - The write → compile → run workflow
Colab has no CUDA editor, so you will keep each program inside a code cell and turn it into a file using the `%%writefile` command. Every exercise follows these three steps:

| Step | Action | Cell Content |
| :--- | :--- | :--- |
| **1. Write** | Save the program as a `.cu` file on Colab's disk | `%%writefile name.cu` on the first line, followed by code |
| **2. Compile** | Turn the file into an executable | `!nvcc -arch=sm_75 name.cu -o name` |
| **3. Run** | Run the executable | `!./name` |

Steps 2 and 3 can share one cell:  
`!nvcc -arch=sm_75 name.cu -o name && ./name`  
*(The `&&` means "run the program only if the compile succeeded".)*

> **COMMON MISTAKE:**  
> If you change the code, you **must** run the `%%writefile` cell again and then compile again. Editing the text without doing both means you are still running the old program.  
> `%%writefile` **must** be the very first line of its cell. No blank lines or comments above it!

### Step 7 - Smoke test: Hello from the GPU
Run these two cells to prove the whole toolchain works.

**Cell 1 (Write `hello.cu`):**
```cpp
%%writefile hello.cu
#include <cstdio>

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main(void) {
    hello<<<1, 4>>>(); // 1 block of 4 threads
    cudaDeviceSynchronize(); // Wait so the GPU's output is flushed
    return 0;
}
```

**Cell 2 (Compile and Run):**
```bash
!nvcc -arch=sm_75 hello.cu -o hello && ./hello
```

You should see four lines: `"Hello from GPU thread 0"` to `"thread 3"`. The lines may appear in any order since the four threads run concurrently.

### Step 8 - Colab Housekeeping
* **Temporary Disk:** Your code is safe (saved to Google Drive), but `.cu` files and executables live on a temporary disk. If the runtime disconnects, reconnect and use **Runtime → Run all**.
* **Release GPU:** Free GPU time is limited. When finished, select **Runtime → Disconnect and delete runtime**.
* **No Cost:** You do not need Colab Pro, credit cards, or AWS accounts. Ignore any upgrade prompts.

---

### Exercise 1 Checklist
- [ ] `nvidia-smi` cell shows a GPU (note its name and compute capability).
- [ ] `nvcc --version` cell shows a CUDA release.
- [ ] `hello` program prints four lines.

---

## Exercise 2 - Simple Calculation Program from the Lecture

This program adds two integers on the GPU using **1 block containing 1 thread**.

Every CUDA program follows three key steps:
1. **Copy inputs** from host (CPU) to device (GPU) memory: `cudaMemcpy(..., cudaMemcpyHostToDevice)`
2. **Launch kernel** on the GPU: `add<<<1, 1>>>(d_a, d_b, d_c)`
3. **Copy result** back from device to host: `cudaMemcpy(..., cudaMemcpyDeviceToHost)`

**Cell 1 (Write `ex2_add.cu`):**
```cpp
%%writefile ex2_add.cu
#include <cstdio>

// Kernel: runs on the device (NVIDIA GPU)
__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

// Host = CPU, Device = GPU
int main(void) {
    int a, b, c;         // Host copies
    int *d_a, *d_b, *d_c; // Device copies
    int size = sizeof(int);

    // Allocate space for device copies
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Set up input values
    a = 2;
    b = 7;

    // 1. Copy inputs from host to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // 2. Launch kernel: 1 block, 1 thread
    add<<<1, 1>>>(d_a, d_b, d_c);

    // Report a failed launch instead of silently printing garbage
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    // 3. Copy result from device to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Result is %d\n", c);

    // Cleanup
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);

    return 0;
}
```

**Cell 2 (Compile and Run):**
```bash
!nvcc -arch=sm_75 ex2_add.cu -o ex2_add && ./ex2_add
```
*Expected Output:* `Result is 9`

### Think
1. Why do we need separate variables `d_a`, `d_b`, `d_c` on the device? Why can the kernel not use `a`, `b`, `c` directly?
2. In `cudaMemcpy(d_a, &a, ...)`, we pass `&a`, but in the kernel launch, we pass `d_a`. Why?
3. How many threads run the `add` kernel? What would you change to run it with 10 threads, and would the answer change?

---

## Exercise 3 - Vector Addition using 512 Blocks and One Thread

Now we add two vectors of 512 integers. The kernel is launched with **512 blocks of 1 thread each**. Inside the kernel, `blockIdx.x` tells each thread which element it is responsible for.

**Cell 1 (Write `ex3_blocks.cu`):**
```cpp
%%writefile ex3_blocks.cu
#include <cstdio>
#include <cstdlib>

#define N 512

__global__ void add(int *a, int *b, int *c) {
    // Each block runs only one addition
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void random_ints(int *x, int size) {
    for (int i = 0; i < size; i++)
        x[i] = rand() % 100;
}

int main(void) {
    int *a, *b, *c;        // Host copies
    int *d_a, *d_b, *d_c;  // Device copies
    int size = N * sizeof(int);

    // Allocate space for device copies
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate space for host copies and fill with inputs
    a = (int*)malloc(size); random_ints(a, N);
    b = (int*)malloc(size); random_ints(b, N);
    c = (int*)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel: N blocks, 1 thread per block
    add<<<N, 1>>>(d_a, d_b, d_c);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (int r = 0; r < N; r++)
        printf("%d + %d = %d\n", a[r], b[r], c[r]);

    // Check GPU's answers against CPU
    int errors = 0;
    for (int r = 0; r < N; r++)
        if (c[r] != a[r] + b[r]) errors++;

    printf("Verification: %d mismatches out of %d\n", errors, N);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
```

**Cell 2 (Compile and Run):**
```bash
!nvcc -arch=sm_75 ex3_blocks.cu -o ex3_blocks && ./ex3_blocks
```

*Expected Last Line:* `Verification: 0 mismatches out of 512`

### Predict Before You Run
How many blocks and how many threads in total does `add<<<N, 1>>>` launch? A GPU executes threads in groups of 32 called **warps**. What does that mean for a block that contains only one thread?

---

## Exercise 4 - Vector Addition using One Block and 512 Threads

The same calculation, but now with **1 block of 512 threads**. The differences are the launch configuration `<<<1, N>>>` and using `threadIdx.x` inside the kernel.

**Cell 1 (Write `ex4_threads.cu`):**
```cpp
%%writefile ex4_threads.cu
#include <cstdio>
#include <cstdlib>

#define N 512

__global__ void addT(int *a, int *b, int *c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void random_ints(int *x, int size) {
    for (int i = 0; i < size; i++)
        x[i] = rand() % 100;
}

int main(void) {
    int *a, *b, *c;        // Host copies
    int *d_a, *d_b, *d_c;  // Device copies
    int size = N * sizeof(int);

    // Allocate space for device copies
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate host copies and set up inputs
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel: 1 block of N threads
    addT<<<1, N>>>(d_a, d_b, d_c);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (int r = 0; r < N; r++)
        printf("%d) %d + %d = %d\n", r, a[r], b[r], c[r]);

    // Check GPU's answers against CPU
    int errors = 0;
    for (int r = 0; r < N; r++)
        if (c[r] != a[r] + b[r]) errors++;

    printf("Verification: %d mismatches out of %d\n", errors, N);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
```

**Cell 2 (Compile and Run):**
```bash
!nvcc -arch=sm_75 ex4_threads.cu -o ex4_threads && ./ex4_threads
```

### Experiment - Find the Limit
In `ex4_threads.cu`, change `#define N 512` to `#define N 2048`. Re-run the write cell, compile, and run. What is printed, and which part of the program told you? Which limit of the GPU did you hit? *(Change `N` back to 512 afterwards.)*

---

## Exercise 5 - Multiply Two Vectors using Blocks and Threads

Write a CUDA program that multiplies two vectors element by element: $c[i] = a[i] \times b[i]$.

**Requirements:**
- [x] Array size of 10,000,000 elements.
- [x] Use 512 threads per block, and enough blocks to cover all 10,000,000 elements.
- [x] Print the last 1000 results calculated by the kernel.

### The Global Index Formula
Inside a grid of blocks:
$$\text{Global Index } i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

To round up the total number of blocks needed:
$$\text{blocks} = \frac{N + \text{THREADS\_PER\_BLOCK} - 1}{\text{THREADS\_PER\_BLOCK}}$$

**Cell 1 (Write `ex5_vecmul.cu` - Fill in TODOs):**
```cpp
%%writefile ex5_vecmul.cu
#include <cstdio>
#include <cstdlib>

#define N 10000000             // 10 million elements
#define THREADS_PER_BLOCK 512

__global__ void vecMul(int *a, int *b, int *c, int n) {
    // TODO 1: Calculate global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO 2: Guard against out-of-bounds access
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

void random_ints(int *x, int size) {
    for (int i = 0; i < size; i++)
        x[i] = rand() % 100;
}

int main(void) {
    int *a, *b, *c;        // Host copies
    int *d_a, *d_b, *d_c;  // Device copies

    size_t size = (size_t)N * sizeof(int);

    // Allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate host memory and generate values
    a = (int*)malloc(size); random_ints(a, N);
    b = (int*)malloc(size); random_ints(b, N);
    c = (int*)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // TODO 3: Calculate blocks needed to cover N elements (rounded UP)
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Launching %d blocks x %d threads\n", blocks, THREADS_PER_BLOCK);

    // TODO 4: Launch vecMul kernel
    vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    // TODO 5: Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // TODO 6: Print the last 1000 results
    for (int i = N - 1000; i < N; i++) {
        printf("%d) %d x %d = %d\n", i, a[i], b[i], c[i]);
    }

    // TODO 7: Verify all N results on the CPU
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] * b[i]) errors++;
    }
    printf("Verification: %d mismatches out of %d\n", errors, N);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
```

**Cell 2 (Compile and Run):**
```bash
!nvcc -arch=sm_75 ex5_vecmul.cu -o ex5_vecmul && ./ex5_vecmul
```

### Self-Check Table
| Item | Expected Value |
| :--- | :--- |
| Blocks launched | `19532 blocks x 512 threads` |
| Total threads launched | `10,000,384` (384 more than $N$) |
| Active threads in last block | `128` (384 stopped by guard) |
| First of last 1000 results | `9999000) 76 x 52 = 3952` |
| Last result | `9999999) 31 x 73 = 2263` |
| Verification | `0 mismatches out of 10000000` |

---

## Exercise 6 - Multiply Two $10,000 \times 10,000$ Matrices Element by Element

Extend Exercise 5 to a 2D matrix of $10,000 \times 10,000$ integers for $A$, $B$, and $C$. Perform element-wise multiplication: $C[r][c] = A[r][c] \times B[r][c]$.

### Mapping a 2D Grid onto a Matrix
- **Columns ($x$):** `col = blockIdx.x * blockDim.x + threadIdx.x`
- **Rows ($y$):** `row = blockIdx.y * blockDim.y + threadIdx.y`
- **Flat Index:** `idx = row * COLS + col`
- **Threads/Block:** `dim3 threadsPerBlock(32, 16);` (512 threads)
- **Grid Dimensions:** `dim3 numBlocks((COLS + 31)/32, (ROWS + 15)/16);`

**Cell 1 (Write `ex6_matrix.cu` - Fill in TODOs):**
```cpp
%%writefile ex6_matrix.cu
#include <cstdio>
#include <cstdlib>

#define ROWS 10000
#define COLS 10000

// Element-wise product: C[r][c] = A[r][c] * B[r][c]
__global__ void mulElem(int *a, int *b, int *c, int rows, int cols) {
    // TODO 1: Map 2D coordinates
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO 2 & 3: Guard and convert to 1D index
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] * b[idx];
    }
}

void random_ints(int *x, size_t size) {
    for (size_t i = 0; i < size; i++)
        x[i] = rand() % 100;
}

int main(void) {
    int *a, *b, *c;        // Host flat arrays
    int *d_a, *d_b, *d_c;  // Device copies

    size_t count = (size_t)ROWS * COLS;
    size_t size = count * sizeof(int);

    // Allocate GPU memory (400 MB per matrix)
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate host memory
    a = (int*)malloc(size); random_ints(a, count);
    b = (int*)malloc(size); random_ints(b, count);
    c = (int*)malloc(size);

    // Copy to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 32 x 16 = 512 threads per block
    dim3 threadsPerBlock(32, 16);

    // TODO 4: Round up grid dimensions
    dim3 numBlocks((COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Grid: %d x %d blocks\n", numBlocks.x, numBlocks.y);

    // TODO 5: Launch Kernel
    mulElem<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, ROWS, COLS);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    // TODO 6: Copy back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // TODO 7: Print last 1000 elements (C[9999][9000] to C[9999][9999])
    for (int col = 9000; col < 10000; col++) {
        int idx = 9999 * COLS + col;
        printf("C[9999][%d] = %d x %d = %d\n", col, a[idx], b[idx], c[idx]);
    }

    // TODO 8: CPU Verification
    size_t errors = 0;
    for (size_t i = 0; i < count; i++) {
        if (c[i] != a[i] * b[i]) errors++;
    }
    printf("Verification: %zu mismatches out of %zu\n", errors, count);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
```

**Cell 2 (Compile and Run):**
```bash
!nvcc -arch=sm_75 ex6_matrix.cu -o ex6_matrix && ./ex6_matrix
```

### Self-Check Table
| Item | Expected Value |
| :--- | :--- |
| Grid | `Grid: 313 x 625 blocks` (195,625 total blocks) |
| Total threads launched | `100,160,000` (160,000 extra threads) |
| $C[9999][9000]$ | `29 x 56 = 1624` |
| $C[9999][9999]$ | `86 x 21 = 1806` |
| Verification | `0 mismatches out of 100000000` |

---

## Quick Reference API

| Item | Description |
| :--- | :--- |
| `__global__` | Keyword marking a GPU kernel function (called from host CPU, executed on device GPU). Must return `void`. |
| `cudaMalloc((void**)&d_p, size)` | Allocates GPU device memory. |
| `cudaMemcpy(dest, src, size, dir)` | Copies data (`cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`). |
| `cudaFree(d_p)` | Frees GPU memory. |
| `kernel<<<blocks, threads>>>(...)` | Launches kernel grid. Arguments can be integers or `dim3`. |
| `cudaGetLastError()` | Returns status/error code of the last CUDA call or launch. |
| `cudaGetErrorString(err)` | Translates error code to string. |
| `cudaDeviceSynchronize()` | Blocks host CPU until GPU kernel execution completes. |
| `threadIdx.x / .y` | Thread coordinates within its current block. |
| `blockIdx.x / .y` | Block coordinates within the global grid. |
| `blockDim.x / .y` | Number of threads per block along $x$/$y$. |
| `gridDim.x / .y` | Number of blocks in the grid along $x$/$y$. |

---

## Troubleshooting in Colab

| Error / Symptom | Likely Cause & Fix |
| :--- | :--- |
| `nvidia-smi` fails or driver unavailable | No GPU runtime attached. Go to **Runtime → Change runtime type → T4 GPU**. |
| `nvcc: command not found` | Use the full binary path: `/usr/local/cuda/bin/nvcc`. |
| Error/Issue with `%%writefile` | Ensure `%%writefile filename.cu` is on the **very first line** of the cell with no preceding whitespace/comments. |
| Kernel launch error: `no kernel image is available` | `-arch` flag mismatch with GPU capability. Check compute capability using `nvidia-smi`. |
| Kernel launch error: `invalid configuration argument` | Exceeded block limits (>1024 threads) or launched 0 blocks. |
| Code changes don't take effect | You must re-run the `%%writefile` cell **and** the compile cell before re-running executables. |
| Incorrect values at vector ends | Block counts were rounded down instead of up, or bounds check guard is missing/wrong. |
| Session crashed due to RAM limits | Declared large matrices on local stack instead of heap dynamic allocation (`malloc`). Use single flat arrays. |