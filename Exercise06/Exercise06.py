%%writefile ex6_matrix.cu
#include <cstdio>
#include <cstdlib>
 
#define ROWS 10000
#define COLS 10000
 
// Element-wise product: C[r][c] = A[r][c] * B[r][c]  (NOT matrix multiplication)
__global__ void mulElem(int *a, int *b, int *c, int rows, int cols) {
    // TODO 1: row comes from the y dimension, col from the x dimension
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO 2: only compute if BOTH row and col are inside the matrix
    // TODO 3: convert (row, col) into a flat index: row * cols + col
    if(row < rows && col < cols){
        int idx = row * cols + col;
        c[idx] = a[idx] * b[idx];
    }
}
 
void random_ints(int *x, size_t size) {
    for (size_t i = 0; i < size; i++)
        x[i] = rand() % 100;
}
 
int main(void) {
    int *a, *b, *c;             // host copies (flat arrays, row-major)
    int *d_a, *d_b, *d_c;       // device copies
    size_t count = (size_t)ROWS * COLS;
    size_t size  = count * sizeof(int);         // 400 MB per matrix
 
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
 
    a = (int *)malloc(size); random_ints(a, count);
    b = (int *)malloc(size); random_ints(b, count);
    c = (int *)malloc(size);
 
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
 
    // 32 x 16 = 512 threads per block
    dim3 threadsPerBlock(32, 16);
    // TODO 4: blocks needed in x (columns) and in y (rows), rounding UP
    dim3 numBlocks(
        (COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    printf("Grid: %d x %d blocks\n", numBlocks.x, numBlocks.y);
 
    // TODO 5: launch mulElem with <<<numBlocks, threadsPerBlock>>>
    mulElem<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c,ROWS, COLS);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
 
    // TODO 6: copy d_c back into c on the host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
 
    // TODO 7: print C[9999][9000] ... C[9999][9999]  (the last 1000 elements)
    for(int col = 9000; col < 10000; col++){
        int idx = 9999 * COLS + col;
        printf("C[9999][%d] = %d x %d = %d\n",col,a[idx],b[idx],c[idx]);
    }
    
    // TODO 8: verify all elements on the CPU and print the number of mismatches
    long long errors = 0;

    for(size_t i = 0; i < count; i++){
        if(c[i] != a[i] * b[i])
            errors++;
    }

    printf("Verification: %lld mismatches out of %zu\n",errors,count);
    
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
} 