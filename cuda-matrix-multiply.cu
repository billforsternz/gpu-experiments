/*******************************************************
 * Simple raw CUDA matrix multiply example
 ********************************************************/

/*
 * Based on multShare.c and multNoShare.c by Robert Hochberg, 2012
 * From http://www.shodor.org/petascale/materials/UPModules/matrixMultiplication/
 * Which in turn was based nearly entirely on the code from the CUDA C Programming Guide
 */

#include <stdio.h>
#include <Windows.h>    // For GetTickCount()

// Define a debug only version of printf, so we don't printf in time sensitive places
#ifdef _DEBUG
    #define dprintf(...) printf(__VA_ARGS__)
#else
    #define dprintf(...)
#endif

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
struct Matrix
{
    int width;
    int height;
    int stride;
    float* elements;

    void alloc( int height_p, int width_p )
    {
        height = height_p;
        width = width_p;
        elements = (float*)malloc(width * height * sizeof(float));
        if( !elements )
        {
            printf( "Panic: memory allocation error" );
            exit(-1);
        }
    }
    void free()
    {
        ::free(elements);
    }

    // Initialise to a known pattern for reproducability
    void init()
    {
        for( int i=0; i<height; i++ )
            for( int j=0; j<width; j++ )
                elements[i*width + j] = (float)(i+j);
    }

    // Print up to a 4x4 portion of the matrix
    void print()
    {
        for(int i = 0; i < min(4, height); i++)
        {
            for(int j = 0; j < min(4, width); j++)
                printf("%f ", elements[i*width + j]);
            printf("\n");
        }
    }
};

// Thread block size
#define BLOCK_SIZE 16

__global__ void MatMulKernelShared( Matrix, Matrix, Matrix );
__global__ void MatMulKernelSimple(const Matrix, const Matrix, Matrix);

void CpuMatMul( const Matrix &A, const Matrix &B, Matrix &C );
void MatMul( bool shared, const Matrix &A, const Matrix &B, Matrix &C, bool overhead_only=false );
bool IsEquals( const Matrix &C, const Matrix &D );

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE

void MatMul( bool shared, const Matrix &A, const Matrix &B, Matrix &C, bool overhead_only )
{

    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaError_t err = cudaMalloc(&d_A.elements, size);
    dprintf("CUDA malloc A: %s\n",cudaGetErrorString(err));
    err = cudaMemcpy(d_A.elements, A.elements,size, cudaMemcpyHostToDevice);
    dprintf("Copy A to device: %s\n",cudaGetErrorString(err));

    Matrix d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    err = cudaMalloc(&d_B.elements, size);
    dprintf("CUDA malloc B: %s\n",cudaGetErrorString(err));
    err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    dprintf("Copy B to device: %s\n",cudaGetErrorString(err));

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    err = cudaMalloc(&d_C.elements, size);
    dprintf("CUDA malloc C: %s\n",cudaGetErrorString(err));

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    if( shared )
    {
        dim3 dimGrid( B.width  / dimBlock.x, A.height / dimBlock.y );
        if( !overhead_only )    // overhead_only is set when we want to time host-device memory copy only
        {
            MatMulKernelShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
            err = cudaThreadSynchronize();
        }
    }
    else
    {
        dim3 dimGrid( (B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y );
        if( !overhead_only )
        {
            MatMulKernelSimple<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
            err = cudaThreadSynchronize();
        }
    }
    dprintf("Run kernel: %s\n", cudaGetErrorString(err));
    
    // Read C from device memory
    err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    dprintf("Copy C off of device: %s\n",cudaGetErrorString(err));
    
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul(), simple version
__global__ void MatMulKernelSimple(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float value = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= A.height || col >= B.width)
        return;
    for (int e = 0; e < A.width; ++e)
        value += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);
    C.elements[row * C.width + col] = value;
}

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// Matrix multiplication kernel called by MatMul(), shared version
__global__ void MatMulKernelShared( Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0.0;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m)
    {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
    
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
    
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
    
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
    
        // Multiply Asub and Bsub together
        for (int e = 0;  e < BLOCK_SIZE;  ++e)
            Cvalue += As[row][e] * Bs[e][col];
    
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

// For simplicity, do square DIMxDIM matrices, only
const int DIM=512;   // min 16 for shared, max 512

int main()
{
    // Allocate the matrixes and init A and B to known values
    Matrix A, B, C, D, E, F;
    A.alloc( DIM, DIM );
    A.init();
    B.alloc( DIM, DIM );
    B.init();
    C.alloc( DIM, DIM );
    D.alloc( DIM, DIM );
    E.alloc( DIM, DIM );
    F.alloc( DIM, DIM );

    // Loop until satisfied with timing    
    bool do_cpu=true;
    bool do_overhead=true;
    bool do_simple=true;
    bool do_shared=true;
    unsigned int ntimes_cpu=1;
    unsigned int ntimes_overhead=1;
    unsigned int ntimes_simple=1;
    unsigned int ntimes_shared=1;
    unsigned int time_cpu = 0;
    unsigned int time_simple = 0;
    unsigned int time_shared = 0;
    unsigned int time_overhead = 0;
    for(;;)
    {

        // Set base reference in matrix C with CPU multiplication
        unsigned int time_cpu_1 = GetTickCount();
        for( unsigned int i=0; do_cpu && i<ntimes_cpu; i++ )
            CpuMatMul(A, B, C);
        unsigned int time_cpu_2 = GetTickCount();

        // Test simple GPU implementation
        unsigned int time_simple_1 = GetTickCount();
        for( unsigned int i=0; do_simple && i<ntimes_simple; i++ )
            MatMul( false, A, B, D );
        unsigned int time_simple_2 = GetTickCount();
        if( do_simple )
        {
            bool okay = IsEquals(C,D);
            if( okay )
                printf( "Success simple GPU and CPU matrix multiplication match\n" );
            else
                printf( "Error simple GPU and CPU matrix multiplication don't match\n" );
        }

        // Test shared memory implementation
        unsigned int time_shared_1 = GetTickCount();
        for( unsigned int i=0; do_shared && i<ntimes_shared; i++ )
            MatMul( true, A, B, E );
        unsigned int time_shared_2 = GetTickCount();
        if( do_shared )
        {
            bool okay = IsEquals(C,E);
            if( okay )
                printf( "Success shared memory GPU and CPU matrix multiplication match\n" );
            else
                printf( "Error shared memory GPU and CPU matrix multiplication don't match\n" );

        }
        // Calculate GPU overhead time (can be subtracted from GPU times, to get core GPU times) 
        unsigned int time_overhead_1 = GetTickCount();
        for( unsigned int i=0; do_overhead && i<ntimes_overhead; i++ )
            MatMul( false, A, B, F, true );
        unsigned int time_overhead_2 = GetTickCount();

        // Report on timing
        if( do_cpu )
        {
            time_cpu = time_cpu_2-time_cpu_1;
            if( ntimes_cpu == 1 )
                printf( "CPU time %u milliseconds\n", time_cpu );
            else
                printf( "CPU time %.2f milliseconds (per iteration, %u iterations, %u ms total)\n", ((float)time_cpu)/ntimes_cpu, ntimes_cpu, time_cpu );
        }
        if( do_simple )
        {
            time_simple = time_simple_2-time_simple_1;
            if( ntimes_simple == 1 )
                printf( "Simple GPU time %u milliseconds\n", time_simple );
            else
                printf( "Simple GPU time %.2f milliseconds (per iteration, %u iterations, %u ms total)\n", ((float)time_simple)/ntimes_simple, ntimes_simple, time_simple );
        }
        if( do_shared )
        {
            time_shared = time_shared_2-time_shared_1;
            if( ntimes_shared == 1 )
                printf( "Shared GPU time %u milliseconds\n", time_shared );
            else
                printf( "Shared GPU time %.2f milliseconds (per iteration, %u iterations, %u ms total)\n", ((float)time_shared)/ntimes_shared, ntimes_shared, time_shared );
        }
        if( do_overhead )
        {
            time_overhead = time_overhead_2-time_overhead_1;
            if( ntimes_overhead == 1 )
                printf( "Overhead time %u milliseconds\n", time_overhead );
            else
                printf( "Overhead time %.2f milliseconds (per iteration, %u iterations, %u ms total)\n", ((float)time_overhead)/ntimes_overhead, ntimes_overhead, time_overhead );
        }

        // If necessary, repeat with looping to increase timing accuracy
        const unsigned int min_time_ms = 1000;
        if( do_cpu && time_cpu<min_time_ms )
            ntimes_cpu *= 4;
        else
            do_cpu=false;
        if( do_simple && time_simple<min_time_ms )
            ntimes_simple *= 4;
        else
            do_simple=false;
        if( do_shared &&  time_shared<min_time_ms )
            ntimes_shared *= 4;
        else
            do_shared=false;
        if( do_overhead && time_overhead<min_time_ms )
            ntimes_overhead *= 4;
        else
            do_overhead=false;
        if( do_cpu || do_simple || do_shared || do_overhead )
            printf( "\nRepeating with more loops, to improve timing accuracy\n" );
        else
            break;
    }

    printf( "\nTiming summary:\n" );
    printf( "Naive CPU calculation time %.2f milliseconds\n", ((float)time_cpu)/ntimes_cpu );
    printf( "Simple GPU time %.2f milliseconds\n", ((float)time_simple)/ntimes_simple );
    printf( "Simple GPU time minus host<->device overhead %.2f milliseconds\n", ((float)time_simple)/ntimes_simple - ((float)time_overhead)/ntimes_overhead  );
    printf( "Shared GPU time %.2f milliseconds\n", ((float)time_shared)/ntimes_shared );
    printf( "Shared GPU time minus host<->device overhead %.2f milliseconds\n", ((float)time_shared)/ntimes_shared - ((float)time_overhead)/ntimes_overhead );

    // Print up to a 4x4 portion of the matrices
    printf( "\nMatrix A (all matrices top left 4x4 portion only):\n" );
    A.print();
    printf( "\nMatrix B:\n" );
    B.print();
    printf( "\nMatrix C = A*B (CPU):\n" );
    C.print();
    printf( "\nMatrix D = A*B (GPU, simple):\n" );
    D.print();
    printf( "\nMatrix E = A*B (GPU, shared memory):\n" );
    E.print();

    // Free in reverse order to alloc()
    F.free();
    E.free();
    D.free();
    C.free();
    B.free();
    A.free();

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}

void CpuMatMul( const Matrix &A, const Matrix &B, Matrix &C )
{
    for( int row=0; row<A.height; row++ )
    {
        for( int col=0; col<A.width; col++ )
        {
            float value=0;
            for( int e=0; e<A.width; ++e )
                value += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);
            C.elements[row * C.width + col] = value;
        }
    }
}

bool IsEquals( const Matrix &C, const Matrix &D )
{
    for( int row=0; row<C.height; row++ )
    {
        for( int col=0; col<C.width; col++ )
        {
            float value1 = C.elements[row * C.width + col];
            float value2 = D.elements[row * C.width + col];
            float delta = value1>=value2 ? value1-value2 : value2-value1;
            bool ok = delta < (value1/1000000000.0);     // don't insist on bitwise float equality
            if( !ok )
                return false;
        }
    }
    return true;
}

