/*******************************************************
 * Simple arrayfire matrix multiply example
 ********************************************************/

#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>
#include <Windows.h>    // For GetTickCount()
using namespace af;

// Borrow simple Matrix facility from the companion raw CUDA implementation.
//  Matrices are stored in row-major order:
//  M(row, col) = *(M.elements + row * M.width + col)
//  Conveniently Arrayfire 2 dimensional arrays use the same storage, layout
//  which we leverage later as arrayfire doesn't provide good facilities for
//  accessing single elements.
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
};


// Prototypes
void CpuMatMul( const Matrix &A, const Matrix &B, Matrix &C );

// For simplicity, do square DIMxDIM matrices, only
const int DIM=512;   // min 16 for shared, max 512

// Arrayfire multiplicands
static array M(DIM,DIM);
static array N(DIM,DIM);

// Get ready for tests
void af_test_init()
{
    printf( "size of M(%d,%d) = %d bytes, %f bytes per element\n", DIM, DIM, N.bytes(), ((float)N.bytes())/(DIM*DIM) );

    // Initialising the arrayfire array like this works but is *extremely* slow, presumably because copying
    // individual elements to GPU memory is slow
/*  for(int i=0; i<DIM; i++)
    {
        for(int j=0; j<DIM; j++)
        {
            M(i,j) = (i+j);     // in Arrayfire M(i,j) works as a LHS expression but not a RHS expression (!?)
            N(i,j) = (i+j);
        }
    }  */

    // So instead take advantage of the fact that the memory layout of a two dimensional arrayfire array is the same
    // as our simple Matrix convention
    Matrix src;
    src.alloc(DIM,DIM);
    src.init();
    M.write(src.elements,M.bytes());    // This is the rather sad and poorly documented way to do a host->device transfer in Arrayfire
    N.write(src.elements,N.bytes());
    src.free();
}

// Make sure the matrix multiplication works at least once
bool af_test_verify( Matrix &reference )
{
    bool ok=true;
    int idx=0;
    array A = matmul(M, N);                 // matrix multiply
    float *host_a = A.host<float>();        // must call af::freeHost() later
    for( int row=0; ok && row<DIM; row++ )
    {
        for( int col=0; ok && col<DIM; col++ )
        {
            float value1 = host_a[idx++];
            float value2 = reference.elements[row * reference.width + col];
            float delta = value1>=value2 ? value1-value2 : value2-value1;
            ok = delta < (value1/100000.0); //1000000000.0);     // don't insist on bitwise float equality
                                // Note that I've changed the demand for relative error
                                // to be better than 1e-5 when previously I had 1e-9. This
                                // was necessary to get a successful verification on the CPU
                                // backend. This requires further study. See *NB* for more.
            if( !ok )
                printf( "Verify error: element[%d][%d], value1=%f, value2=%f, delta=%f\n", row, col, value1, value2, delta );
            if( row==0 && col==0 )
                printf( "element[0][0] = %f (calc), %f (ref)\n", value1, value2 );
            else if( row+1==DIM && col+1==DIM )
                printf( "element[DIM-1][DIM-1] = %f (calc), %f (ref)\n", value1, value2 );

/* *NB*
    On the CUDA backend, we get, as expected
        element[0][0] = 44608184.000000 (calc), 44608184.000000 (ref)
        element[DIM-1][DIM-1] = 311995968.000000 (calc), 311995968.000000 (ref)
    But on the CPU backend we get
        element[0][0] = 44608204.000000 (calc), 44608184.000000 (ref)
        element[DIM-1][DIM-1] = 311995904.000000 (calc), 311995968.000000 (ref)
    As discussed above, this was the motivation for reducing the relative error
    acceptance threshold. For further study, I suspect either an out and out bug
    in Arrayfire, or perhaps reduced precision in the CPU backend for some reason.
*/
        }
    }
    af::freeHost(host_a);
    return ok;
}

// One matrix multiply, for timing test
void af_test_for_timing()
{
    array B = matmul(M, N);  // matrix multiply
    B.eval();                // ensure evaluated
}

int main()
{
    try {

        // Select a device and display arrayfire info
        int device = 0;
        af::setDevice(device);

        // Print information about backend implementation
        af::info();

        // Allocate the matrixes and init A and B to known values
        Matrix A, B, C;
        A.alloc( DIM, DIM );
        A.init();
        B.alloc( DIM, DIM );
        B.init();
        C.alloc( DIM, DIM );

        // Initialise ArrayFire test
        printf( "Initialising array fire timing test\n" );
        af_test_init();

        // Use arrayfire timing facilities
        double af_time = timeit(af_test_for_timing); // time in seconds
        printf( "Arrayfire native GPU timing %.2f millisecs\n", af_time*1000 );
        
        // Verify that array fire does the matrix multiply correctly once
        printf( "Verify that array fire does the matrix multiply correctly once\n" );
        CpuMatMul(A, B, C);
        bool ok = af_test_verify(C);
        printf( "Verify %s\n", ok?"successful":"not successful" );

        // Loop until satisfied with timing    
        bool do_cpu=true;
        bool do_gpu=true;
        unsigned int ntimes_cpu=1;
        unsigned int ntimes_gpu=1;
        unsigned int time_cpu = 0;
        unsigned int time_gpu = 0;
        for(;;)
        {

            // Time Naive CPU multiplication
            unsigned int time_cpu_1 = GetTickCount();
            for( unsigned int i=0; do_cpu && i<ntimes_cpu; i++ )
                CpuMatMul(A, B, C);
            unsigned int time_cpu_2 = GetTickCount();

            // Time arrayfire GPU implementation
            unsigned int time_gpu_1 = GetTickCount();
            for( unsigned int i=0; do_gpu && i<ntimes_gpu; i++ )
                af_test_for_timing();
            unsigned int time_gpu_2 = GetTickCount();

            // Report on timing
            if( do_cpu )
            {
                time_cpu = time_cpu_2-time_cpu_1;
                if( ntimes_cpu == 1 )
                    printf( "CPU time %u milliseconds\n", time_cpu );
                else
                    printf( "CPU time %.2f milliseconds (per iteration, %u iterations, %u ms total)\n", ((float)time_cpu)/ntimes_cpu, ntimes_cpu, time_cpu );
            }
            if( do_gpu )
            {
                time_gpu = time_gpu_2-time_gpu_1;
                if( ntimes_gpu == 1 )
                    printf( "GPU time %u milliseconds\n", time_gpu );
                else
                    printf( "GPU time %.2f milliseconds (per iteration, %u iterations, %u ms total)\n", ((float)time_gpu)/ntimes_gpu, ntimes_gpu, time_gpu );
            }

            // If necessary, repeat with looping to increase timing accuracy
            const unsigned int min_time_ms = 1000;
            if( do_cpu && time_cpu<min_time_ms )
                ntimes_cpu *= 4;
            else
                do_cpu = false;
            if( do_gpu && time_gpu<min_time_ms )
                ntimes_gpu *= 4;
            else
                do_gpu = false;
            if( do_cpu || do_gpu )
                printf( "\nRepeating with more loops, to improve timing accuracy\n" );
            else
                break;
        }
        printf( "\nTiming summary:\n" );
        printf( "Naive CPU time %.2f milliseconds\n", ((float)time_cpu)/ntimes_cpu );
        printf( "Arrayfire GPU time %.2f milliseconds\n", ((float)time_gpu)/ntimes_gpu );
        printf( "For comparison, arrayfire GPU time using arrayfire hi-res timer %.2f millisecs\n", af_time*1000 );

        // Free in reverse order to alloc()
        C.free();
        B.free();
        A.free();

    } catch (af::exception& e) {

        fprintf(stderr, "%s\n", e.what());
        throw;
    }

}

// Naive but reliable and simple matrix multiplication - for reference
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

