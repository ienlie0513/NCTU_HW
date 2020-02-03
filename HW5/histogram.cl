typedef struct
{
    uchar R;
    uchar G;
    uchar B;
    uchar align;
} RGB;

typedef struct
{
    bool type;
    uint size;
    uint height;
    uint weight;
    RGB *data;
} Image;


__kernel void histogram(__global Image *img,
    __global RGB *pixels,
    __global uint R[256],
    __global uint G[256],
    __global uint B[256])
{
    const long unsigned int ix = get_global_id(0);
    const long unsigned int iy = get_global_id(1);

    RGB pixel = pixels[iy*(img->weight)+ix];
    atomic_add(&R[pixel.R], 1);
    atomic_add(&G[pixel.G], 1);
    atomic_add(&B[pixel.B], 1);

}
