#include <fstream>
#include <iostream>
#include <string>
#include <ios>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

typedef struct
{
    cl_uchar R;
    cl_uchar G;
    cl_uchar B;
    cl_uchar align;
} RGB;

typedef struct
{
    bool type;
    cl_uint size;
    cl_uint height;
    cl_uint weight;
    RGB *data;
} Image;

Image *readbmp(const char *filename)
{
    std::ifstream bmp(filename, std::ios::binary);
    char header[54];
    bmp.read(header, 54);
    cl_uint size = *(int *)&header[2];
    cl_uint offset = *(int *)&header[10];
    cl_uint w = *(int *)&header[18];
    cl_uint h = *(int *)&header[22];
    uint16_t depth = *(uint16_t *)&header[28];
    if (depth != 24 && depth != 32)
    {
        printf("we don't suppot depth with %d\n", depth);
        exit(0);
    }
    bmp.seekg(offset, bmp.beg);

    Image *ret = new Image();
    ret->type = 1;
    ret->height = h;
    ret->weight = w;
    ret->size = w * h;
    ret->data = new RGB[w * h]{};
    for (int i = 0; i < ret->size; i++)
    {
        bmp.read((char *)&ret->data[i], depth / 8);
    }
    return ret;
}

int writebmp(const char *filename, Image *img)
{

    cl_uchar header[54] = {
        0x42,        // identity : B
        0x4d,        // identity : M
        0, 0, 0, 0,  // file size
        0, 0,        // reserved1
        0, 0,        // reserved2
        54, 0, 0, 0, // RGB data offset
        40, 0, 0, 0, // struct BITMAPINFOHEADER size
        0, 0, 0, 0,  // bmp width
        0, 0, 0, 0,  // bmp height
        1, 0,        // planes
        32, 0,       // bit per pixel
        0, 0, 0, 0,  // compression
        0, 0, 0, 0,  // data size
        0, 0, 0, 0,  // h resolution
        0, 0, 0, 0,  // v resolution
        0, 0, 0, 0,  // used colors
        0, 0, 0, 0   // important colors
    };

    // file size
    cl_uint file_size = img->size * 4 + 54;
    header[2] = (unsigned char)(file_size & 0x000000ff);
    header[3] = (file_size >> 8) & 0x000000ff;
    header[4] = (file_size >> 16) & 0x000000ff;
    header[5] = (file_size >> 24) & 0x000000ff;

    // width
    cl_uint width = img->weight;
    header[18] = width & 0x000000ff;
    header[19] = (width >> 8) & 0x000000ff;
    header[20] = (width >> 16) & 0x000000ff;
    header[21] = (width >> 24) & 0x000000ff;

    // height
    cl_uint height = img->height;
    header[22] = height & 0x000000ff;
    header[23] = (height >> 8) & 0x000000ff;
    header[24] = (height >> 16) & 0x000000ff;
    header[25] = (height >> 24) & 0x000000ff;

    std::ofstream fout;
    fout.open(filename, std::ios::binary);
    fout.write((char *)header, 54);
    fout.write((char *)img->data, img->size * 4);
    fout.close();
}

cl_program load_program(cl_context context, cl_device_id device, const char* filename)
{
    std::ifstream in(filename, std::ios_base::binary);
    if(!in.good()) {
        return 0;
    }

    // get file length
    in.seekg(0, std::ios_base::end);
    size_t length = in.tellg();
    in.seekg(0, std::ios_base::beg);

    // read program source
    std::vector<char> data(length + 1);
    in.read(&data[0], length);
    data[length] = 0;

    // create and build program 
    const char* source = &data[0];
    cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
    
    if(program == 0) {
        std::cerr << "Can't load program\n";
        return 0;
    }

    if(clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
        std::cerr << "Can't build program\n";

        cl_int err;
        size_t len = 0;
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char *buffer=(char*)malloc(len);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        if(err == CL_SUCCESS){
            std::cerr << buffer;
        }

        return 0;
    }
    

    return program;
}

int main(int argc, char *argv[])
{
    char *filename;
    if (argc >= 2)
    {
        //get platforms
        cl_int err;
        cl_uint num;
        err = clGetPlatformIDs(0, 0, &num);
        if(err != CL_SUCCESS) {
            std::cerr << "Unable to get platforms\n";
            return 0;
        }

        std::vector<cl_platform_id> platforms(num);
        err = clGetPlatformIDs(num, &platforms[0], &num);
        if(err != CL_SUCCESS) {
            std::cerr << "Unable to get platform ID\n";
            return 0;
        }

        //create  context
        cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
        cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
        if(context == 0) {
            std::cerr << "Can't create OpenCL context\n";
            return 0;
        }

        //get context info
        size_t cb;
        clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
        std::vector<cl_device_id> devices(cb / sizeof(cl_device_id));
        clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

        //get device info
        clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
        std::string devname;
        devname.resize(cb);
        clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);

        //create command queue
        cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, 0);
        if(queue == 0) {
            std::cerr << "Can't create command queue\n";
            clReleaseContext(context);
            return 0;
        }

        int many_img = argc - 1;
        for (int i = 0; i < many_img; i++)
        {
            filename = argv[i + 1];
            Image *img = readbmp(filename);

            std::cout << img->weight << ":" << img->height << "\n";

            cl_uint R[256];
            cl_uint G[256];
            cl_uint B[256];
            std::fill(R, R+256, 0);
            std::fill(G, G+256, 0);
            std::fill(B, B+256, 0);

            //alloc memory
            cl_mem cl_R = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * 256, &R[0], NULL);
            cl_mem cl_G = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * 256, &G[0], NULL);
            cl_mem cl_B = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * 256, &B[0], NULL);
            cl_mem cl_img = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Image), img, NULL);
            cl_mem cl_pixels = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(RGB) * img->weight * img-> height, img->data, NULL);
            if(cl_R == 0 || cl_G == 0 || cl_B == 0 || cl_img == 0 || cl_pixels == 0) {
                std::cerr << "Can't create OpenCL buffer\n";
                clReleaseMemObject(cl_R);
                clReleaseMemObject(cl_G);
                clReleaseMemObject(cl_B);
                clReleaseMemObject(cl_img);
                clReleaseMemObject(cl_pixels);
                continue;
            }

            //move data from host to device
            err = clEnqueueWriteBuffer(queue, cl_R, CL_TRUE, 0, sizeof(cl_uint) * 256, &R[0], 0, NULL, NULL);
            err += clEnqueueWriteBuffer(queue, cl_G, CL_TRUE, 0, sizeof(cl_uint) * 256, &G[0], 0, NULL, NULL);
            err += clEnqueueWriteBuffer(queue, cl_B, CL_TRUE, 0, sizeof(cl_uint) * 256, &B[0], 0, NULL, NULL);
            err += clEnqueueWriteBuffer(queue, cl_img, CL_TRUE, 0, sizeof(Image), img, 0, NULL, NULL);
            err += clEnqueueWriteBuffer(queue, cl_pixels, CL_TRUE, 0, sizeof(RGB) * img->weight * img-> height, img->data, 0, NULL, NULL);
            if(err != CL_SUCCESS) {
                std::cerr << "Can't move data to devices\n";
                std::cerr << err << "\n";
                clReleaseMemObject(cl_R);
                clReleaseMemObject(cl_G);
                clReleaseMemObject(cl_B);
                clReleaseMemObject(cl_img);
                clReleaseMemObject(cl_pixels);
                continue;
            }

            //load program
            cl_program program = load_program(context, devices[0], "histogram.cl");
            if(program == 0) {
                clReleaseMemObject(cl_R);
                clReleaseMemObject(cl_G);
                clReleaseMemObject(cl_B);
                clReleaseMemObject(cl_img);
                clReleaseMemObject(cl_pixels);
                continue;
            }

            //create kernel
            cl_kernel hist = clCreateKernel(program, "histogram", 0);
            if(hist == 0) {
                std::cerr << "Can't load kernel\n";
                clReleaseProgram(program);
                clReleaseMemObject(cl_R);
                clReleaseMemObject(cl_G);
                clReleaseMemObject(cl_B);
                clReleaseMemObject(cl_img);
                clReleaseMemObject(cl_pixels);
                continue;
            }

            //set argument
            clSetKernelArg(hist, 2, sizeof(cl_mem), &cl_R);
            clSetKernelArg(hist, 3, sizeof(cl_mem), &cl_G);
            clSetKernelArg(hist, 4, sizeof(cl_mem), &cl_B);
            clSetKernelArg(hist, 0, sizeof(cl_mem), &cl_img);
            clSetKernelArg(hist, 1, sizeof(cl_mem), &cl_pixels);

            //run kernel
            //size_t localws[2] = {16,16} ; 
            size_t globalws[2] = {img->weight, img->height};
            err = clEnqueueNDRangeKernel(queue, hist, 2, 0, globalws, 0, 0, 0, 0);
    
            //move data from device to host
            if(err == CL_SUCCESS) {
                err = clEnqueueReadBuffer(queue, cl_R, CL_TRUE, 0, sizeof(cl_uint) * 256, &R[0], 0, 0, 0);
                err += clEnqueueReadBuffer(queue, cl_G, CL_TRUE, 0, sizeof(cl_uint) * 256, &G[0], 0, 0, 0);
                err += clEnqueueReadBuffer(queue, cl_B, CL_TRUE, 0, sizeof(cl_uint) * 256, &B[0], 0, 0, 0);
            }
            else{
                std::cerr << "Can't run kernel\n";
                std::cerr << err << "\n";
                clReleaseKernel(hist);
                clReleaseProgram(program);
                clReleaseMemObject(cl_R);
                clReleaseMemObject(cl_G);
                clReleaseMemObject(cl_B);
                clReleaseMemObject(cl_img);
                clReleaseMemObject(cl_pixels);
                continue;
            }

            //output histogram
            if(err == CL_SUCCESS) {
                int max = 0;
                for(int i=0;i<256;i++){
                    max = R[i] > max ? R[i] : max;
                    max = G[i] > max ? G[i] : max;
                    max = B[i] > max ? B[i] : max;
                    // printf("%d %d %d\n", R[i], G[i], B[i]);
                }

                Image *ret = new Image();
                ret->type = 1;
                ret->height = 256;
                ret->weight = 256;
                ret->size = 256 * 256;
                ret->data = new RGB[256 * 256];

                for(int i=0;i<ret->height;i++){
                    for(int j=0;j<256;j++){
                        if(R[j]*256/max > i)
                            ret->data[256*i+j].R = 255;
                        if(G[j]*256/max > i)
                            ret->data[256*i+j].G = 255;
                        if(B[j]*256/max > i)
                            ret->data[256*i+j].B = 255;
                    }
                }

                std::string newfile = "hist_" + std::string(filename); 
                writebmp(newfile.c_str(), ret);
            }
            else{
                std::cerr << "Can't read back data\n";
                std::cerr << err << "\n";
            }

            clReleaseKernel(hist);
            clReleaseProgram(program);
            clReleaseMemObject(cl_R);
            clReleaseMemObject(cl_G);
            clReleaseMemObject(cl_B);
            clReleaseMemObject(cl_img);
            clReleaseMemObject(cl_pixels);
        }

        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }else{
        printf("Usage: ./hist <img.bmp> [img2.bmp ...]\n");
    }
    return 0;
}
