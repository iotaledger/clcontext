#include "clcontext.h"
#include <stdio.h>
#include <string.h>

/*
#define DEBUG
*/

static void kernel_init_buffers (CLContext *ctx);
static void create_kernel (CLContext *ctx);
static void get_devices(CLContext *ctx);

static void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data){
	fprintf(stderr, "W: caught an error in ocl_pfn_notify:\nW: %s", errinfo);
}
static void check_clerror(cl_int err, char *comment, ...) {
	if(err == CL_SUCCESS) {
		return;
	}
	printf("E: OpenCL implementation returned an error: %d\n", err);
	va_list args;
	vprintf(comment, args);
	printf("\n\n");
	exit(1);
}

static void get_devices(CLContext *ctx) {
	/* List devices for each platforms.*/ 
	cl_uint num_platforms;
	ctx->num_devices = 0;
	cl_device_id devices[CLCONTEXT_MAX_DEVICES];
	cl_platform_id platforms[MAX_PLATFORMS];

	check_clerror(clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms), "Failed to execute clGetPlatformIDs.");
	if(num_platforms > MAX_PLATFORMS) {
		fprintf(stderr, "W: The number of platforms available on your system exceeds MAX_PLATFORMS. Consider increasing MAX_PLATFORMS.\n");
		num_platforms = MAX_PLATFORMS;
	}
	for(size_t i=0; i< num_platforms; i++) {
		cl_uint pf_num_devices;
		check_clerror(
				clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, CLCONTEXT_MAX_DEVICES-ctx->num_devices, &devices[ctx->num_devices], &pf_num_devices),
				"Failed to execute clGetDeviceIDs for platform id = %zd.", i);
		if(pf_num_devices > CLCONTEXT_MAX_DEVICES-ctx->num_devices) {
			fprintf(stderr, "W: The number of devices available on your system exceeds CLCONTEXT_MAX_DEVICES. Consider increasing CLCONTEXT_MAX_DEVICES.\n");
			pf_num_devices = CLCONTEXT_MAX_DEVICES - ctx->num_devices;
		}
		ctx->num_devices += pf_num_devices;
	}


	/* Create OpenCL context. */
	for(size_t i=0; i< ctx->num_devices; i++) {
		cl_int errno;
		ctx->clctx[i] = clCreateContext(NULL, 1, &(devices[i]), pfn_notify, NULL, &errno);
		check_clerror(errno, "Failed to execute clCreateContext.");
	}
	/* Create command queue */
	for(size_t i=0; i< ctx->num_devices; i++) {
		cl_int errno;
#ifndef CL_VERSION_2_0
		/* For OpenCL version < 2.0 */
		ctx->clcmdq[i] = clCreateCommandQueue(ctx->clctx[i], devices[i], 0, &errno);
#else
		/* For OpenCL version >= 2.0 */
		ctx->clcmdq[i] = clCreateCommandQueueWithProperties(ctx->clctx[i], devices[i], 0, &errno);
#endif
		check_clerror(errno, "Failed to execute clCreateCommandQueueWithProperties.");
	}

	if(ctx->kernel.num_src == 0) return;
	for(size_t i=0; i< ctx->num_devices; i++) {
		cl_int errno;
		ctx->programs[i] = clCreateProgramWithSource(ctx->clctx[i], ctx->kernel.num_src, (const char **)ctx->kernel.src, ctx->kernel.size, &errno);
		check_clerror(errno, "Failed to execute clCreateProgramWithSource");
	}


	for(size_t i=0; i< ctx->num_devices; i++) {
		cl_int errno ;
		errno = clBuildProgram(ctx->programs[i], 0, NULL, NULL, NULL, NULL);
		char *build_log = malloc(0xFFFF);
		size_t log_size;
		clGetProgramBuildInfo(ctx->programs[i], devices[i], CL_PROGRAM_BUILD_LOG, 0xFFFF, build_log, &log_size);
		free(build_log);
		check_clerror(errno, "Failed to execute clBuildProgram");
	}
}

static void create_kernel (CLContext *ctx) {
	// Create kernel.
	size_t i, j;
	for(i=0; i< ctx->num_devices; i++) {
		for(j=0;j< ctx->kernel.num_kernels; j++) {
			cl_int errno;
			ctx->clkernel[i][j] = clCreateKernel(ctx->programs[i], ctx->kernel.names[j], &errno);
			check_clerror(errno, "Failed to execute clCreateKernel");
		}
	}
}

static void kernel_init_buffers (CLContext *ctx) {
	int i, j, k;
	for(i=0; i< ctx->num_devices; i++) {
		cl_int errno;

		for(j=0;j< ctx->kernel.num_buffers;j++) {
			ctx->buffers[i][j] = clCreateBuffer(ctx->clctx[i], ctx->kernel.buffer[j].flags, ctx->kernel.buffer[j].size, NULL, &errno);
			for(k=0;k< ctx->kernel.num_kernels;k++) {
				if(ctx->kernel.buffer[j].local > 0) {
					fprintf(stderr, "\nI: Kernel Local Variable found on, d%d-k%d-b%d",i,k,j);
					errno = clSetKernelArg(ctx->clkernel[i][k], j, sizeof(cl_mem), NULL);
				} else {
					errno = clSetKernelArg(ctx->clkernel[i][k], j, sizeof(cl_mem),(void *)&(ctx->buffers[i][j]));
				}
				check_clerror(errno, "Failed to execute clCreateBuffer for %d:%d",i,j);
			}
		}
	}
}

void write_buffers(CLContext *ctx, size_t device, size_t num_buffers, BufferVal *args) {
	size_t i;
	for(i=0; i < num_buffers; i++) {
		clEnqueueWriteBuffer(ctx->clcmdq[device], ctx->buffers[device][args[i].index], 
				args[i].blocking, args[i].offset, args[i].size, args[i].val, args[i].num_wl, args[i].wl, args[i].ev);
	}
}
void read_buffers(CLContext *ctx, size_t device, size_t num_buffers, BufferVal *args) {
	size_t i;
	for(i=0; i < num_buffers; i++) {
		clEnqueueReadBuffer(ctx->clcmdq[device], ctx->buffers[device][args[i].index], 
				args[i].blocking, args[i].offset, args[i].size, args[i].val, args[i].num_wl, args[i].wl, args[i].ev);
	}
}

void run_kernel(CLContext *ctx, size_t device, size_t num_kernels, KernelVal *args) {
		cl_int errno;
		size_t i;
		for(i = 0; i < num_kernels; i++) {
			errno = clEnqueueNDRangeKernel (ctx->clcmdq[device], 
					ctx->clkernel[device][args[i].index], args[i].dimensions, 
					args[i].global_offset, args[i].global_size, args[i].local_size, args[i].num_wl,
					args[i].wl, args[i].ev);
			//fprintf(stderr, "\nE: Invalid kernel on device %zu kernel %zu\n", device, args[i].index);
			if(errno == CL_INVALID_KERNEL)
				check_clerror(errno, "Failed to execute clEnqueueNDRangeKernel. Dev:%zu, Index: \n", device, args[i].index);

		}
}

int init_kernel(CLContext *ctx) {
	create_kernel(ctx);
	kernel_init_buffers(ctx);
	return 0;
}

void init_cl(CLContext *ctx) {
	if(!ctx) {
		ctx = malloc(sizeof(CLContext));
	}
	get_devices(ctx);
	init_kernel(ctx);
}

void destroy_cl(CLContext *ctx) {
	/*
	free(&(ctx->kernel));
	free(ctx->kernel.src);
	free(ctx->kernel.size);
	free(ctx->kernel.names);
	free(ctx->kernel.buffer);
	free(ctx->kernel.src);
	free(ctx->kernel.size);
	free(ctx->kernel.names);
	free(ctx->kernel.buffer);
	free(ctx->clcmdq);
	cl_mem buffers[CLCONTEXT_MAX_DEVICES][MAX_BUFFERS];
	cl_kernel clkernel[CLCONTEXT_MAX_DEVICES][MAX_KERNELS];
	cl_program programs[CLCONTEXT_MAX_DEVICES];
	cl_context clctx[CLCONTEXT_MAX_DEVICES];
	KernelInfo kernel;
	unsigned char **src;
	const size_t *size;
	char ** names;
	BufferInfo *buffer;
	size_t num_buffers;
	size_t num_kernels;
	size_t num_src;
	*/
}

void finalize_cl(CLContext *ctx) {
	size_t i,j;
	for(i=0; i< ctx->num_devices; i++) {
		clFlush(ctx->clcmdq[i]);
		clFinish(ctx->clcmdq[i]);
		if(&(ctx->kernel) !=NULL){
			for(j=0; j< ctx->kernel.num_kernels; j++) {
				clReleaseKernel(ctx->clkernel[i][j]);
			}	
			if(ctx->kernel.num_src > 0)
				clReleaseProgram(ctx->programs[i]);
			for(j=0; j < ctx->kernel.num_buffers;j++) {
				clReleaseMemObject(ctx->buffers[i][j]);
			}
		}
		clReleaseCommandQueue(ctx->clcmdq[i]);
		clReleaseContext(ctx->clctx[i]);
	}
}
