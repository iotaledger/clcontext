#include "clcontext.h"
#include <stdio.h>
#include <string.h>

/*
#define DEBUG
*/


static void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data){
	fprintf(stderr, "W: caught an error in ocl_pfn_notify:\nW: %s", errinfo);
}
void check_clerror(cl_int err, char *comment, ...) {
	if(err == CL_SUCCESS) {
		return;
	}
	printf("E: OpenCL implementation returned an error: %d\n", err);
	va_list args;
	vprintf(comment, args);
	printf("\n\n");
	exit(1);
}

static void get_devices(CLContext *ctx, unsigned char **src, size_t *size) {
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
		ctx->programs[i] = clCreateProgramWithSource(ctx->clctx[i], ctx->kernel.num_src, (const char**)src, size, &errno);
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

static void create_kernel (CLContext *ctx,char **names) {
	// Create kernel.
	size_t i, j;
	for(i=0; i< ctx->num_devices; i++) {
		for(j=0;j< ctx->kernel.num_kernels; j++) {
			cl_int errno;
			ctx->clkernel[i][j] = clCreateKernel(ctx->programs[i], names[j], &errno);
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
					errno = clSetKernelArg(ctx->clkernel[i][k], j, sizeof(cl_mem), NULL);
				} else {
					errno = clSetKernelArg(ctx->clkernel[i][k], j, sizeof(cl_mem),(void *)&(ctx->buffers[i][j]));
				}
				check_clerror(errno, "Failed to execute clCreateBuffer for %d:%d",i,j);
			}
		}
	}
}


int init_kernel(CLContext *ctx, char **names) {
	create_kernel(ctx, names);
	kernel_init_buffers(ctx);
	return 0;
}


void pd_init_cl(CLContext *ctx, unsigned char **src, size_t *size,char **names){
	if(!ctx) {
		ctx = malloc(sizeof(CLContext));
	}
	get_devices(ctx, src, size);
	init_kernel(ctx, names);
}

void destroy_cl(CLContext *ctx) {
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
