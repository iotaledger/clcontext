#ifndef CLCONTEXT_H
#define CLCONTEXT_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CLCONTEXT_MAX_DEVICES (16)
#define MAX_PLATFORMS (8)
#define MAX_KERNELS (18)
//#define MAX_PROGRAMS (18)
#define MAX_BUFFERS (18)

enum cl_flag_t {
	CLCURL_SUCCESS,
	CLCURL_FAIL
};

typedef struct {
	size_t index;
	cl_bool blocking;
	size_t offset;
	size_t size;
	void *val;
	cl_uint num_wl;
	const cl_event *wl;
	cl_event *ev;
} BufferVal;

typedef struct {
	size_t index;
	size_t dimensions;
	const size_t *global_offset;
	const size_t *global_size;
	const size_t *local_size;
	cl_uint num_wl;
	const cl_event *wl;
	cl_event *ev;
} KernelVal;

typedef struct {
	size_t size;
	cl_mem_flags flags;
	int local;
} BufferInfo;

typedef struct {
	unsigned char **src;
	size_t *size;
	char ** names;
	BufferInfo *buffer;
	size_t num_buffers;
	size_t num_kernels;
	size_t num_src;
} KernelInfo;

typedef struct {
	cl_uint num_devices;
	cl_uint num_programs;
	cl_command_queue clcmdq[CLCONTEXT_MAX_DEVICES];
	cl_mem buffers[CLCONTEXT_MAX_DEVICES][MAX_BUFFERS];
	cl_kernel clkernel[CLCONTEXT_MAX_DEVICES][MAX_KERNELS];
	cl_program programs[CLCONTEXT_MAX_DEVICES];
	cl_context clctx[CLCONTEXT_MAX_DEVICES];
	KernelInfo kernel;
} CLContext;



void init_cl(CLContext *ctx);
int init_kernel(CLContext *ctx);
void destroy_cl(CLContext *ctx);
void finalize_cl(CLContext *ctx);
void write_buffers(CLContext *ctx, size_t device, size_t num_buffers, BufferVal *args);
void read_buffers(CLContext *ctx, size_t device, size_t num_buffers, BufferVal *args);
void run_kernel(CLContext *ctx, size_t device, size_t index, KernelVal *args);

#endif
