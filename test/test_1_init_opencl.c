#include "acunit/i.h"

#include "hello.cl.h"
#include "hello_n.cl.h"
#include "hello_n_l.cl.h"
#include "hello.h.h"
#include "../clcontext.h"

#include "BCUnit/BCUnit.h"
#include "BCUnit/Basic.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>


CLContext *ctx;

static int init_suite(void) {
	ctx = malloc(sizeof(CLContext));
	return 0;
}
static int clean_suite(void) {
	free(ctx);
	return 0;
}
static void init_cl_test(void) {
	memset(ctx,0,sizeof(CLContext));
}
static void teardown_cl_test(void) {
	finalize_cl(ctx);
	destroy_cl(ctx);
}

static void test_init_kernel(void) {
	size_t mem_size = 128;

	unsigned char *src[] = (unsigned char*[]){ hello_h,hello_cl};
	size_t size[] = (size_t []){ hello_h_len, hello_cl_len };
	char *names[] = (char *[]){"hello"};
	ctx->kernel.buffer[0] = (BufferInfo ){sizeof(char)* mem_size, CL_MEM_READ_WRITE};
	ctx->kernel.num_src = 2;
	ctx->kernel.num_kernels = 1;
	ctx->kernel.num_buffers = 1;

	pd_init_cl(ctx, src, size, names);
}

static void test_run_hello_kernel(void) {
	size_t mem_size = 128;
	char string[mem_size];
	cl_int ret;

	unsigned char *src[] = (unsigned char*[]){ hello_h,hello_cl};
	size_t size[] = (size_t []){ hello_h_len, hello_cl_len };
	char *names[] = (char *[]){"hello", "world"};
	ctx->kernel.buffer[0] = (BufferInfo){sizeof(string), CL_MEM_READ_WRITE};
	ctx->kernel.num_src = 2;
	ctx->kernel.num_kernels = 2;
	ctx->kernel.num_buffers = 1;
	pd_init_cl(ctx, src, size, names);

	/* this is what I was missing. */

	ret = clEnqueueNDRangeKernel (ctx->clcmdq[0], ctx->clkernel[0][0], 1, NULL, (size_t []){1}, (size_t []){1}, 0, NULL,NULL);
	if(ret != CL_SUCCESS) { 
		CU_FAIL("couldn't run kernel ");
	}

	ret = clEnqueueNDRangeKernel (ctx->clcmdq[0], ctx->clkernel[0][1], 1, NULL, (size_t []){1}, (size_t []){1}, 0, NULL,NULL);
	if(ret != CL_SUCCESS) { 
		CU_FAIL("couldn't run kernel ");
	}

	ret = clEnqueueReadBuffer(ctx->clcmdq[0], ctx->buffers[0][0], CL_TRUE, 0, mem_size * sizeof(char),string, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(ctx->clcmdq[0], ctx->buffers[0][1], CL_TRUE, 0, mem_size * sizeof(char),string, 0, NULL, NULL);

	//puts(string);
	CU_ASSERT_STRING_EQUAL(string, "Hello, World!");
}

static void test_run_hello_n(void) {
	size_t mem_size = 128;
	size_t global_work_size = 13, local_work_size = 13;
	char string[mem_size];
	cl_int ret;

	unsigned char *src[] = (unsigned char*[]){ hello_n_cl};
	size_t size[] = (size_t []){ hello_n_cl_len };
	char *names[] = (char *[]){"hello"};
	ctx->kernel.buffer[0] = (BufferInfo){sizeof(string), CL_MEM_READ_WRITE};
	ctx->kernel.num_src = 1;
	ctx->kernel.num_kernels = 1;
	ctx->kernel.num_buffers = 1;
	pd_init_cl(ctx, src, size, names);

	cl_event ev;
	ret = clEnqueueNDRangeKernel (ctx->clcmdq[0], ctx->clkernel[0][0], 1, NULL, &global_work_size, &local_work_size, 0, NULL,&ev);
	if(ret != CL_SUCCESS) { CU_FAIL("couldn't run kernel ");}

	ret = clEnqueueReadBuffer(ctx->clcmdq[0], ctx->buffers[0][0], CL_TRUE, 0, mem_size * sizeof(char),string, 0, NULL, &ev);

	CU_ASSERT_STRING_EQUAL(string, "Hello, World!");
	//puts(string);
}
static void test_run_hello_l(void) {
	size_t mem_size = 128;
	size_t global_work_size[] = (size_t []){1}, local_work_size[] = (size_t []){1};
	char string[mem_size];
	cl_int ret;

	unsigned char *src[] = (unsigned char*[]){ hello_n_l_cl};
	size_t size[] = (size_t []){ hello_n_l_cl_len };
	char *names[] = (char *[]){"hello"};
	ctx->kernel.num_src = 1;
	ctx->kernel.num_kernels = 1;
	ctx->kernel.num_buffers = 2;
	ctx->kernel.buffer[0] = (BufferInfo){sizeof(string), CL_MEM_READ_WRITE};
	ctx->kernel.buffer[1] = (BufferInfo){sizeof(int), CL_MEM_READ_WRITE, 1 };
	pd_init_cl(ctx, src, size, names);

	int i=0;
	cl_event ev[13];
	ret=clEnqueueNDRangeKernel(ctx->clcmdq[0], ctx->clkernel[0][0], 1, NULL, global_work_size, local_work_size, 0, NULL,&(ev[i]));
	if(ret != CL_SUCCESS) { CU_FAIL("couldn't run kernel ");}

	for(i=1; i<13; i++) {
		ret=clEnqueueNDRangeKernel(ctx->clcmdq[0], ctx->clkernel[0][0], 1, NULL, global_work_size, local_work_size, 1, &(ev[i-1]),&(ev[i]));
		if(ret != CL_SUCCESS) { CU_FAIL("couldn't run kernel ");}
	}

	ret = clEnqueueReadBuffer(ctx->clcmdq[0], ctx->buffers[0][0], CL_TRUE, 0, mem_size * sizeof(char),string, 1, &(ev[i-1]), NULL);

	CU_ASSERT_STRING_EQUAL(string, "Hello, World!");
	puts(string);
}
static CU_TestInfo tests[] = {
	{"Test Init CL Kernels", test_init_kernel},
	{"Test Run Hello Kernel", test_run_hello_kernel},
	{"Test Run Hello N Kernels", test_run_hello_n},
	{"Test Run Hello Local Kernels", test_run_hello_l},
	CU_TEST_INFO_NULL,
};

static CU_SuiteInfo suites[] = {
	{ "CLContext Test Suite", init_suite, clean_suite, init_cl_test,teardown_cl_test, tests },
	CU_SUITE_INFO_NULL,
};

ADD_SUITE(suites);
