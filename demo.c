#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/IndexFlat_c.h>
#include <faiss/c_api/index_factory_c.h>
#include <faiss/c_api/error_c.h>
#include <faiss/c_api/AutoTune_c.h>
#include <faiss/c_api/gpu/GpuIndex_c.h>
#include <faiss/c_api/gpu/StandardGpuResources_c.h>
#include <faiss/c_api/Clustering_c.h>
#include <faiss/c_api/gpu/GpuClonerOptions_c.h>
#include <faiss/c_api/gpu/GpuAutoTune_c.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// FAISS 错误检查宏
#define FAISS_TRY(C) \
	{ \
		if (C) { \
			fprintf(stderr, "FAISS Error: %s\n", faiss_get_last_error()); \
			exit(-1); \
		} \
	}

// CUDA 错误检查宏
#define CUDA_CHECK(C) \
	{ \
		cudaError_t err = (C); \
		if (err != cudaSuccess) { \
			fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
			exit(-1); \
		} \
	}

// cuBLAS 错误检查宏
#define CUBLAS_CHECK(C) \
	{ \
		cublasStatus_t err = (C); \
		if (err != CUBLAS_STATUS_SUCCESS) { \
			fprintf(stderr, "cuBLAS Error: %d\n", err); \
			exit(-1); \
		} \
	}

// 性能测试结果结构体
typedef struct {
	double cpu_time;
	double gpu_load_time;
	double gpu_compute_time;
} PerformanceResult;

// 矩阵操作结果结构体
typedef struct {
	float *l2_distances_cpu;
	faiss_idx_t *l2_labels_cpu;
	float *l2_distances_gpu;
	faiss_idx_t *l2_labels_gpu;
	
	float *ip_distances_cpu;
	faiss_idx_t *ip_labels_cpu;
	float *ip_distances_gpu;
	faiss_idx_t *ip_labels_gpu;
	
	float l2_max_error;
	float ip_max_error;
	
	PerformanceResult l2_perf;
	PerformanceResult ip_perf;
} MatrixOperationResult;

// 矩阵向量乘法结果结构体
typedef struct {
	float *result_cpu_naive;
	float *result_cpu_blas;
	float *result_gpu;
	
	double cpu_naive_time;
	double cpu_blas_time;
	double gpu_load_time;
	double gpu_compute_time;
	
	float max_error_blas;
	float max_error_gpu;
} MatrixVectorMultResult;

// K-means 结果结构体
typedef struct {
	float *centroids_cpu;
	float *centroids_gpu;
	
	double cpu_time;
	double gpu_load_time;
	double gpu_compute_time;
	
	float max_centroid_error;
} KMeansResult;

// 生成随机数据
void generate_random_data(float *data, size_t n, int d) {
	for (size_t i = 0; i < n * d; i++) {
		data[i] = (float)rand() / RAND_MAX;
	}
}

// 向量归一化
void normalize_vectors(float *data, size_t n, int d) {
	for (size_t i = 0; i < n; i++) {
		float norm = 0.0f;
		for (int j = 0; j < d; j++) {
			norm += data[i * d + j] * data[i * d + j];
		}
		norm = sqrtf(norm);
		if (norm > 0) {
			for (int j = 0; j < d; j++) {
				data[i * d + j] /= norm;
			}
		}
	}
}

// 获取当前时间（毫秒）
double get_time_ms() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// 矩阵操作测试 (L2 + IP)
MatrixOperationResult* test_matrix_operation(
	float *xb, int nb,
	float *xq, int nq,
	int d, int k) {
	
	MatrixOperationResult *result = (MatrixOperationResult *)malloc(sizeof(MatrixOperationResult));
	
	// 分配输出缓冲区
	result->l2_distances_cpu = (float *)malloc(nq * k * sizeof(float));
	result->l2_labels_cpu = (faiss_idx_t *)malloc(nq * k * sizeof(faiss_idx_t));
	result->l2_distances_gpu = (float *)malloc(nq * k * sizeof(float));
	result->l2_labels_gpu = (faiss_idx_t *)malloc(nq * k * sizeof(faiss_idx_t));
	
	result->ip_distances_cpu = (float *)malloc(nq * k * sizeof(float));
	result->ip_labels_cpu = (faiss_idx_t *)malloc(nq * k * sizeof(faiss_idx_t));
	result->ip_distances_gpu = (float *)malloc(nq * k * sizeof(float));
	result->ip_labels_gpu = (faiss_idx_t *)malloc(nq * k * sizeof(faiss_idx_t));
	
	printf("\n***** 测试: 向量距离度量对比 (L2 vs IP) *****\n");
	printf("数据规模: %d 个向量, 维度: %d, 查询: %d 个, Top-K: %d\n", nb, d, nq, k);
	
	// ========== 1. L2距离测试 ==========
	printf("\n========== 1. L2 距离 (欧氏距离) ==========\n");
	printf("测试规模: %d 向量库, %d 查询向量\n", nb, nq);
	
	// CPU L2距离
	double start = get_time_ms();
	FaissIndex *index_l2_cpu = NULL;
	FAISS_TRY(faiss_index_factory(&index_l2_cpu, d, "Flat", METRIC_L2));
	FAISS_TRY(faiss_Index_add(index_l2_cpu, nb, xb));
	FAISS_TRY(faiss_Index_search(index_l2_cpu, nq, xq, k, 
								  result->l2_distances_cpu, result->l2_labels_cpu));
	double end = get_time_ms();
	result->l2_perf.cpu_time = end - start;
	printf("CPU 时间: %.3f ms\n", result->l2_perf.cpu_time);
	
	// GPU L2距离
	start = get_time_ms();
	FaissStandardGpuResources *res = NULL;
	FAISS_TRY(faiss_StandardGpuResources_new(&res));
	
	FaissIndex *index_l2_gpu = NULL;
	FaissGpuClonerOptions *options_l2 = NULL;
	FAISS_TRY(faiss_GpuClonerOptions_new(&options_l2));
	FAISS_TRY(faiss_index_cpu_to_gpu_with_options(
		(FaissGpuResourcesProvider*)res, 0, index_l2_cpu, options_l2, &index_l2_gpu));
	
	end = get_time_ms();
	result->l2_perf.gpu_load_time = end - start;
	printf("GPU 加载时间: %.3f ms\n", result->l2_perf.gpu_load_time);
	
	start = get_time_ms();
	FAISS_TRY(faiss_Index_search(index_l2_gpu, nq, xq, k, 
								  result->l2_distances_gpu, result->l2_labels_gpu));
	end = get_time_ms();
	result->l2_perf.gpu_compute_time = end - start;
	printf("GPU 计算时间: %.3f ms\n", result->l2_perf.gpu_compute_time);
	
	// 计算L2误差
	result->l2_max_error = 0.0f;
	for (int i = 0; i < nq * k; i++) {
		float error = fabsf(result->l2_distances_cpu[i] - result->l2_distances_gpu[i]);
		if (error > result->l2_max_error) {
			result->l2_max_error = error;
		}
	}
	printf("L2距离 - CPU vs GPU 最大误差: %f\n", result->l2_max_error);
	
	// ========== 2. 内积 (IP) 测试 ==========
	printf("\n========== 2. 内积 (Inner Product) ==========\n");
	printf("测试规模: %d 向量库, %d 查询向量\n", nb, nq);
	
	// CPU 内积
	start = get_time_ms();
	FaissIndex *index_ip_cpu = NULL;
	FAISS_TRY(faiss_index_factory(&index_ip_cpu, d, "Flat", METRIC_INNER_PRODUCT));
	FAISS_TRY(faiss_Index_add(index_ip_cpu, nb, xb));
	FAISS_TRY(faiss_Index_search(index_ip_cpu, nq, xq, k, 
								  result->ip_distances_cpu, result->ip_labels_cpu));
	end = get_time_ms();
	result->ip_perf.cpu_time = end - start;
	printf("CPU 时间: %.3f ms\n", result->ip_perf.cpu_time);
	
	// GPU 内积
	start = get_time_ms();
	FaissIndex *index_ip_gpu = NULL;
	FaissGpuClonerOptions *options_ip = NULL;
	FAISS_TRY(faiss_GpuClonerOptions_new(&options_ip));
	FAISS_TRY(faiss_index_cpu_to_gpu_with_options(
		(FaissGpuResourcesProvider*)res, 0, index_ip_cpu, options_ip, &index_ip_gpu));
	end = get_time_ms();
	result->ip_perf.gpu_load_time = end - start;
	printf("GPU 加载时间: %.3f ms\n", result->ip_perf.gpu_load_time);
	
	start = get_time_ms();
	FAISS_TRY(faiss_Index_search(index_ip_gpu, nq, xq, k, 
								  result->ip_distances_gpu, result->ip_labels_gpu));
	end = get_time_ms();
	result->ip_perf.gpu_compute_time = end - start;
	printf("GPU 计算时间: %.3f ms\n", result->ip_perf.gpu_compute_time);
	
	// 计算IP误差
	result->ip_max_error = 0.0f;
	for (int i = 0; i < nq * k; i++) {
		float error = fabsf(result->ip_distances_cpu[i] - result->ip_distances_gpu[i]);
		if (error > result->ip_max_error) {
			result->ip_max_error = error;
		}
	}
	printf("内积 - CPU vs GPU 最大误差: %f\n", result->ip_max_error);
	
	// 性能对比总结
	printf("\n╔════════════════════╦═════════════╦═════════════╦═══════════════════╦════════════════════╗\n");
	printf("║ 距离度量		   ║ CPU时间(ms) ║ GPU计算(ms) ║ 加速比(仅计算)	║ 加速比(含加载)	 ║\n");
	printf("╠════════════════════╬═════════════╬═════════════╬═══════════════════╬════════════════════╣\n");
	
	double l2_speedup_compute = result->l2_perf.gpu_compute_time > 0 ? 
								result->l2_perf.cpu_time / result->l2_perf.gpu_compute_time : 0;
	double l2_speedup_total = (result->l2_perf.gpu_load_time + result->l2_perf.gpu_compute_time) > 0 ? 
							  result->l2_perf.cpu_time / (result->l2_perf.gpu_load_time + result->l2_perf.gpu_compute_time) : 0;
	printf("║ L2距离			 ║ %11.3f ║ %11.3f ║ %17.3fx ║ %18.3fx ║\n",
		   result->l2_perf.cpu_time, result->l2_perf.gpu_compute_time, l2_speedup_compute, l2_speedup_total);
	
	double ip_speedup_compute = result->ip_perf.gpu_compute_time > 0 ? 
								result->ip_perf.cpu_time / result->ip_perf.gpu_compute_time : 0;
	double ip_speedup_total = (result->ip_perf.gpu_load_time + result->ip_perf.gpu_compute_time) > 0 ? 
							  result->ip_perf.cpu_time / (result->ip_perf.gpu_load_time + result->ip_perf.gpu_compute_time) : 0;
	printf("║ 内积 (IP)		  ║ %11.3f ║ %11.3f ║ %17.3fx ║ %18.3fx ║\n",
		   result->ip_perf.cpu_time, result->ip_perf.gpu_compute_time, ip_speedup_compute, ip_speedup_total);
	
	printf("╚════════════════════╩═════════════╩═════════════╩═══════════════════╩════════════════════╝\n");
	
	// 清理资源
	faiss_Index_free(index_l2_cpu);
	faiss_Index_free(index_l2_gpu);
	faiss_Index_free(index_ip_cpu);
	faiss_Index_free(index_ip_gpu);
	faiss_StandardGpuResources_free(res);
	faiss_GpuClonerOptions_free(options_l2);
	faiss_GpuClonerOptions_free(options_ip);
	
	return result;
}

// 矩阵向量乘法测试
MatrixVectorMultResult* test_matrix_vector_mult(
	int matrix_size,
	int num_vectors,
	int d) {
	
	printf("\n***** 测试: 矩阵向量乘法 *****\n");
	printf("矩阵大小: %d x %d, 向量个数: %d, 向量维度: %d\n", 
		   matrix_size, matrix_size, num_vectors, d);
	
	MatrixVectorMultResult *result = (MatrixVectorMultResult *)malloc(sizeof(MatrixVectorMultResult));
	
	// 分配内存
	float *matrix = (float *)malloc(matrix_size * matrix_size * sizeof(float));
	float *vectors = (float *)malloc(num_vectors * d * sizeof(float));
	result->result_cpu_naive = (float *)malloc(num_vectors * matrix_size * sizeof(float));
	result->result_cpu_blas = (float *)malloc(num_vectors * matrix_size * sizeof(float));
	result->result_gpu = (float *)malloc(num_vectors * matrix_size * sizeof(float));
	
	// 生成随机数据
	generate_random_data(matrix, matrix_size, matrix_size);
	generate_random_data(vectors, num_vectors, d);
	
	// ========== 1. CPU 暴力计算 ==========
	printf("\n--- 1. CPU 暴力计算 ---\n");
	double start = get_time_ms();
	for (int i = 0; i < num_vectors; i++) {
		for (int j = 0; j < matrix_size; j++) {
			float sum = 0.0f;
			for (int k = 0; k < matrix_size; k++) {
				sum += matrix[j * matrix_size + k] * vectors[i * matrix_size + k];
			}
			result->result_cpu_naive[i * matrix_size + j] = sum;
		}
	}
	double end = get_time_ms();
	result->cpu_naive_time = end - start;
	printf("时间: %.3f ms\n", result->cpu_naive_time);
	
	// ========== 2. CPU BLAS 计算 ==========
	printf("\n--- 2. CPU BLAS 计算 (cblas_sgemm) ---\n");
	start = get_time_ms();
	float alpha = 1.0f, beta = 0.0f;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				num_vectors, matrix_size, matrix_size,
				alpha, vectors, matrix_size,
				matrix, matrix_size,
				beta, result->result_cpu_blas, matrix_size);
	end = get_time_ms();
	result->cpu_blas_time = end - start;
	printf("时间: %.3f ms\n", result->cpu_blas_time);
	
	// ========== 3. GPU CUDA 计算 ==========
	printf("\n--- 3. GPU CUDA 计算 (cuBLAS) ---\n");
	float *d_matrix, *d_vectors, *d_result;
	
	start = get_time_ms();
	CUDA_CHECK(cudaMalloc((void **)&d_matrix, matrix_size * matrix_size * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&d_vectors, num_vectors * matrix_size * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&d_result, num_vectors * matrix_size * sizeof(float)));
	
	CUDA_CHECK(cudaMemcpy(d_matrix, matrix, matrix_size * matrix_size * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_vectors, vectors, num_vectors * matrix_size * sizeof(float), cudaMemcpyHostToDevice));
	end = get_time_ms();
	result->gpu_load_time = end - start;
	printf("GPU 数据加载时间: %.3f ms\n", result->gpu_load_time);
	
	cublasHandle_t handle;
	CUBLAS_CHECK(cublasCreate(&handle));
	
	start = get_time_ms();
	CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
							matrix_size, num_vectors, matrix_size,
							&alpha, d_matrix, matrix_size,
							d_vectors, matrix_size,
							&beta, d_result, matrix_size));
	CUDA_CHECK(cudaDeviceSynchronize());
	end = get_time_ms();
	result->gpu_compute_time = end - start;
	printf("GPU 计算时间: %.3f ms\n", result->gpu_compute_time);
	
	CUDA_CHECK(cudaMemcpy(result->result_gpu, d_result, num_vectors * matrix_size * sizeof(float), cudaMemcpyDeviceToHost));
	
	// 清理 GPU 资源
	CUDA_CHECK(cudaFree(d_matrix));
	CUDA_CHECK(cudaFree(d_vectors));
	CUDA_CHECK(cudaFree(d_result));
	CUBLAS_CHECK(cublasDestroy(handle));
	
	// ========== 性能对比 ==========
	printf("\n========== 性能对比总结 ==========\n");
	printf("CPU 暴力计算:		%.3f ms\n", result->cpu_naive_time);
	printf("CPU BLAS 计算:	   %.3f ms\n", result->cpu_blas_time);
	printf("GPU 加载时间:		%.3f ms\n", result->gpu_load_time);
	printf("GPU 计算时间:		%.3f ms\n", result->gpu_compute_time);
	printf("GPU 总时间:		  %.3f ms\n", result->gpu_load_time + result->gpu_compute_time);
	
	printf("\n========== 加速比 ==========\n");
	if (result->cpu_blas_time > 0) 
		printf("CPU暴力 vs CPU BLAS: %.3fx\n", result->cpu_naive_time / result->cpu_blas_time);
	if (result->gpu_compute_time > 0) 
		printf("CPU BLAS vs GPU计算: %.3fx\n", result->cpu_blas_time / result->gpu_compute_time);
	if (result->gpu_compute_time > 0) 
		printf("CPU暴力 vs GPU计算:  %.3fx\n", result->cpu_naive_time / result->gpu_compute_time);
	if ((result->gpu_load_time + result->gpu_compute_time) > 0) 
		printf("CPU暴力 vs GPU总时间: %.3fx\n", result->cpu_naive_time / (result->gpu_load_time + result->gpu_compute_time));
	
	// ========== 结果验证 ==========
	printf("\n========== 结果验证 ==========\n");
	
	result->max_error_blas = 0.0f;
	result->max_error_gpu = 0.0f;
	
	for (int i = 0; i < num_vectors * matrix_size; i++) {
		float error_blas = fabsf(result->result_cpu_naive[i] - result->result_cpu_blas[i]);
		float error_gpu = fabsf(result->result_cpu_naive[i] - result->result_gpu[i]);
		
		if (error_blas > result->max_error_blas) result->max_error_blas = error_blas;
		if (error_gpu > result->max_error_gpu) result->max_error_gpu = error_gpu;
	}
	
	printf("CPU暴力 vs CPU BLAS 最大误差: %.6f\n", result->max_error_blas);
	printf("CPU BLAS 验证结果: %s\n\n", result->max_error_blas < 1e-3 ? "✓ 正确" : "✗ 存在误差");
	
	printf("CPU暴力 vs GPU BLAS 最大误差: %.6f\n", result->max_error_gpu);
	printf("GPU BLAS 验证结果: %s\n", result->max_error_gpu < 1e-3 ? "✓ 正确" : "✗ 存在误差");
	
	// 详细验证前几个元素
	printf("\n前 5 个结果详细对比:\n");
	printf("索引 | CPU暴力	  | CPU BLAS	 | GPU CUDA	 |CPU BLAS误差   | GPU BLAS误差\n");
	printf("-----|--------------|--------------|--------------|------------|----------\n");
	int show_count = (num_vectors * matrix_size < 5) ? num_vectors * matrix_size : 5;
	for (int i = 0; i < show_count; i++) {
		printf("%4d | %12.6f | %12.6f | %12.6f | %10.6f | %10.6f\n",
			   i,
			   result->result_cpu_naive[i],
			   result->result_cpu_blas[i],
			   result->result_gpu[i],
			   fabsf(result->result_cpu_naive[i] - result->result_cpu_blas[i]),
			   fabsf(result->result_cpu_naive[i] - result->result_gpu[i]));
	}
	
	// 清理内存
	free(matrix);
	free(vectors);
	
	return result;
}

// ========== K-means 聚类测试 ==========
KMeansResult* test_kmeans(float *x, int n, int d, int k_clusters) {
	printf("\n\n***** 测试: K-means 聚类 *****\n");
	printf("数据规模: %d 个向量, 维度: %d, 聚类数: %d\n", n, d, k_clusters);
	
	KMeansResult *result = (KMeansResult *)malloc(sizeof(KMeansResult));

	result->centroids_cpu = (float *)malloc(k_clusters * d * sizeof(float));
	result->centroids_gpu = (float *)malloc(k_clusters * d * sizeof(float));
	
	// CPU K-means
	printf("\n--- 1. CPU K-means 聚类 ---\n");
	double start = get_time_ms();
	
	FaissIndex *index_cpu = NULL;
	FAISS_TRY(faiss_index_factory(&index_cpu, d, "Flat", METRIC_L2));
	FAISS_TRY(faiss_Index_add(index_cpu, n, x));
	
	FaissClustering *clustering_cpu = NULL;
	FAISS_TRY(faiss_Clustering_new(&clustering_cpu, d, k_clusters));
	FAISS_TRY(faiss_Clustering_train(clustering_cpu, n, x, index_cpu));
	
	double end = get_time_ms();
	result->cpu_time = end - start;
	printf("时间: %.3f ms\n", result->cpu_time);
	
	// GPU K-means
	printf("\n--- 2. GPU K-means 聚类 ---\n");
	
	double start_load = get_time_ms();
	FaissStandardGpuResources *res = NULL;
	FAISS_TRY(faiss_StandardGpuResources_new(&res));
	
	FaissIndex *index_gpu_base = NULL;
	FAISS_TRY(faiss_index_factory(&index_gpu_base, d, "Flat", METRIC_L2));
	
	FaissIndex *index_gpu = NULL;
	FaissGpuClonerOptions *options = NULL;
	FAISS_TRY(faiss_GpuClonerOptions_new(&options));
	FAISS_TRY(faiss_index_cpu_to_gpu_with_options(
		(FaissGpuResourcesProvider*)res, 0, index_gpu_base, options, &index_gpu));
	
	double end_load = get_time_ms();
	result->gpu_load_time = end_load - start_load;
	printf("GPU 加载时间: %.3f ms\n", result->gpu_load_time);
	
	double start_compute = get_time_ms();
	FaissClustering *clustering_gpu = NULL;
	FAISS_TRY(faiss_Clustering_new(&clustering_gpu, d, k_clusters));
	FAISS_TRY(faiss_Clustering_train(clustering_gpu, n, x, index_gpu));
	double end_compute = get_time_ms();
	result->gpu_compute_time = end_compute - start_compute;
	printf("GPU 计算时间: %.3f ms\n", result->gpu_compute_time);
	
	// 性能对比
	printf("\n========== 性能对比总结 ==========\n");
	printf("CPU 时间:			%.3f ms\n", result->cpu_time);
	printf("GPU 数据加载时间:	%.3f ms\n", result->gpu_load_time);
	printf("GPU 计算时间:		%.3f ms\n", result->gpu_compute_time);
	printf("GPU 总时间:		  %.3f ms\n", result->gpu_load_time + result->gpu_compute_time);
	
	printf("\n========== 加速比 ==========\n");
	if (result->gpu_compute_time > 0) 
		printf("加速比 (仅计算):	 %.3fx\n", result->cpu_time / result->gpu_compute_time);
	if ((result->gpu_load_time + result->gpu_compute_time) > 0) 
		printf("加速比 (含加载):	 %.3fx\n", result->cpu_time / (result->gpu_load_time + result->gpu_compute_time));
	
	// 清理资源
	faiss_Clustering_free(clustering_cpu);
	faiss_Clustering_free(clustering_gpu);
	faiss_Index_free(index_cpu);
	faiss_Index_free(index_gpu_base);
	faiss_Index_free(index_gpu);
	faiss_StandardGpuResources_free(res);
	faiss_GpuClonerOptions_free(options);
	
	return result;
}

// ========== 主函数 ==========
int main() {
	srand(time(NULL));
	
	int d = 128;	 // dimension
	int nb = 10000000; // database size
	int nq = 10000;  // nb of queries
	
	float* xb = malloc(d * nb * sizeof(float));
	float* xq = malloc(d * nq * sizeof(float));

	generate_random_data(xb, nb, d);
	generate_random_data(xq, nq, d);

	
	printf("===== Faiss GPU vs CPU 性能对比测试 =====\n");
	
	// 测试矩阵操作
	MatrixOperationResult *mat_result = test_matrix_operation(xb, nb, xq, nq, d, 10);
	
	// 测试矩阵乘向量
	MatrixVectorMultResult *mv_result = test_matrix_vector_mult(1536, 1, 1536);
	
	// 测试 K-means
	KMeansResult *kmeans_result = test_kmeans(xb, nb, d, 20000);
	
	// 清理资源
	free(xb);
	free(xq);
	
	// 清理结果结构体
	if (mat_result) {
		free(mat_result->l2_distances_cpu);
		free(mat_result->l2_labels_cpu);
		free(mat_result->l2_distances_gpu);
		free(mat_result->l2_labels_gpu);
		free(mat_result->ip_distances_cpu);
		free(mat_result->ip_labels_cpu);
		free(mat_result->ip_distances_gpu);
		free(mat_result->ip_labels_gpu);
		free(mat_result);
	}
	
	if (mv_result) {
		free(mv_result->result_cpu_naive);
		free(mv_result->result_cpu_blas);
		free(mv_result->result_gpu);
		free(mv_result);
	}
	
	if (kmeans_result) {
		free(kmeans_result->centroids_cpu);
		free(kmeans_result->centroids_gpu);
		free(kmeans_result);
	}
	
	printf("\n\n===== 所有测试完成 =====\n");
	
	return 0;
}

