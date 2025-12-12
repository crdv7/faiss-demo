#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/Clustering.h>
#include <cblas.h>
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;
using namespace chrono;

void generate_random_data(float* data, size_t n, int d) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> dis(0.0f, 1.0f);
	for (size_t i = 0; i < n * d; i++) {
		data[i] = dis(gen);
	}
}

void normalize_vectors(float* data, size_t n, int d) {
	for (size_t i = 0; i < n; i++) {
		float norm = 0.0f;
		for (int j = 0; j < d; j++) {
			norm += data[i * d + j] * data[i * d + j];
		}
		norm = sqrt(norm);
		if (norm > 0) {
			for (int j = 0; j < d; j++) {
				data[i * d + j] /= norm;
			}
		}
	}
}

void print_performance(const string& operation, double cpu_time, 
					  double gpu_load_time, double gpu_compute_time) {
	double gpu_total = gpu_load_time + gpu_compute_time;
	double speedup = (cpu_time > 0 && gpu_compute_time > 0) ? (cpu_time / gpu_compute_time) : 0;
	
	cout << "\n========== " << operation << " ==========\n";
	cout << fixed << setprecision(3);
	cout << "CPU 时间:		  " << cpu_time << " ms\n";
	cout << "GPU 数据加载时间:  " << gpu_load_time << " ms\n";
	cout << "GPU 计算时间:	  " << gpu_compute_time << " ms\n";
	cout << "GPU 总时间:		" << gpu_total << " ms\n";
	if (speedup > 0) {
		cout << "加速比 (仅计算):   " << speedup << "x\n";
		cout << "加速比 (含加载):   " << (cpu_time / gpu_total) << "x\n";
	}
}

void test_matrix_operation(int nb = 100000, int nq = 1000, int d = 256, int k = 10) {
	cout << "\n***** 测试: 向量距离度量对比 (L2 vs IP vs Cosine) *****\n";
	cout << "数据规模: " << nb << " 个向量, 维度: " << d << ", 查询: " << nq << " 个, Top-K: " << k << "\n";
	
	vector<float> xb(nb * d), xq(nq * d);
	generate_random_data(xb.data(), nb, d);
	generate_random_data(xq.data(), nq, d);
	
	cout << fixed << setprecision(3);
	
	// ========== 1. L2距离测试 ==========
	cout << "\n\n========== 1. L2 距离 (欧氏距离) ==========\n";
	cout << "测试规模: " << nb << " 向量库, " << nq << " 查询向量\n";
	
	vector<faiss::idx_t> labels_l2_cpu(nq * k);
	vector<float> distances_l2_cpu(nq * k);
	vector<faiss::idx_t> labels_l2_gpu(nq * k);
	vector<float> distances_l2_gpu(nq * k);
	
	// CPU L2距离
	auto start_l2_cpu = high_resolution_clock::now();
	faiss::IndexFlatL2 index_l2_cpu(d);
	index_l2_cpu.add(nb, xb.data());
	index_l2_cpu.search(nq, xq.data(), k, distances_l2_cpu.data(), labels_l2_cpu.data());
	auto end_l2_cpu = high_resolution_clock::now();
	double l2_cpu_time = duration<double, milli>(end_l2_cpu - start_l2_cpu).count();
	
	// GPU L2距离
	faiss::gpu::StandardGpuResources res;
	auto start_l2_gpu_load = high_resolution_clock::now();
	faiss::gpu::GpuIndexFlatL2 index_l2_gpu(&res, d);
	index_l2_gpu.add(nb, xb.data());
	auto end_l2_gpu_load = high_resolution_clock::now();
	double l2_gpu_load_time = duration<double, milli>(end_l2_gpu_load - start_l2_gpu_load).count();
	
	auto start_l2_gpu_compute = high_resolution_clock::now();
	index_l2_gpu.search(nq, xq.data(), k, distances_l2_gpu.data(), labels_l2_gpu.data());
	auto end_l2_gpu_compute = high_resolution_clock::now();
	double l2_gpu_compute_time = duration<double, milli>(end_l2_gpu_compute - start_l2_gpu_compute).count();
	
	cout << "CPU 时间:           " << l2_cpu_time << " ms\n";
	cout << "GPU 加载时间:       " << l2_gpu_load_time << " ms\n";
	cout << "GPU 计算时间:       " << l2_gpu_compute_time << " ms\n";
	cout << "GPU 总时间:         " << (l2_gpu_load_time + l2_gpu_compute_time) << " ms\n";
	if (l2_gpu_compute_time > 0) {
		cout << "加速比 (仅计算):    " << (l2_cpu_time / l2_gpu_compute_time) << "x\n";
		cout << "加速比 (含加载):    " << (l2_cpu_time / (l2_gpu_load_time + l2_gpu_compute_time)) << "x\n";
	}
	
	cout << "\nL2距离 Top-5 结果样本 (第1个查询):\n";
	cout << "  CPU: ";
	for (int j = 0; j < min(5, k); j++) {
		cout << "idx=" << labels_l2_cpu[j] << " dist=" << distances_l2_cpu[j] << " | ";
	}
	cout << "\n  GPU: ";
	for (int j = 0; j < min(5, k); j++) {
		cout << "idx=" << labels_l2_gpu[j] << " dist=" << distances_l2_gpu[j] << " | ";
	}
	cout << "\n";
	
	// ========== 2. 内积 (IP) 测试 ==========
	cout << "\n\n========== 2. 内积 (Inner Product) ==========\n";
	cout << "测试规模: " << nb << " 向量库, " << nq << " 查询向量\n";
	
	vector<faiss::idx_t> labels_ip_cpu(nq * k);
	vector<float> distances_ip_cpu(nq * k);
	vector<faiss::idx_t> labels_ip_gpu(nq * k);
	vector<float> distances_ip_gpu(nq * k);
	
	// CPU 内积
	auto start_ip_cpu = high_resolution_clock::now();
	faiss::IndexFlatIP index_ip_cpu(d);
	index_ip_cpu.add(nb, xb.data());
	index_ip_cpu.search(nq, xq.data(), k, distances_ip_cpu.data(), labels_ip_cpu.data());
	auto end_ip_cpu = high_resolution_clock::now();
	double ip_cpu_time = duration<double, milli>(end_ip_cpu - start_ip_cpu).count();
	
	// GPU 内积
	auto start_ip_gpu_load = high_resolution_clock::now();
	faiss::gpu::GpuIndexFlatIP index_ip_gpu(&res, d);
	index_ip_gpu.add(nb, xb.data());
	auto end_ip_gpu_load = high_resolution_clock::now();
	double ip_gpu_load_time = duration<double, milli>(end_ip_gpu_load - start_ip_gpu_load).count();
	
	auto start_ip_gpu_compute = high_resolution_clock::now();
	index_ip_gpu.search(nq, xq.data(), k, distances_ip_gpu.data(), labels_ip_gpu.data());
	auto end_ip_gpu_compute = high_resolution_clock::now();
	double ip_gpu_compute_time = duration<double, milli>(end_ip_gpu_compute - start_ip_gpu_compute).count();
	
	cout << "CPU 时间:           " << ip_cpu_time << " ms\n";
	cout << "GPU 加载时间:       " << ip_gpu_load_time << " ms\n";
	cout << "GPU 计算时间:       " << ip_gpu_compute_time << " ms\n";
	cout << "GPU 总时间:         " << (ip_gpu_load_time + ip_gpu_compute_time) << " ms\n";
	if (ip_gpu_compute_time > 0) {
		cout << "加速比 (仅计算):    " << (ip_cpu_time / ip_gpu_compute_time) << "x\n";
		cout << "加速比 (含加载):    " << (ip_cpu_time / (ip_gpu_load_time + ip_gpu_compute_time)) << "x\n";
	}
	
	cout << "\n内积 Top-5 结果样本 (第1个查询):\n";
	cout << "  CPU: ";
	for (int j = 0; j < min(5, k); j++) {
		cout << "idx=" << labels_ip_cpu[j] << " sim=" << distances_ip_cpu[j] << " | ";
	}
	cout << "\n  GPU: ";
	for (int j = 0; j < min(5, k); j++) {
		cout << "idx=" << labels_ip_gpu[j] << " sim=" << distances_ip_gpu[j] << " | ";
	}
	cout << "\n";
	
	// ========== 3. 余弦距离 (Cosine) 测试 ==========
	cout << "\n\n========== 3. 余弦相似度 (Cosine Similarity) ==========\n";
	cout << "测试规模: " << nb << " 向量库, " << nq << " 查询向量\n";
	cout << "说明: 通过 L2归一化 + 内积 实现\n";
	
	// 复制数据用于归一化
	vector<float> xb_normalized = xb;
	vector<float> xq_normalized = xq;
	
	// 对向量进行 L2 归一化
	normalize_vectors(xb_normalized.data(), nb, d);
	normalize_vectors(xq_normalized.data(), nq, d);
	
	vector<faiss::idx_t> labels_cos_cpu(nq * k);
	vector<float> distances_cos_cpu(nq * k);
	vector<faiss::idx_t> labels_cos_gpu(nq * k);
	vector<float> distances_cos_gpu(nq * k);
	
	// CPU 余弦相似度
	auto start_cos_cpu = high_resolution_clock::now();
	faiss::IndexFlatIP index_cos_cpu(d);
	index_cos_cpu.add(nb, xb_normalized.data());
	index_cos_cpu.search(nq, xq_normalized.data(), k, distances_cos_cpu.data(), labels_cos_cpu.data());
	auto end_cos_cpu = high_resolution_clock::now();
	double cos_cpu_time = duration<double, milli>(end_cos_cpu - start_cos_cpu).count();
	
	// GPU 余弦相似度
	auto start_cos_gpu_load = high_resolution_clock::now();
	faiss::gpu::GpuIndexFlatIP index_cos_gpu(&res, d);
	index_cos_gpu.add(nb, xb_normalized.data());
	auto end_cos_gpu_load = high_resolution_clock::now();
	double cos_gpu_load_time = duration<double, milli>(end_cos_gpu_load - start_cos_gpu_load).count();
	
	auto start_cos_gpu_compute = high_resolution_clock::now();
	index_cos_gpu.search(nq, xq_normalized.data(), k, distances_cos_gpu.data(), labels_cos_gpu.data());
	auto end_cos_gpu_compute = high_resolution_clock::now();
	double cos_gpu_compute_time = duration<double, milli>(end_cos_gpu_compute - start_cos_gpu_compute).count();
	
	cout << "CPU 时间:           " << cos_cpu_time << " ms\n";
	cout << "GPU 加载时间:       " << cos_gpu_load_time << " ms\n";
	cout << "GPU 计算时间:       " << cos_gpu_compute_time << " ms\n";
	cout << "GPU 总时间:         " << (cos_gpu_load_time + cos_gpu_compute_time) << " ms\n";
	if (cos_gpu_compute_time > 0) {
		cout << "加速比 (仅计算):    " << (cos_cpu_time / cos_gpu_compute_time) << "x\n";
		cout << "加速比 (含加载):    " << (cos_cpu_time / (cos_gpu_load_time + cos_gpu_compute_time)) << "x\n";
	}
	
	cout << "\n余弦相似度 Top-5 结果样本 (第1个查询):\n";
	cout << "  CPU: ";
	for (int j = 0; j < min(5, k); j++) {
		cout << "idx=" << labels_cos_cpu[j] << " sim=" << distances_cos_cpu[j] << " | ";
	}
	cout << "\n  GPU: ";
	for (int j = 0; j < min(5, k); j++) {
		cout << "idx=" << labels_cos_gpu[j] << " sim=" << distances_cos_gpu[j] << " | ";
	}
	cout << "\n";
	
	// ========== 性能对比总结 ==========
	cout << "\n\n";
	cout << "╔════════════════════╦═════════════╦═════════════╦═══════════════════╦════════════════════╗\n";
	cout << "║ 距离度量           ║ CPU时间(ms) ║ GPU计算(ms) ║ 加速比(仅计算)    ║ 加速比(含加载)     ║\n";
	cout << "╠════════════════════╬═════════════╬═════════════╬═══════════════════╬════════════════════╣\n";
	
	cout << "║ L2距离             ║ " << setw(11) << l2_cpu_time 
	     << " ║ " << setw(11) << l2_gpu_compute_time 
	     << " ║ " << setw(17) << (l2_gpu_compute_time > 0 ? l2_cpu_time / l2_gpu_compute_time : 0) << "x"
	     << " ║ " << setw(18) << (l2_gpu_compute_time > 0 ? l2_cpu_time / (l2_gpu_load_time + l2_gpu_compute_time) : 0) << "x ║\n";
	
	cout << "║ 内积 (IP)          ║ " << setw(11) << ip_cpu_time 
	     << " ║ " << setw(11) << ip_gpu_compute_time 
	     << " ║ " << setw(17) << (ip_gpu_compute_time > 0 ? ip_cpu_time / ip_gpu_compute_time : 0) << "x"
	     << " ║ " << setw(18) << (ip_gpu_compute_time > 0 ? ip_cpu_time / (ip_gpu_load_time + ip_gpu_compute_time) : 0) << "x ║\n";
	
	cout << "║ 余弦相似度 (Cos)   ║ " << setw(11) << cos_cpu_time 
	     << " ║ " << setw(11) << cos_gpu_compute_time 
	     << " ║ " << setw(17) << (cos_gpu_compute_time > 0 ? cos_cpu_time / cos_gpu_compute_time : 0) << "x"
	     << " ║ " << setw(18) << (cos_gpu_compute_time > 0 ? cos_cpu_time / (cos_gpu_load_time + cos_gpu_compute_time) : 0) << "x ║\n";
	
	cout << "╚════════════════════╩═════════════╩═════════════╩═══════════════════╩════════════════════╝\n";
	
	// ========== 结果验证 ==========
	cout << "\n========== 结果验证 ==========\n";
	
	// L2 验证
	float l2_max_error = 0.0f;
	for (int i = 0; i < nq * k; i++) {
		float error = fabs(distances_l2_cpu[i] - distances_l2_gpu[i]);
		l2_max_error = max(l2_max_error, error);
	}
	cout << "L2距离 - CPU vs GPU 最大误差: " << l2_max_error << "\n";
	
	// IP 验证
	float ip_max_error = 0.0f;
	for (int i = 0; i < nq * k; i++) {
		float error = fabs(distances_ip_cpu[i] - distances_ip_gpu[i]);
		ip_max_error = max(ip_max_error, error);
	}
	cout << "内积 - CPU vs GPU 最大误差: " << ip_max_error << "\n";
	
	// Cosine 验证
	float cos_max_error = 0.0f;
	for (int i = 0; i < nq * k; i++) {
		float error = fabs(distances_cos_cpu[i] - distances_cos_gpu[i]);
		cos_max_error = max(cos_max_error, error);
	}
	cout << "余弦相似度 - CPU vs GPU 最大误差: " << cos_max_error << "\n";
	
	// ========== 性能分析 ==========
	cout << "\n========== 性能分析 ==========\n";
	cout << "三种距离度量中最快的CPU实现: ";
	double min_cpu_time = min({l2_cpu_time, ip_cpu_time, cos_cpu_time});
	if (min_cpu_time == l2_cpu_time) {
		cout << "L2距离 (" << l2_cpu_time << " ms)\n";
	} else if (min_cpu_time == ip_cpu_time) {
		cout << "内积 (" << ip_cpu_time << " ms)\n";
	} else {
		cout << "余弦相似度 (" << cos_cpu_time << " ms)\n";
	}
	
	cout << "三种距离度量中最快的GPU实现: ";
	double min_gpu_compute = min({l2_gpu_compute_time, ip_gpu_compute_time, cos_gpu_compute_time});
	if (min_gpu_compute == l2_gpu_compute_time) {
		cout << "L2距离 (" << l2_gpu_compute_time << " ms)\n";
	} else if (min_gpu_compute == ip_gpu_compute_time) {
		cout << "内积 (" << ip_gpu_compute_time << " ms)\n";
	} else {
		cout << "余弦相似度 (" << cos_gpu_compute_time << " ms)\n";
	}
	
	cout << "总体最高加速比: ";
	double max_speedup = max({
		(l2_gpu_compute_time > 0 ? l2_cpu_time / l2_gpu_compute_time : 0),
		(ip_gpu_compute_time > 0 ? ip_cpu_time / ip_gpu_compute_time : 0),
		(cos_gpu_compute_time > 0 ? cos_cpu_time / cos_gpu_compute_time : 0)
	});
	cout << max_speedup << "x\n";
	
	cout << "\n建议:\n";
	if (max_speedup > 10) {
		cout << "✓ GPU加速效果显著，建议使用GPU进行计算\n";
	} else if (max_speedup > 2) {
		cout << "~ GPU加速效果一般，可根据数据规模选择\n";
	} else {
		cout << "✗ GPU加速效果不明显，建议使用CPU计算\n";
	}
}

void test_matrix_vector_mult(int matrix_size = 1536, int num_vectors = 100) {
	cout << "\n***** 测试: 矩阵乘向量 (" << matrix_size << "x" << matrix_size << ") *****\n";
	cout << "向量个数: " << num_vectors << "\n";
	
	vector<float> matrix(matrix_size * matrix_size);
	vector<float> vectors(num_vectors * matrix_size);
	vector<float> result_cpu_naive(num_vectors * matrix_size);
	vector<float> result_cpu_blas(num_vectors * matrix_size);
	vector<float> result_gpu(num_vectors * matrix_size);
	
	generate_random_data(matrix.data(), matrix_size, matrix_size);
	generate_random_data(vectors.data(), num_vectors, matrix_size);
	
	cout << fixed << setprecision(3);
	
	// ========== 1. CPU 暴力计算 ==========
	cout << "\n--- 1. CPU 暴力计算 ---\n";
	auto start_cpu_naive = high_resolution_clock::now();
	for (int i = 0; i < num_vectors; i++) {
		for (int j = 0; j < matrix_size; j++) {
			float sum = 0.0f;
			for (int k = 0; k < matrix_size; k++) {
				sum += matrix[j * matrix_size + k] * vectors[i * matrix_size + k];
			}
			result_cpu_naive[i * matrix_size + j] = sum;
		}
	}
	auto end_cpu_naive = high_resolution_clock::now();
	double cpu_naive_time = duration<double, milli>(end_cpu_naive - start_cpu_naive).count();
	cout << "时间: " << cpu_naive_time << " ms\n";
	
	// ========== 2. CPU BLAS 计算 ==========
	cout << "\n--- 2. CPU BLAS 计算 (cblas_sgemm) ---\n";
	auto start_cpu_blas = high_resolution_clock::now();
	float alpha = 1.0f, beta = 0.0f;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
				num_vectors, matrix_size, matrix_size, 
				alpha, vectors.data(), matrix_size, 
				matrix.data(), matrix_size, 
				beta, result_cpu_blas.data(), matrix_size);
	auto end_cpu_blas = high_resolution_clock::now();
	double cpu_blas_time = duration<double, milli>(end_cpu_blas - start_cpu_blas).count();
	cout << "时间: " << cpu_blas_time << " ms\n";
	
	// ========== 3. GPU CUDA 计算 ==========
	cout << "\n--- 3. GPU CUDA 计算 (cuBLAS) ---\n";
	float *d_matrix, *d_vectors, *d_result;
	auto start_gpu_load = high_resolution_clock::now();
	cudaMalloc(&d_matrix, matrix_size * matrix_size * sizeof(float));
	cudaMalloc(&d_vectors, num_vectors * matrix_size * sizeof(float));
	cudaMalloc(&d_result, num_vectors * matrix_size * sizeof(float));
	cudaMemcpy(d_matrix, matrix.data(), matrix_size * matrix_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vectors, vectors.data(), num_vectors * matrix_size * sizeof(float), cudaMemcpyHostToDevice);
	auto end_gpu_load = high_resolution_clock::now();
	double gpu_load_time = duration<double, milli>(end_gpu_load - start_gpu_load).count();
	cout << "GPU 数据加载时间: " << gpu_load_time << " ms\n";
	
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	auto start_gpu_compute = high_resolution_clock::now();
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, matrix_size, num_vectors, 
				matrix_size, &alpha, d_matrix, matrix_size, d_vectors, 
				matrix_size, &beta, d_result, matrix_size);
	cudaDeviceSynchronize();
	auto end_gpu_compute = high_resolution_clock::now();
	double gpu_compute_time = duration<double, milli>(end_gpu_compute - start_gpu_compute).count();
	cout << "GPU 计算时间: " << gpu_compute_time << " ms\n";
	
	cudaMemcpy(result_gpu.data(), d_result, num_vectors * matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_matrix);
	cudaFree(d_vectors);
	cudaFree(d_result);
	cublasDestroy(handle);
	
	// ========== 性能对比 ==========
	cout << "\n========== 性能对比总结 ==========\n";
	cout << "CPU 暴力计算:        " << cpu_naive_time << " ms\n";
	cout << "CPU BLAS 计算:       " << cpu_blas_time << " ms\n";
	cout << "GPU 加载时间:        " << gpu_load_time << " ms\n";
	cout << "GPU 计算时间:        " << gpu_compute_time << " ms\n";
	cout << "GPU 总时间:          " << (gpu_load_time + gpu_compute_time) << " ms\n";
	
	cout << "\n========== 加速比 ==========\n";
	cout << "CPU暴力 vs CPU BLAS: " << (cpu_naive_time / cpu_blas_time) << "x\n";
	cout << "CPU BLAS vs GPU计算: " << (cpu_blas_time / gpu_compute_time) << "x\n";
	cout << "CPU暴力 vs GPU计算:  " << (cpu_naive_time / gpu_compute_time) << "x\n";
	cout << "CPU暴力 vs GPU总时间: " << (cpu_naive_time / (gpu_load_time + gpu_compute_time)) << "x\n";
	
	// ========== 结果验证 ==========
	cout << "\n========== 结果验证 ==========\n";
	
	float max_error_blas = 0.0f;
	float max_error_gpu = 0.0f;
	bool blas_correct = true;
	bool gpu_correct = true;
	
	// 检查所有结果
	for (int i = 0; i < num_vectors * matrix_size; i++) {
		float error_blas = fabs(result_cpu_naive[i] - result_cpu_blas[i]);
		float error_gpu = fabs(result_cpu_naive[i] - result_gpu[i]);
		
		max_error_blas = max(max_error_blas, error_blas);
		max_error_gpu = max(max_error_gpu, error_gpu);
		
		if (error_blas > 1e-3) {
			blas_correct = false;
		}
		if (error_gpu > 1e-3) {
			gpu_correct = false;
		}
	}
	
	cout << "CPU暴力 vs CPU BLAS 最大误差: " << max_error_blas << "\n";
	cout << "CPU BLAS 验证结果: " << (blas_correct ? "✓ 正确" : "✗ 存在误差") << "\n\n";
	
	cout << "CPU暴力 vs GPU CUDA 最大误差: " << max_error_gpu << "\n";
	cout << "GPU CUDA 验证结果: " << (gpu_correct ? "✓ 正确" : "✗ 存在误差") << "\n";
	
	// 详细验证前几个元素
	cout << "\n前 5 个结果详细对比:\n";
	cout << "索引 | CPU暴力      | CPU BLAS     | GPU CUDA     | BLAS误差   | GPU误差\n";
	cout << "-----|--------------|--------------|--------------|------------|----------\n";
	for (int i = 0; i < min(5, num_vectors * matrix_size); i++) {
		cout << setw(4) << i << " | " 
		     << setw(12) << result_cpu_naive[i] << " | "
		     << setw(12) << result_cpu_blas[i] << " | "
		     << setw(12) << result_gpu[i] << " | "
		     << setw(10) << fabs(result_cpu_naive[i] - result_cpu_blas[i]) << " | "
		     << setw(10) << fabs(result_cpu_naive[i] - result_gpu[i]) << "\n";
	}
}

// K-means测试
void test_kmeans(int n = 10000, int d = 256, int k_clusters = 100) {
	cout << "\n***** 测试: K-means 聚类 *****\n";
	cout << "数据规模: " << n << " 个向量, 维度: " << d << ", 聚类数: " << k_clusters << "\n";
	
	vector<float> x(n * d);
	generate_random_data(x.data(), n, d);
	
	auto start_cpu = high_resolution_clock::now();
	faiss::Clustering clustering_cpu(d, k_clusters);
	clustering_cpu.verbose = false;
	faiss::IndexFlatL2 index_cpu(d);
	clustering_cpu.train(n, x.data(), index_cpu);
	auto end_cpu = high_resolution_clock::now();
	double cpu_time = duration<double, milli>(end_cpu - start_cpu).count();
	
	faiss::gpu::StandardGpuResources res;
	auto start_gpu_load = high_resolution_clock::now();
	faiss::gpu::GpuIndexFlatL2 index_gpu(&res, d);
	auto end_gpu_load = high_resolution_clock::now();
	double gpu_load_time = duration<double, milli>(end_gpu_load - start_gpu_load).count();
	
	auto start_gpu_compute = high_resolution_clock::now();
	faiss::Clustering clustering_gpu(d, k_clusters);
	clustering_gpu.verbose = false;
	clustering_gpu.train(n, x.data(), index_gpu);
	auto end_gpu_compute = high_resolution_clock::now();
	double gpu_compute_time = duration<double, milli>(end_gpu_compute - start_gpu_compute).count();
	
	print_performance("K-means 聚类", cpu_time, gpu_load_time, gpu_compute_time);
}

int main(int argc, char** argv) {
	int scale = 3; // 数据规模：1=小, 2=中, 3=大, 4=超大(千万级)
	
	if (argc > 1) {
		scale = atoi(argv[1]);
	}
	
	cout << "===== Faiss GPU vs CPU 性能对比测试 =====\n";
	
	int nb, nq, d, k, np = 100;
	
	switch(scale) {
		case 1: // 小规模
			nb = 100000; nq = 100; d = 128; k = 10;
			cout << "数据规模: 小 (100K 向量)\n";
			break;
		case 2: // 中规模
			nb = 1000000; nq = 1000; d = 128; k = 10,np=2000;
			cout << "数据规模: 中 (1M 向量)\n";
			break;
		case 3: // 大规模
			nb = 10000000; nq = 10000; d = 128; k = 10,np=20000;
			cout << "数据规模: 大 (10M 向量)\n";
			break;
		case 4: // 超大规模（千万级）
			nb = 100000000; nq = 100000; d = 128; k = 10;
			cout << "数据规模: 超大 (1亿向量)\n";
			break;
		default:
			nb = 100000; nq = 100; d = 512; k = 10;
			cout << "数据规模: 自定义\n";
	}
	
	cout << "======================================\n\n";
	
	
	// K-means测试
	test_kmeans(nb, d, np);

	// 测试内积运算
	test_matrix_operation(nb, nq, d, k);
	
	// 测试矩阵乘向量
	test_matrix_vector_mult(1536, min(1000000, nb / 100));
	
	cout << "\n===== 所有测试完成 =====\n";
	return 0;
}
