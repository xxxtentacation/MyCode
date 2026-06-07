import numpy as np
from typing import List, Tuple
import itertools


def calculate_makespan(sequence: List[int], processing_times: np.ndarray) -> int:
    """计算给定工件序列的最大完工时间"""
    n_jobs = len(sequence)
    n_machines = processing_times.shape[1]

    completion_times = np.zeros((n_jobs, n_machines))

    # 第一个工件
    first_job = sequence[0]
    completion_times[0, 0] = processing_times[first_job, 0]
    for j in range(1, n_machines-1):
        completion_times[0, j] = completion_times[0, j - 1] + processing_times[first_job, j]

    # 剩余工件
    for i in range(1, n_jobs):
        job = sequence[i]
        completion_times[i, 0] = completion_times[i - 1, 0] + processing_times[job, 0]
        for j in range(1, n_machines):
            completion_times[i, j] = max(completion_times[i, j - 1], completion_times[i - 1, j]) + processing_times[
                job, j]

    return int(completion_times[-1, -1])


def neh_algorithm(processing_times: np.ndarray) -> Tuple[List[int], int]:
    """NEH算法实现"""
    n_jobs, n_machines = processing_times.shape

    # 第一步：按总加工时间降序排序
    total_times = np.sum(processing_times, axis=1)
    job_order = sorted([(total_times[i], i) for i in range(n_jobs)],
                       key=lambda x: x[0], reverse=True)
    ordered_jobs = [job[1] for job in job_order]

    print(f"按总加工时间排序的工件顺序: {ordered_jobs}")
    print(f"对应的总加工时间: {[int(total_times[i]) for i in ordered_jobs]}")

    # 第二步：处理前两个工件
    if n_jobs >= 2:
        job1, job2 = ordered_jobs[0], ordered_jobs[1]

        seq1 = [job1, job2]
        seq2 = [job2, job1]

        makespan1 = calculate_makespan(seq1, processing_times)
        makespan2 = calculate_makespan(seq2, processing_times)

        print(f"\n初始两个工件的测试:")
        print(f"序列 {seq1}: makespan = {makespan1}")
        print(f"序列 {seq2}: makespan = {makespan2}")

        if makespan1 <= makespan2:
            current_sequence = seq1
            current_makespan = makespan1
        else:
            current_sequence = seq2
            current_makespan = makespan2
    else:
        current_sequence = ordered_jobs
        current_makespan = calculate_makespan(current_sequence, processing_times)
        return current_sequence, current_makespan

    print(f"初始序列: {current_sequence}, makespan = {current_makespan}")

    # 第三步：逐个插入剩余工件
    for k in range(2, n_jobs):
        current_job = ordered_jobs[k]
        best_insert_pos = 0
        best_makespan = float('inf')

        print(f"\n插入工件 {current_job} (总加工时间: {int(total_times[current_job])})")
        print(f"当前序列: {current_sequence}, 当前makespan: {current_makespan}")

        # 测试所有可能的插入位置
        test_results = []
        for insert_pos in range(len(current_sequence) + 1):
            test_sequence = current_sequence.copy()
            test_sequence.insert(insert_pos, current_job)
            test_makespan = calculate_makespan(test_sequence, processing_times)
            test_results.append((insert_pos, test_makespan))

            if test_makespan < best_makespan:
                best_makespan = test_makespan
                best_insert_pos = insert_pos

        # 显示所有测试结果
        for pos, makespan in test_results:
            marker = " ← 最优" if pos == best_insert_pos else ""
            print(f"  位置 {pos}: makespan = {makespan}{marker}")

        # 插入到最佳位置
        current_sequence.insert(best_insert_pos, current_job)
        current_makespan = best_makespan
        print(f"选择插入位置 {best_insert_pos}, 新makespan: {current_makespan}")

    return current_sequence, current_makespan


def exhaustive_search(processing_times: np.ndarray) -> Tuple[List[int], int]:
    """穷举搜索找到全局最优解"""
    n_jobs = processing_times.shape[0]
    all_sequences = list(itertools.permutations(range(n_jobs)))

    best_sequence = None
    best_makespan = float('inf')
    all_results = []

    for seq in all_sequences:
        seq_list = list(seq)
        makespan = calculate_makespan(seq_list, processing_times)
        all_results.append((seq_list, makespan))

        if makespan < best_makespan:
            best_makespan = makespan
            best_sequence = seq_list

    # 按makespan排序显示所有结果
    all_results.sort(key=lambda x: x[1])

    print("\n=== 所有可能序列的makespan排序 ===")
    for i, (seq, makespan) in enumerate(all_results[:10]):
        marker = " ← 全局最优" if makespan == best_makespan else ""
        print(f"{i + 1:2d}. 序列 {seq}: makespan = {makespan}{marker}")  # 修复缩进
    return best_sequence, best_makespan


def test_neh():
    """测试NEH算法"""
    # 使用一个更复杂的例子来展示NEH的局限性
    processing_times = np.array([
        [3, 2, 4],  # 工件0
        [1, 4, 2],  # 工件1
        [5, 1, 3],  # 工件2
        [2, 3, 1]  # 工件3
    ])

    print("加工时间矩阵:")
    print(processing_times)
    print("工件总加工时间:", np.sum(processing_times, axis=1))

    print("\n" + "=" * 60)
    print("NEH算法执行过程")
    print("=" * 60)

    # 运行NEH算法
    neh_sequence, neh_makespan = neh_algorithm(processing_times)

    print("\n" + "=" * 60)
    print("NEH算法结果")
    print("=" * 60)
    print(f"NEH找到的序列: {neh_sequence}")
    print(f"NEH的makespan: {neh_makespan}")

    # 穷举搜索找到全局最优
    print("\n" + "=" * 60)
    print("穷举搜索验证")
    print("=" * 60)

    optimal_sequence, optimal_makespan = exhaustive_search(processing_times)

    print(f"\n全局最优序列: {optimal_sequence}")
    print(f"全局最优makespan: {optimal_makespan}")

    # 比较结果
    print("\n" + "=" * 60)
    print("结果比较")
    print("=" * 60)
    print(f"NEH结果与最优解的差距: {neh_makespan - optimal_makespan}")
    print(f"相对误差: {(neh_makespan - optimal_makespan) / optimal_makespan * 100:.2f}%")

    if neh_makespan == optimal_makespan:
        print("✅ NEH找到了全局最优解！")
    else:
        print("⚠️ NEH没有找到全局最优解（这是正常的，NEH是启发式算法）")



if __name__ == "__main__":
    test_neh()

