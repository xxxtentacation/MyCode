from typing import List, Tuple, Dict, Any
import heapq
import random
import time

# 定义一个全局整型变量
n = 100  # 减小规模以便快速测试
m = 0.4
rt_s = 0
rt_e = int(50.5 * n * m)
pt_s = 1
pt_e = 100


def increment():
    global rt_s, rt_e, pt_s, pt_e, n


class Job:
    def __init__(self, index: int, ready_time: int, processing_time: int):
        self.index = index
        self.ready_time = ready_time
        self.processing_time = processing_time

    def __repr__(self):
        return f"Job{self.index}(r={self.ready_time}, p={self.processing_time})"


class Node:
    def __init__(self, partial_sequence: List[Job], level: int, lower_bound: float, upper_bound: float,
                 completion_time: float):
        self.partial_sequence = partial_sequence
        self.level = level
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.completion_time = completion_time  # C_k for the partial sequence
        self.id = id(self) % 1000

    def __lt__(self, other):
        # 修改：优先比较下界，若下界相同，则比较上界（更紧的上界优先）
        if self.lower_bound == other.lower_bound:
            return self.upper_bound < other.upper_bound
        return self.lower_bound < other.lower_bound

    def __repr__(self):
        seq_str = "->".join([f"J{j.index}" for j in self.partial_sequence])
        return f"Node{self.id}[L{self.level}: {seq_str}, LB={self.lower_bound:.1f}, UB={self.upper_bound:.1f}]"


def print_operation(message: str, indent: int = 0):
    prefix = " " * indent
    print(f"{prefix}▶ {message}")


def compute_start_completion_times(sequence: List[Job]) -> Tuple[List[float], List[float]]:
    start_times = []
    completion_times = []
    current_time = 0
    for job in sequence:
        start_time = max(current_time, job.ready_time)
        completion_time = start_time + job.processing_time
        start_times.append(start_time)
        completion_times.append(completion_time)
        current_time = completion_time
    return start_times, completion_times


# --- 新增辅助函数 ---
def get_blocks(sequence: List[Job]) -> List[List[Job]]:
    """根据论文定义，将序列划分为blocks"""
    if not sequence:
        return []
    blocks = []
    current_block = [sequence[0]]
    _, comp_times = compute_start_completion_times(sequence)

    for i in range(1, len(sequence)):
        # 检查是否开始新块: r_i > C_{i-1}
        if sequence[i].ready_time > comp_times[i - 1]:
            blocks.append(current_block)
            current_block = [sequence[i]]
        else:
            current_block.append(sequence[i])
    blocks.append(current_block)
    return blocks


def is_ect_sequence(sequence: List[Job]) -> bool:
    """检查一个序列（或块）是否遵循ECT规则"""
    if len(sequence) <= 1:
        return True
    current_time = 0
    for i, job in enumerate(sequence):
        start_time = max(current_time, job.ready_time)
        completion_time = start_time + job.processing_time

        # 找到下一个应该被调度的作业（ECT）
        remaining = sequence[i + 1:]
        if not remaining:
            break

        best_next = None
        best_ct = float('inf')
        best_st = float('inf')
        for candidate in remaining:
            cand_st = max(completion_time, candidate.ready_time)
            cand_ct = cand_st + candidate.processing_time
            if (cand_ct, cand_st, candidate.index) < (best_ct, best_st, best_next.index if best_next else float('inf')):
                best_next = candidate
                best_ct = cand_ct
                best_st = cand_st

        # 如果下一个实际作业不是ECT选择的，则不是ECT序列
        if sequence[i + 1] != best_next:
            return False

        current_time = completion_time
    return True


def passes_optimality_test(sequence: List[Job]) -> bool:
    """实现论文Theorem 5的最优性检验"""
    blocks = get_blocks(sequence)
    for block in blocks:
        # 1. 块必须按ECT排序
        if not is_ect_sequence(block):
            return False
        # 2. 块的第一个作业必须是该块中准备时间最小的
        first_job = block[0]
        min_ready_time_in_block = min(job.ready_time for job in block)
        if first_job.ready_time != min_ready_time_in_block:
            return False
    return True


def ect_rule_sequence_from_time(jobs: List[Job], start_time: float) -> List[Job]:
    """应用ECT规则排序作业，从给定的start_time开始"""
    unscheduled = jobs.copy()
    sequence = []
    current_time = start_time
    while unscheduled:
        candidate_info = []
        for job in unscheduled:
            start_time_j = max(current_time, job.ready_time)
            completion_time_j = start_time_j + job.processing_time
            candidate_info.append({
                'job': job,
                'start_time': start_time_j,
                'completion_time': completion_time_j
            })
        selected_info = min(candidate_info, key=lambda x: (x['completion_time'], x['start_time'], x['job'].index))
        selected_job = selected_info['job']
        sequence.append(selected_job)
        unscheduled.remove(selected_job)
        current_time = selected_info['completion_time']
    return sequence


def compute_bounds(partial_sequence: List[Job], all_jobs: List[Job], C_k: float) -> Tuple[float, float, List[Job]]:
    """
    根据论文Procedure BOUND计算下界(LB)和上界(UB)。
    返回: (LB, UB, 完整的上界序列)
    """
    unscheduled = [j for j in all_jobs if j not in partial_sequence]
    if not unscheduled:
        _, comp_times = compute_start_completion_times(partial_sequence)
        total_ct = sum(comp_times)
        return total_ct, total_ct, partial_sequence.copy()

    # Step 1: 构建上界序列 S = (partial, S_g) 其中 S_g 是 ECT 序列
    S_g = ect_rule_sequence_from_time(unscheduled, C_k)
    full_upper_seq = partial_sequence + S_g
    _, comp_times_upper = compute_start_completion_times(full_upper_seq)
    UB = sum(comp_times_upper)

    # Step 2: 构建下界序列 S' = (partial, S'_g)
    # S'_g 与 S_g 作业顺序相同，但准备时间被松弛
    S_prime_g = S_g.copy()  # 保持顺序
    LB_scheduled = sum(comp_times_upper[:len(partial_sequence)]) if partial_sequence else 0

    # 初始化下界序列的计算
    C_prime_prev = C_k
    LB_unscheduled = 0

    remaining_jobs_for_m = unscheduled.copy()
    for job in S_prime_g:
        # m_x = min{ r_j for j in remaining_jobs_for_m }
        m_x = min(j.ready_time for j in remaining_jobs_for_m)
        # R'_x = max(C'_{x-1}, m_x)
        R_prime_x = max(C_prime_prev, m_x)
        # C'_x = R'_x + p_x
        C_prime_x = R_prime_x + job.processing_time
        LB_unscheduled += C_prime_x
        C_prime_prev = C_prime_x
        # 从remaining中移除当前作业，为下一个m_x做准备
        remaining_jobs_for_m.remove(job)

    LB = LB_scheduled + LB_unscheduled
    return LB, UB, full_upper_seq


def dominance_test(current_node: Node, candidate_jobs: List[Job], all_jobs: List[Job]) -> List[Job]:
    """应用优势测试（定理1-3）"""
    print_operation("应用支配测试", 1)
    if not candidate_jobs:
        return []

    C_k = current_node.completion_time

    # --- 定理2修正 ---
    # 先为所有候选作业计算 C_i(S_K)
    C_i_values = {}
    for job in candidate_jobs:
        R_i = max(job.ready_time, C_k)
        C_i = R_i + job.processing_time
        C_i_values[job] = C_i

    C_min = min(C_i_values.values())
    remaining_after_theorem2 = []
    for job in candidate_jobs:
        if job.ready_time >= C_min:  # 注意是 >=
            print_operation(f"定理2: 消除 {job} (r_j={job.ready_time} >= C_min={C_min})", 2)
        else:
            remaining_after_theorem2.append(job)

    if not remaining_after_theorem2:
        return []

    # --- 定理1修正 ---
    remaining_after_theorem1 = []
    for job_j in remaining_after_theorem2:
        dominated = False
        R_j = max(job_j.ready_time, C_k)
        for job_i in remaining_after_theorem2:
            if job_i != job_j and job_i.processing_time <= job_j.processing_time:
                R_i = max(job_i.ready_time, C_k)
                if R_j >= R_i:
                    print_operation(f"定理1: {job_i} 优于 {job_j}", 2)
                    dominated = True
                    break
        if not dominated:
            remaining_after_theorem1.append(job_j)

    if not remaining_after_theorem1:
        return []

    # --- 定理3修正 ---
    final_remaining = []
    for job_j in remaining_after_theorem1:
        dominated = False
        C_j = max(job_j.ready_time, C_k) + job_j.processing_time
        for job_i in remaining_after_theorem1:
            if job_i != job_j and job_i.processing_time <= job_j.processing_time:  # 论文是 p_i <= p_j
                C_i = max(job_i.ready_time, C_k) + job_i.processing_time
                if C_j >= C_i:  # 论文是 C_j >= C_i
                    print_operation(f"定理3: {job_i} 优于 {job_j}", 2)
                    dominated = True
                    break
        if not dominated:
            final_remaining.append(job_j)

    print_operation(f"优势测试完成: {len(final_remaining)}个作业保留", 2)
    return final_remaining


class BranchAndBoundScheduler:
    def __init__(self, jobs: List[Job]):
        self.jobs = jobs
        self.n = len(jobs)
        self.best_solution = None
        self.best_upper_bound = float('inf')
        self.iteration_count = 0
        self.node_count = 0

    def solve(self) -> Dict[str, Any]:
        print_operation("=== 分支定界算法开始 (基于Dessouky & Deogun 1981) ===")
        print_operation(f"问题规模: {self.n}个作业")

        # 阶段I: 初始化
        print_operation("阶段I: 初始化")
        # 初始ECT序列
        initial_sequence = ect_rule_sequence_from_time(self.jobs, 0)
        # 计算初始序列的完成时间和C_k
        _, comp_times_init = compute_start_completion_times(initial_sequence)
        C_k_init = comp_times_init[-1] if comp_times_init else 0
        # 使用统一的bound计算函数
        initial_lb, initial_ub, initial_full_seq = compute_bounds([], self.jobs, 0)
        self.best_upper_bound = initial_ub
        self.best_solution = initial_full_seq

        print_operation(f"初始ECT序列: {[j.index for j in initial_sequence]}")
        print_operation(f"初始下界: {initial_lb}, 上界: {initial_ub}")

        # --- 新增：最优性检验 ---
        if passes_optimality_test(initial_sequence):
            print_operation("初始ECT序列通过最优性检验(Theorem 5)，算法终止。")
            self.best_solution = initial_sequence
            start_times, comp_times = compute_start_completion_times(self.best_solution)
            total_completion = sum(comp_times)
            total_flow = total_completion - sum(j.ready_time for j in self.jobs)
            mean_flow = total_flow / self.n
            return {
                'optimal_sequence': [j.index for j in self.best_solution],
                'total_completion_time': total_completion,
                'total_flow_time': total_flow,
                'mean_flow_time': mean_flow,
                'iterations': 0,
                'nodes_generated': 0,
                'early_termination': True
            }

        # 创建初始节点
        initial_node = Node([], 0, initial_lb, initial_ub, 0)
        active_nodes = [initial_node]
        self.node_count = 1
        print_operation("阶段II: 分支定界搜索")

        while active_nodes:
            self.iteration_count += 1
            print_operation(f"\n--- 迭代 {self.iteration_count} ---")
            # 选择下界最小的节点
            current_node = heapq.heappop(active_nodes)
            print_operation(f"当前节点: {current_node}")

            # 如果当前节点的下界已经不小于已知最优上界，可以剪枝（虽然堆顶通常不会这样）
            if current_node.lower_bound >= self.best_upper_bound:
                print_operation(f"剪枝: 节点LB {current_node.lower_bound} >= 最优UB {self.best_upper_bound}")
                continue

            # 获取候选作业
            candidate_jobs = [j for j in self.jobs if j not in current_node.partial_sequence]
            # 应用优势测试
            remaining_jobs = dominance_test(current_node, candidate_jobs, self.jobs)

            # 为每个候选作业生成子节点
            for job in remaining_jobs:
                print_operation(f"扩展作业 {job}")
                # 创建新部分序列
                new_partial = current_node.partial_sequence + [job]
                # 计算新部分序列的完成时间 C_k_new
                _, comp_times_new = compute_start_completion_times(new_partial)
                C_k_new = comp_times_new[-1]
                # --- 关键修正：使用统一的、符合论文的bound计算函数 ---
                new_lb, new_ub, new_full_seq = compute_bounds(new_partial, self.jobs, C_k_new)

                # 更新全局最优上界
                if new_ub < self.best_upper_bound:
                    self.best_upper_bound = new_ub
                    self.best_solution = new_full_seq
                    print_operation(f"更新最优上界: {self.best_upper_bound}")

                    # 剪枝：如果下界大于等于当前最优上界
                if new_lb >= self.best_upper_bound:
                    print_operation(f"剪枝: LB={new_lb} >= 当前最优UB={self.best_upper_bound}")
                    continue

                    # 创建新节点
                new_node = Node(new_partial, current_node.level + 1, new_lb, new_ub, C_k_new)
                heapq.heappush(active_nodes, new_node)
                self.node_count += 1
                print_operation(f"创建新节点: {new_node}")

            print_operation(f"活跃节点数: {len(active_nodes)}")
            # 可选：添加一个安全退出机制防止无限循环
            if self.iteration_count > 10000:
                print_operation("达到最大迭代次数，强制终止。")
                break

        # 阶段III: 终止
        print_operation("\n阶段III: 终止")
        if self.best_solution:
            start_times, comp_times = compute_start_completion_times(self.best_solution)
            total_completion = sum(comp_times)
            total_flow = total_completion - sum(j.ready_time for j in self.jobs)
            mean_flow = total_flow / self.n
            print_operation(f"最优序列: {[j.index for j in self.best_solution]}")
            print_operation(f"总完成时间: {total_completion}")
            print_operation(f"总流程时间: {total_flow}")
            print_operation(f"平均流程时间: {mean_flow:.2f}")

            return {
                'optimal_sequence': [j.index for j in self.best_solution],
                'total_completion_time': total_completion,
                'total_flow_time': total_flow,
                'mean_flow_time': mean_flow,
                'iterations': self.iteration_count,
                'nodes_generated': self.node_count,
                'early_termination': False
            }
        else:
            raise RuntimeError("未能找到可行解")


# ... (generate_random_jobs, print_job_table, test_random_example 函数保持不变) ...

def generate_random_jobs(n: int, ready_time_range: tuple = (rt_s, rt_e), processing_time_range: tuple = (pt_s, pt_e)) -> \
List[Job]:
    jobs = []
    for i in range(1, n + 1):
        ready_time = random.randint(ready_time_range[0], ready_time_range[1])
        processing_time = random.randint(processing_time_range[0], processing_time_range[1])
        jobs.append(Job(i, ready_time, processing_time))
    return jobs


def print_job_table(jobs: List[Job]):
    print("生成的作业数据:")
    print("-" * 50)
    print(f"{'作业索引':<10} {'准备时间':<10} {'处理时间':<10}")
    print("-" * 50)
    for job in jobs:
        print(f"{job.index:<10} {job.ready_time:<10} {job.processing_time:<10}")
    print("-" * 50)


def test_random_example(n_jobs, ready_time_range: tuple = (rt_s, rt_e), processing_time_range: tuple = (pt_s, pt_e),
                        verbose: bool = True):
    global original_print
    print("=" * 60)
    print("测试分支定界算法（随机生成作业数据）")
    print("=" * 60)
    random.seed(42)  # 固定种子以便复现
    jobs = generate_random_jobs(n_jobs, ready_time_range, processing_time_range)
    print_job_table(jobs)
    start_time = time.time()
    scheduler = BranchAndBoundScheduler(jobs)
    if not verbose:
        original_print = print_operation
        scheduler.print_operation = lambda msg, indent=0: None
    result = scheduler.solve()
    if not verbose:
        scheduler.print_operation = original_print
    end_time = time.time()
    print("\n" + "=" * 60)
    print("最终结果摘要:")
    print("=" * 60)
    print(f"作业数量: {n_jobs}")
    print(f"计算时间: {end_time - start_time:.4f}秒")
    print(f"最优序列: {result['optimal_sequence']}")
    print(f"总完成时间: {result['total_completion_time']}")
    print(f"总流程时间: {result['total_flow_time']}")
    print(f"平均流程时间: {result['mean_flow_time']:.2f}")
    print(f"迭代次数: {result['iterations']}")
    print(f"生成节点数: {result['nodes_generated']}")
    if result.get('early_termination', False):
        print("算法因初始解最优而提前终止。")


if __name__ == "__main__":
    increment()
    test_random_example(n_jobs=n, verbose=True)