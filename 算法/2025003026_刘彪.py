if __name__ == "__main__":
	times = [2, 14, 4, 16, 6, 5, 3]  
 #times = [2, 14, 4, 16, 6, 5, 3]，machines = 3
 #times = [5, 5, 5, 5, 5, 5]，machines = 2
 #times = [100, 1, 2, 3]，machines = 3
	machines = 3


	n = len(times)
	for i in range(n - 1):
		for j in range(0, n - 1 - i):
			if times[j] < times[j + 1]:
				times[j], times[j + 1] = times[j + 1], times[j]


	loads = [0 for _ in range(machines)]

	for  p_time in times:
		m_idx = 0
		for k in range(1, machines):
			if loads[k] < loads[m_idx]:
				m_idx = k
		loads[m_idx] += p_time


	cmax = loads[0]
	for k in range(1, machines):
		if loads[k] > cmax:
			cmax = loads[k]
				
	print(cmax)

