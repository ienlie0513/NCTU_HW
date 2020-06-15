import matplotlib.pyplot as plt

if __name__ == '__main__':
	mean_list = []
	with open("mean_.txt", 'r') as f:
		for line in f:
			mean_list.append(line.replace('\n',''))
	print ("mean", mean_list)

	max_list = []
	with open("max_.txt", 'r') as f:
		for line in f:
			max_list.append(line.replace('\n',''))
	print ("max", max_list)

	plt.figure(figsize=(20, 12))
	x = range(len(mean_list))
	plt.ylabel("Scores")
	plt.xlabel("Epochs")
	plt.title("Score Curve", fontsize=18)
	plt.semilogy(x, mean_list, label='Mean Scores')
	plt.semilogy(x, max_list, label='Max Scores')
	plt.legend()
	plt.show()

