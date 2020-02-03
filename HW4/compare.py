def main():

	filename1 = 'ss'
	filename2 = 'cc'
	line1 = []
	line2 = []
	with open(filename1) as f:
	    for l in f:
	        line1.append(l)

	with open(filename2) as f:
	    for l in f:
	        line2.append(l)

	line1 = line1[4:-3]
	line2 = line2[4:-3]
	size = len(line1)

	num = 0
	for i in range(size):
	    list1 = [float(x) for x in line1[i].split()]
	    list2 = [float(x) for x in line2[i].split()]
	    list_size = len(list1)
	    for j in range(list_size):
	        if abs(list1[j]-list2[j]) > 0.001:
	            num += 1
	            print('line ', i)
	            print(line1[i])
	            print(line2[i])
	            break
	            
	print('Numeber of different line:', num)

if __name__ == "__main__":
    main()