import os


for x in range(6):	#number of 0's after decimal in SA const

	for y in range(6):	#1/5 number of problem instances per evaluation

		num_matrices = (y+1)*5
		filename = "job" + str(num_matrices) + "-nocap-pos-SA" + str(x+1)

		# If the job file already exists, try to remove it first so we can
		# recreate it cleanly.
		if os.path.exists(filename):
			try:
				os.remove(filename)
			except OSError:
				# If we cannot remove it, skip creating this job.
				continue

		f = open(filename, "w")
		f.write("#!/bin/bash\n")
		f.write("#SBATCH -N 1\n")
		f.write("#SBATCH -t 24:00:00\n")
		f.write("echo start of job\n")
		run_string = "./mat_mult " + str(num_matrices) + " 1000000 10 " + str(10**-(x+2)) + " nocap pos SA\n"
		f.write(run_string)
		f.write("echo end of job\n")
		f.close()

#filename = "./mat_mult " + str((4+1)*5) + " 1000000 10 " + str(10**-(5+2)) + " nocap pos"
#print(filename)
