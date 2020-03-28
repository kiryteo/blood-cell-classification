## How to

Look at the `build_data.sh` file. It supposes you have a directory `train_1` with the mat files to use and a `test_1` directory with the mat files to use to do the testing part.

(Same for train_2 and test_2)

It will generates the preprocessed files into tmp/train1, tmp/test1 (same for 2)

The file `train_and_testallfile.py` will do the experiments and it has path variables to the directory where the processed data should be.

The training structure of that file is:

for e in epoch:
	for file in files:
		load_that_file()
		for steps in range(steps_per_file)
			X,Y = generate_batch()
			train_on(X,Y)


