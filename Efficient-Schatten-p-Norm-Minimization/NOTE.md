## Some note for this file

this *res* file contains matlab source code from the author of paper *Low-Rank Matrix Recovery via Efficient Schatten p-Norm Minimization* .But the original code seems to have some implementation glitches,so I did some changes. Finally the code runs successfully. 

By the way ,my implementation of these 2 algorithms are in python which are in the `src` file.

Due to the inefficiency of python API `scipy.linalg.fractional_matrix_power( )`. I honestly don't recommend testing of the Python version code on a really large and sparse matrix (eg.ml-1m), which would be a huge waste of resources without a high-performance computer. You can consider using my modified Matlab code in the `res` to run the tests, which will be more efficient. Additionally, you can experiment with both algorithms with generate matrices of any size by setting the parameter `load_data = False` in `main.py`.