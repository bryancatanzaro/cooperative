nested_sum: nested_sum.cu
	nvcc -o nested_sum -I$(THRUST_PATH) -I../.. nested_sum.cu -arch=sm_20

for_each: for_each.cu
	nvcc -o for_each -I$(THRUST_PATH) -I../.. for_each.cu -arch=sm_20

