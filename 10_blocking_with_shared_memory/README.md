## Results
* Comparsion

|  	| w/o shared mem 	| w/ shared mem 	|
|----------	|----------------	|---------------	|
| Run Time 	| 250.19ms 	| 51.514ms 	|

* GPU Version 1: without using shared memory in block
    ```
    [ec2-user@ip-172-31-23-23 10_blocking_with_shared_memory]$ nvprof ./a1.out
    ==6637== NVPROF is profiling process 6637, command: ./a1.out
    0. Run time: 0
    1. Run time: 0
    2. Run time: 0
    3. Run time: 0
    4. Run time: 0
    5. Run time: 0
    6. Run time: 0
    7. Run time: 0
    8. Run time: 0
    9. Run time: 0
    10. Run time: 0
    11. Run time: 0
    12. Run time: 0
    13. Run time: 0
    14. Run time: 0
    15. Run time: 0
    16. Run time: 0
    17. Run time: 0
    18. Run time: 0
    19. Run time: 0
    Fastest time: 0
    Final results:
    0.2265
    1.1141
    2.6383
    3.3030
    4.2569
    5.7800
    6.5012
    7.4324
    8.1908
    9.7142
    10.6063
    11.4674
    12.6718
    13.2828
    14.8228
    15.1973
    16.7407
    17.4511
    18.3214
    19.980
    ==6637== Profiling application: ./a1.out
    ==6637== Profiling result:
                Type  Time(%)      Time     Calls       Avg       Min       Max  Name
     GPU activities:   99.98%  250.19ms        20  12.509ms  12.437ms  12.577ms  FindClosetGPU(float3_t*, int*, int)
                        0.01%  30.623us         2  15.311us  10.464us  20.159us  [CUDA memcpy HtoD]
                        0.01%  25.471us         2  12.735us  7.7120us  17.759us  [CUDA memcpy DtoH]
          API calls:   63.50%  249.87ms         4  62.469ms  39.795us  249.67ms  cudaMemcpy
                       36.05%  141.86ms         2  70.932ms  9.4420us  141.85ms  cudaMalloc
                        0.13%  517.60us        20  25.879us  12.713us  170.18us  cudaLaunchKernel
                        0.13%  509.09us         1  509.09us  509.09us  509.09us  cuDeviceTotalMem
                        0.11%  420.09us        96  4.3750us     744ns  139.95us  cuDeviceGetAttribute
                        0.06%  245.30us         2  122.65us  37.566us  207.74us  cudaFree
                        0.01%  35.214us         1  35.214us  35.214us  35.214us  cuDeviceGetName
                        0.00%  4.8200us         1  4.8200us  4.8200us  4.8200us  cuDeviceGetPCIBusId
                        0.00%  4.3020us         3  1.4340us     837ns  2.0260us  cuDeviceGetCount
                        0.00%  2.4660us         2  1.2330us     853ns  1.6130us  cuDeviceGet
                        0.00%     952ns         1     952ns     952ns     952ns  cuDeviceGetUuid
    ```

* GPU Version 2: using shared memory in block.
    ```
    [ec2-user@ip-172-31-23-23 10_blocking_with_shared_memory]$ nvprof ./a2.out
    ==6664== NVPROF is profiling process 6664, command: ./a2.out
    0. Run time: 0
    1. Run time: 0
    2. Run time: 0
    3. Run time: 0
    4. Run time: 0
    5. Run time: 0
    6. Run time: 0
    7. Run time: 0
    8. Run time: 0
    9. Run time: 0
    10. Run time: 0
    11. Run time: 0
    12. Run time: 0
    13. Run time: 0
    14. Run time: 0
    15. Run time: 0
    16. Run time: 0
    17. Run time: 0
    18. Run time: 0
    19. Run time: 0
    Fastest time: 28
    Final results:
    0.7953
    1.8075
    2.2403
    3.3600
    4.3971
    5.9179
    6.3822
    7.2859
    8.2016
    9.2678
    10.324
    11.2256
    12.413
    13.3668
    14.6671
    15.4413
    16.7690
    17.8288
    18.6637
    19.4089
    ==6664== Profiling application: ./a2.out
    ==6664== Profiling result:
                Type  Time(%)      Time     Calls       Avg       Min       Max  Name
     GPU activities:   99.89%  51.514ms        20  2.5757ms  2.5688ms  2.6319ms  FindClosetGPU2(float3_t*, int*, int)
                        0.06%  30.814us         2  15.407us  10.495us  20.319us  [CUDA memcpy HtoD]
                        0.05%  25.503us         2  12.751us  7.7440us  17.759us  [CUDA memcpy DtoH]
          API calls:   72.93%  142.14ms         2  71.070ms  9.5510us  142.13ms  cudaMalloc
                       26.20%  51.071ms         4  12.768ms  38.661us  50.905ms  cudaMemcpy
                        0.27%  535.55us        20  26.777us  15.059us  165.50us  cudaLaunchKernel
                        0.26%  509.12us         1  509.12us  509.12us  509.12us  cuDeviceTotalMem
                        0.22%  424.81us        96  4.4250us     746ns  143.90us  cuDeviceGetAttribute
                        0.09%  175.03us         2  87.516us  29.493us  145.54us  cudaFree
                        0.02%  38.460us         1  38.460us  38.460us  38.460us  cuDeviceGetName
                        0.00%  4.5160us         1  4.5160us  4.5160us  4.5160us  cuDeviceGetPCIBusId
                        0.00%  4.4020us         3  1.4670us     768ns  2.2040us  cuDeviceGetCount
                        0.00%  2.2840us         2  1.1420us     840ns  1.4440us  cuDeviceGet
                        0.00%     977ns         1     977ns     977ns     977ns  cuDeviceGetUuid
    ```
