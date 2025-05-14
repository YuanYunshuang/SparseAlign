
```
//Comment Out
//#include <THE/THC.h>
//extern THCState *state;
//cudaStream_t stream = THCState_getCurrentStream(state);

//Replace with
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
cudaStream_t stream = at::cuda::getCurrentCUDAStream();
```