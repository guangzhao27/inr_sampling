Some important file positions:

## how to install torchgeo?
conda create -p torchgeo 
python 3.11, higher will cause error with hydra and torch_geo


## Important files
1. train_utility_sampling/SamplerWrapper.py
    include the definition of all types of samplers
        the sample function: INRSingle2dSamplerWrapper.sample

2. /train_utility_sampling/taylor_estimation.py
    include using taylor expansion to estimate gradient
    the best functions to use for jacbian estimation:
        frobenius_norm_via_jacrev 
        cell_grad_variance_estimate_with_jacrev


2. train_utility_sampling/train_utility.py
    train_step_single_image: single train step

3. /utils/quadtree.py
    adaptive grid tree class for split



## Test files
3. /test/Run_Test.ipynb
    test file for sampler and training function
4. /test/taylor_gradient_estimation.ipynb
    test gradient estimation using taylor expansion

## script file
4. script file: /inr_sample/single_image_inr.py
5. bash script file: script/inr_sample/nersc-scent-single-inr-sample.sh

TODO: 

Simple version for adaptive grid function:
1. assume each cell sample same number of samples (approximate Neyman)
2. split the cell with larger WS^2 (approximate propotional, Neyman is WS)
3. use cached value to update grid instance, not remove old and add new cells on top
4. /utils/quadtree.py#L37 set the smallest cell size

