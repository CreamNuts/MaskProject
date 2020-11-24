zero2minusone = {
    'mean': [0.5, 0.5, 0.5],
    'std' : [0.5, 0.5, 0.5]
} 

minusone2zero = {
    'mean': [-1.0, -1.0, -1.0],
    'std' : [2.0, 2.0, 2.0]
}

MASK_PARAMETERS = {
    'mean': [0.4712, 0.4701, 0.4689],
    'std': [0.3324, 0.3320, 0.3319]
    }

MASK_UNNORMALIZE = {
    'mean' : [-mean/std for mean, std in zip(MASK_PARAMETERS['mean'], MASK_PARAMETERS['std'])],
    'std' : [1.0/std for std in MASK_PARAMETERS['std']]
}