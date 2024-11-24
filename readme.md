# FlexiFed_VGG

This repository built the simulation code for the paper "FlexiFed: Personalized Federated Learning for Edge Clients with Heterogeneous Model Architectures". The code implements a 40-client VGG training on the CIFAR10 dataset with i.i.d patterns. The code has been tested and validated on both the 4090 platform and the Apple Silicon M4 Pro platform. 

### Dependencies

```conda install --yes --file requirements.txt```

### Known Issues

The algorithm does not converge. This issue is likely due to the training dataset for each user being too small to provide sufficient knowledge for effective learning. Additional experiments have shown that this issue also arises in single-user scenarios with limited training samples, which indicates that data insufficiency is a key factor in model training. Addressing this issue may require increasing the dataset size to improve learning effectiveness.
