## compare different nn lib

use 2000 image from cifar 10



test params:

```
batch_size = 128
num_classes = 10
epochs = 5
steps = 100
```

### outputs cpu

cpu: Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz (8 CPUs), ~2.6GHz

* keras tensorflow

```
Using TensorFlow backend.
x_train shape: (2000, 32, 32, 3)
compile done time cost: 0.0201s
Not using data augmentation.
Train on 1600 samples, validate on 400 samples
Epoch 1/5
8s - loss: 2.3083 - acc: 0.1200 - val_loss: 2.2859 - val_acc: 0.1775
Epoch 2/5
8s - loss: 2.2433 - acc: 0.1556 - val_loss: 3.2979 - val_acc: 0.0850
Epoch 3/5
7s - loss: 2.2435 - acc: 0.2025 - val_loss: 2.2271 - val_acc: 0.1350
Epoch 4/5
8s - loss: 2.0760 - acc: 0.2444 - val_loss: 2.0235 - val_acc: 0.3050
Epoch 5/5
7s - loss: 1.9984 - acc: 0.2819 - val_loss: 2.0313 - val_acc: 0.2675
```

* keras theano

```
Using TensorFlow backend.
Using Theano backend.
x_train shape: (2000, 32, 32, 3)
new x_train shape: (2000, 3, 32, 32)
compile done time cost: 0.0068s
Train on 1600 samples, validate on 400 samples
Epoch 1/5
21s - loss: 2.3293 - acc: 0.1087 - val_loss: 2.2779 - val_acc: 0.1575
Epoch 2/5
20s - loss: 2.2599 - acc: 0.1437 - val_loss: 2.2711 - val_acc: 0.1175
Epoch 3/5
21s - loss: 2.3440 - acc: 0.1738 - val_loss: 2.1322 - val_acc: 0.2650
Epoch 4/5
21s - loss: 2.0417 - acc: 0.2594 - val_loss: 1.9868 - val_acc: 0.2825
Epoch 5/5
21s - loss: 2.0170 - acc: 0.2756 - val_loss: 2.0316 - val_acc: 0.2550
```

* tensorlayer

* tflearn

```
x_train shape: (2000, 32, 32, 3)
2017-09-21 12:31:52.276316: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-21 12:31:52.276354: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-21 12:31:52.276363: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
compile done time cost: 0.4896s
---------------------------------
Run id: cifar10_cnn
Log directory: /tmp/tflearn_logs/
---------------------------------
Training samples: 2000
Validation samples: 400
--
Training Step: 16  | total loss: 23.04293 | time: 7.498s
| RMSProp | epoch: 001 | loss: 23.04293 - acc: 0.0029 | val_loss: 23.03292 - val_acc: 0.0000 -- iter: 2000/2000
--
Training Step: 32  | total loss: 23.03659 | time: 7.356s
| RMSProp | epoch: 002 | loss: 23.03659 - acc: 0.0104 | val_loss: 23.02797 - val_acc: 0.0000 -- iter: 2000/2000
--
Training Step: 48  | total loss: 23.03362 | time: 7.240s
| RMSProp | epoch: 003 | loss: 23.03362 - acc: 0.0408 | val_loss: 23.02669 - val_acc: 0.0100 -- iter: 2000/2000
--
Training Step: 64  | total loss: 23.03183 | time: 7.446s
| RMSProp | epoch: 004 | loss: 23.03183 - acc: 0.0735 | val_loss: 23.02651 - val_acc: 0.0400 -- iter: 2000/2000
--
Training Step: 80  | total loss: 23.03015 | time: 7.512s
| RMSProp | epoch: 005 | loss: 23.03015 - acc: 0.0908 | val_loss: 23.02634 - val_acc: 0.0675 -- iter: 2000/2000
--
```

* native tensorflow

* pytorch