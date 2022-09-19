print("Script based on https://towardsdatascience.com/understand-and-implement-vision-transformer-with-tensorflow-2-0-f5435769093")
print("This script is not meant to be used as a solution, but rather as a working example to comprehensively examine building blocks of a vision transformer implemenation and how to train it.")

import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from itertools import islice, count

from building_blocks import *


##### Hyperparameters #################################################################################################

# architecture
transformer_layers = 6
patch_size = 4
hidden_size = 64
num_heads = 4
mlp_dim = 128

# training
EPOCHS = 2 # 120 on gpu

##### Load Data #######################################################################################################

# from cifar-10 website
class_types = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_lab_categorical = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='uint8')
test_lab_categorical = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='uint8')

train_im, valid_im, train_lab, valid_lab = train_test_split(x_train, train_lab_categorical, test_size=0.20,
                                                            stratify=train_lab_categorical,
                                                            random_state=40, shuffle = True) # stratify is unncessary

training_data = tf.data.Dataset.from_tensor_slices((train_im, train_lab))
validation_data = tf.data.Dataset.from_tensor_slices((valid_im, valid_lab))
test_data = tf.data.Dataset.from_tensor_slices((x_test, test_lab_categorical))

train_ds = training_data.shuffle(buffer_size=40000).batch(128).prefetch(buffer_size=tf.data.AUTOTUNE)
valid_ds = validation_data.shuffle(buffer_size=10000).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_data.shuffle(buffer_size=10000).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

print ('check shapes: ', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print ("train data shape after the split: ", train_im.shape)
print ('new validation data shape: ', valid_im.shape)
print ("validation labels shape: ", valid_lab.shape)
print ('train im and label types: ', type(train_im), type(train_lab))
print ('check types of train and val datasets: ', type(training_data), type(validation_data))

##### Patch Generation + Positional Encoding (V1) #####################################################################
# 1. Extract patches from image
# 2. Flatten patches
# 3. Dense layer with learnable weights to project to hidden dimension (Query and Key vector dimension)
# 4. Add learnable pos. embedding to projected vector

train_iter_7im, train_iter_7label = next(islice(training_data, 7, None)) # access the 7th element from the iterator
train_iter_7im = tf.expand_dims(train_iter_7im, 0)
train_iter_7label = train_iter_7label.numpy()

generate_patch_layer = generate_patch(patch_size=patch_size)
patches = generate_patch_layer(train_iter_7im)

patch_encoder = PatchEncode_Embed(64, 64)(patches)

print('train_iter_7im.shape: ', train_iter_7im.shape)
print ('patches per image and patches shape: ', patches.shape[1], '\n', patches.shape)
print ('shape of patch_encoder: ', tf.shape(patch_encoder))

render_image_and_patches(train_iter_7im, train_iter_7label, patches, class_types, patch_size)

##### Patch Generation + Positional Encoding (V2, original code) #######################################################
# 1. Directly use conv layer to extract patches and add learnable weights to project to hidden dimension (Query and Key vector dimension)
# (Number of filters equals hidden dimension)
# 2. Add learnable pos. embedding to projected vector

train_iter_7im = tf.cast(train_iter_7im, dtype=tf.float16)
generate_patch_conv_layer = generate_patch_conv(patch_size=patch_size)
patches_conv = generate_patch_conv_layer(train_iter_7im)

generate_patch_conv_orgPaper_layer = generate_patch_conv_orgPaper(patch_size=patch_size, hidden_size=64)
patches_conv_org = generate_patch_conv_orgPaper_layer(train_iter_7im)
patches_conv_org_f = generate_patch_conv_orgPaper_f(patch_size, hidden_size, train_iter_7im)

pos_embed_layer = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02))
embedded_inp = pos_embed_layer(patches_conv_org)

print ('patches per image and patches shape (patches_conv): ', patches_conv.shape[1], '\n', patches_conv.shape)
print ('patches per image and patches shape (patches_conv_org): ', patches_conv_org.shape[1], '\n', patches_conv_org.shape)
print ('patches per image and patches shape (patches_conv_org_f): ', patches_conv_org_f.shape[1], '\n', patches_conv_org_f.shape)
print ('embedded input shape: ', embedded_inp.shape)

render_image_and_patches(train_iter_7im, train_iter_7label, patches_conv, class_types, patch_size)

##### Transformer Encoder Block #######################################################################################

# test with the embeddings as input
Encoder1Dblock_layer_out_f = Encoder1Dblock_f(4, 32, embedded_inp)
print ('output shape of Encoder block when inputs are the embeddings: ', Encoder1Dblock_layer_out_f.shape)

##### Build Model, Compile & Train ################################################################################

ViT_model = build_ViT(train_shape=train_im.shape[1:],
                      patch_size=patch_size,
                      hidden_size=hidden_size,
                      transformer_layers=transformer_layers,
                      mlp_dim=mlp_dim,
                      num_heads=num_heads,
                      num_output_units=len(class_types))

#ViT_model.summary()

ViT_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
                  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"), tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5 acc')])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                              patience=5, min_lr=1e-5, verbose=1)

train_session = ViT_model.fit(train_ds,
                        epochs = EPOCHS,
                        validation_data=valid_ds, callbacks=[reduce_lr])

# Visualization
plot_learning_curves(train_session)
pred_class_resnet50 = ViT_model.predict(x_test)
conf_matrix(pred_class_resnet50)
