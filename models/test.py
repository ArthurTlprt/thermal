from keras.applications import VGG16
import tensorflow as tf
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess)


train_dir = '../trimodal-people-segmentation/TrimodalDataset/Scene 1/SyncT'
validation_dir = '../trimodal-people-segmentation/TrimodalDataset/Scene 2/SyncT'

#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(640, 480, 3))


# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False


# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()



train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 10
val_batchsize = 10

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(640, 480),
        batch_size=train_batchsize,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(640, 480),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
#
# # Train the model
# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=train_generator.samples/train_generator.batch_size ,
#       epochs=30,
#       validation_data=validation_generator,
#       validation_steps=validation_generator.samples/validation_generator.batch_size,
#       verbose=1)

# Save the model
model.save('small_last4.h5')
