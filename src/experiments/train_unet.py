import random
from tensorflow import keras
from src.preprocessing import ArealdData
from src.models import get_unet

# Paths
input_path = "./data/train/images"
target_path = "./data/train/masks"

# Dataset split
val_samples = 100
input_img_paths = sorted([os.path.join(input_path, fname) for fname in os.listdir(input_path) if fname.endswith(".tif")])
target_img_paths = sorted([os.path.join(target_path, fname) for fname in os.listdir(target_path) if fname.endswith(".tif")])

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)

train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Generators
train_gen = ArealdData(batch_size=3, img_size=(256,256), input_img_paths=train_input_img_paths, target_img_paths=train_target_img_paths)
val_gen   = ArealdData(batch_size=3, img_size=(256,256), input_img_paths=val_input_img_paths, target_img_paths=val_target_img_paths)

# Model
model = get_unet(img_size=(256,256), num_classes=10)
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [keras.callbacks.ModelCheckpoint("unet_segmentation.h5", save_best_only=True)]

model.fit(train_gen, epochs=20, validation_data=val_gen, callbacks=callbacks)
