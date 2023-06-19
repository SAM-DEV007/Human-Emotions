import tensorflow as tf

import os


def main():
    '''It is the main function to train the model'''

    # Path
    train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Dataset\\Train\\'
    validation_dir = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Dataset\\Validate\\'

    # Save Path
    model_save = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Model_Data\\Model.h5'

    # Defaults
    batch = 32
    img_size = (48, 48)

    # Datasets
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, labels='inferred', shuffle=True, batch_size=batch, image_size=img_size)
    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, labels='inferred', shuffle=True, batch_size=batch, image_size=img_size)

    # Test Dataset
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    # Performance
    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=autotune)
    validation_dataset = validation_dataset.prefetch(buffer_size=autotune)
    test_dataset = test_dataset.prefetch(buffer_size=autotune)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])

    img_shape = img_size + (3,)
    base_model = tf.keras.applications.ResNetRS50(input_shape=img_shape, include_top=False, weights='imagenet')
    base_model.trainable = True

    for layer in base_model.layers[:200]:
        layer.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(7, activation='softmax')

    # Base Model
    base_model.summary()

    # Customizing the base model
    inputs = tf.keras.Input(shape=(48, 48, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    
    model.summary() # Customized model

    initial_epochs = 100

    # Callbacks
    red_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=2)
    es_callback = tf.keras.callbacks.EarlyStopping(patience=4, verbose=1)

    # Train the model
    model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset, callbacks=[red_callback, es_callback])

    model.save(model_save) # Save the model


if __name__ == '__main__':
    main()
