import tensorflow as tf


def predict(image):
    modelCheckpointPath = '../checkpoints/my_model_1.h5'
    classNames = ['mug_with_logo', 'mug_without_logo']

    ########################################
    ### load model and predict
    ########################################
    # Recreate the exact same model, including its weights and the optimizer
    loaded_model = tf.keras.models.load_model(modelCheckpointPath)

    # Show the model architecture
    # loaded_model.summary()

    # predict

    img_height = 180
    img_width = 180
    #
    # prediction_path = 'dataset/Annotated_dataset/test_set/mug_without_logo/mug.003.jpg'
    # img = tf.keras.utils.load_img(
    #     prediction_path, target_size=(img_height, img_width)
    # )
    image = tf.image.resize(
        image,
        [img_height, img_width]
    )
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = loaded_model.predict(img_array)
    classes = predictions.argmax(axis=-1)
    score = tf.nn.softmax(predictions[0])

    # print(predictions)
    # print(classes)
    # print(score)
    return classNames[classes[0]]