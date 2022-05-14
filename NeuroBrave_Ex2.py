from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.model_selection import train_test_split
import scipy.io
import numpy as numpy

features = scipy.io.loadmat('FeatureMat_timeWin.mat')['features']
img = scipy.io.loadmat('images.mat')['img']
subjNum = scipy.io.loadmat('trials_subNums.mat')['subjectNum'][0]
Label = (features[:, -1] - 1).astype(int)
method = ''  # Overall of per subject methods


def testtrain(img, label):
    train, test = train_test_split(range(0, len(img)), test_size=0.3)
    train_images = img[train]
    train_labels = Label[train]
    test_images = img[test]
    test_labels = Label[test]

    # Normalize the images.
    train_images = (train_images / numpy.max(train_images))
    test_images = (test_images / numpy.max(test_images))

    # Set input shape
    sample_shape = train_images[0].shape
    img_width, img_height = sample_shape[1], sample_shape[2]
    input_shape = (img_width, img_height, 3)

    # Reshape data
    train_images = train_images.reshape(len(train_images), input_shape[0], input_shape[1], input_shape[2])
    test_images = test_images.reshape(len(test_images), input_shape[0], input_shape[1], input_shape[2])
    return train_images, train_labels, test_images, test_labels


def runPerSubj(img, label):
    train_images, train_labels, test_images, test_labels = testtrain(img, label)
    model = buildModel()
    # Train the model.
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_split=0.3,
        validation_data=(test_images, test_labels),
    )
    # Predict on the first 5 test images.
    predictions = model.predict(test_images)
    predictionsToCompare = numpy.argmax(predictions, axis=1)
    # Print our model's predictions.
    print(predictionsToCompare[:5])  #
    # Check our predictions against the ground truths.
    print(test_labels[:5])  #
    return numpy.average(test_labels == predictionsToCompare)


def buildModel():
    num_filters = 8  # filters that the convolutional layer will learn.
    kernel_size = 3  # specifying the width and height of the 2D convolution window.
    pool_size = 1  # reduce the spatial dimensions of the output volume.
    dilation_rate = (5, 5)  # a basic convolution only applied to the input volume with defined gaps
    # Build the model.
    model = Sequential([
        Conv2D(num_filters,
               kernel_size,
               input_shape=(32, 32, 3),
               strides=(1, 1),  # specifying the “step” of the convolution along the x and y axis of the input volume.
               padding='same',  # zero-padded
               dilation_rate=dilation_rate,
               ),
        MaxPooling2D(pool_size=(pool_size, pool_size)),  # reduce the spatial dimensions of the output volume.
        Flatten(),
        Dense(512, activation='relu'),
        Dense(16, activation='softmax'),
    ])
    model.summary()
    # # Compile the model.
    model.compile(
        'Nadam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def loadWeights(img, label):
    train_images, train_labels, test_images, test_labels = testtrain(img, label)
    model = buildModel()
    model.load_weights('cnn.h5')
    # Predict on the first 5 test images.
    predictions = model.predict(test_images)
    predictionsToCompare = numpy.argmax(predictions, axis=1)
    # Print our model's predictions.
    print(predictionsToCompare[:5])  #
    # Check our predictions against the ground truths.
    print(test_labels[:5])  #
    return numpy.average(test_labels == predictionsToCompare)


# Overall
if method == 'overall':
    train_images, train_labels, test_images, test_labels = testtrain(img, Label)
    model = buildModel()
    # Train the model.
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_split=0.3,
        validation_data=(test_images, test_labels),
    )
    # Save the model to disk.
    model.save_weights('cnn.h5')

    # Predict on the first 5 test images.
    predictions = model.predict(test_images)
    predictionsToCompare = numpy.argmax(predictions, axis=1)
    # Print our model's predictions.
    print(predictionsToCompare[:5])  #
    # Check our predictions against the ground truths.
    print(test_labels[:5])  #
    print(numpy.average(test_labels == predictionsToCompare))
else:  # Per Subject
    individualAccuracy  = [0] * numpy.max(subjNum)
    overallAccuracy = [0]*numpy.max(subjNum)
    for subject in range(1, numpy.max(subjNum)):
        labelPos = numpy.where(subjNum == subject)
        if numpy.size(labelPos):
            accuracy = runPerSubj(img[labelPos], Label[labelPos])
            individualAccuracy[subject] = accuracy
            print("Subject number: {0} intra accuracy learning is: {1}%".format(subject, str(accuracy * 100)))
            accuracyLoadWeights = loadWeights(img[labelPos], Label[labelPos])
            overallAccuracy[subject] = accuracyLoadWeights
            print(
                "Subject number: {0} inter accuracy learning is: {1}%".format(subject, str(accuracyLoadWeights * 100)))
            if accuracyLoadWeights < accuracy:
                print("Subject number: {0} individual accuracy is better by : {1}%"
                      .format(subject, str(accuracy - accuracyLoadWeights)))
            else:
                print("Subject number: {0} overall accuracy is better by : {1}%"
                      .format(subject, str(accuracyLoadWeights - accuracy)))
        else:
            print("Subject number: " + str(subject) + " does not have any scans")
    print("inter algorithm preferable for {0} subjects".format(numpy.sum(numpy.greater(individualAccuracy, overallAccuracy))))
    print("Algorithms are equal at {0} subjects".format(numpy.sum(numpy.equal(individualAccuracy, overallAccuracy))))
    print("Average inter accuracy {0}".format(individualAccuracy[individualAccuracy!=0].mean()))
    print("Average intra accuracy {0}".format(overallAccuracy[overallAccuracy!=0].mean()))
