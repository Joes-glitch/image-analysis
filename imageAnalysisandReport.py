from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data['image'], train_data['label'], epochs=10, batch_size=32, validation_data=(test_data['image'], test_data['label']))

test_loss, test_acc = model.evaluate(test_data['image'], test_data['label'])
print(f'Test accuracy: {test_acc:.2f}')

def generate_report(image):
    image = np.array(image.split(' ')).reshape(256, 256, 3)
    predictions = model.predict(image)
    report = ''
    if predictions[0][0] > 0.5:
        report += 'The patient has a high risk of disease.\n'
    else:
        report += 'The patient has a low risk of disease.\n'
    return report
image = 'medical_image.jpg'
report = generate_report(image)
print(report)
