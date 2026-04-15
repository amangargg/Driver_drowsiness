#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input, Dense, LSTM, TimeDistributed,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[3]:


from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello Aman! Your project is live 🚀"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


# In[3]:


import os
import cv2
import numpy as np

IMG_SIZE = 224

def load_images(data_dir):
    data = []
    labels = []
    
    classes = ["drowsy", "non_drowsy"]
    
    for label, cls in enumerate(classes):
        path = os.path.join(data_dir, cls)
        
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                
                data.append(img)
                labels.append(label)
            except:
                continue
    
    return np.array(data), np.array(labels)


# In[5]:


import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

# Base path
base_path = os.path.expanduser("~/Desktop/dataset")

# Function to create a generator that yields batches as arrays
def image_batch_generator(folder_path, img_size=(128,128), batch_size=32, shuffle=True):
    class_names = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    paths_labels = []
    
    # Prepare (image_path, label) tuples
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(folder_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            paths_labels.append((img_path, label))
    
    if shuffle:
        np.random.shuffle(paths_labels)
    
    n = len(paths_labels)
    for i in range(0, n, batch_size):
        batch = paths_labels[i:i+batch_size]
        X_batch, y_batch = [], []
        for img_path, label in batch:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            img = img / 255.0
            X_batch.append(img)
            y_batch.append(label)
        if X_batch:  
            X_batch = np.array(X_batch, dtype=np.float32)
            y_batch = to_categorical(np.array(y_batch), num_classes=len(class_names))
            yield X_batch, y_batch


# In[7]:


def generator_to_array(generator):
    X_list, y_list = [], []
    for X_batch, y_batch in generator:
        X_list.append(X_batch)
        y_list.append(y_batch)
    X = np.vstack(X_list)
    y = np.vstack(y_list)
    return X, y

X_train, y_train = generator_to_array(image_batch_generator(os.path.join(base_path, "train"), batch_size=256))
X_val, y_val     = generator_to_array(image_batch_generator(os.path.join(base_path, "val"), batch_size=256, shuffle=False))
X_test, y_test   = generator_to_array(image_batch_generator(os.path.join(base_path, "test"), batch_size=256, shuffle=False))


# In[9]:


SEQ_LEN = 10

def create_sequences(X, y, seq_len=10):
    sequences = []
    seq_labels = []
    
    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])
        seq_labels.append(y[i+seq_len])
    
    return np.array(sequences), np.array(seq_labels)


# In[ ]:


X_train, y_train = create_sequences(X_train, y_train, SEQ_LEN)
X_val, y_val = create_sequences(X_val, y_val, SEQ_LEN)
X_test, y_test = create_sequences(X_test, y_test, SEQ_LEN)


# In[1]:


import os
import cv2
import numpy as np

IMG_SIZE = 224

def load_images(data_dir):
    data = []
    labels = []
    
    classes = ["drowsy", "non_drowsy"]
    
    for label, cls in enumerate(classes):
        path = os.path.join(data_dir, cls)
        
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                
                data.append(img)
                labels.append(label)
            except:
                continue
    
    return np.array(data), np.array(labels)


# In[5]:


IMG_SIZE = 96        
SEQ_LEN = 5          
BATCH_SIZE = 4      
EPOCHS = 15


# In[7]:


def load_images(data_dir):
    data = []
    labels = []
    
    classes = ["drowsy", "non_drowsy"]
    
    for label, cls in enumerate(classes):
        path = os.path.join(data_dir, cls)
        
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                
                data.append(img)
                labels.append(label)
            except:
                continue
    
    return np.array(data), np.array(labels)


# In[1]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[3]:


IMG_SIZE = 96
BATCH_SIZE = 4

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)


# In[11]:


base_path = os.path.expanduser("~/Desktop/dataset")
train_data = train_gen.flow_from_directory(
    os.path.join(base_path, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    os.path.join(base_path, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    os.path.join(base_path, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)


# In[13]:


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96, 96, 3),
    alpha=0.35
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)


# In[15]:


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


# In[17]:


model.evaluate(test_data)


# In[ ]:





# In[ ]:





# In[19]:


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96, 96, 3),
    alpha=0.35
)

# Freeze most layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

for layer in base_model.layers[-20:]:
    layer.trainable = True


# In[21]:


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)   
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)


# In[23]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[25]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2,
        min_lr=1e-6
    )
]


# In[27]:


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=callbacks
    
)


# In[29]:


model.evaluate(test_data)


# In[31]:


from sklearn.metrics import classification_report

y_pred = model.predict(test_data)
y_pred = (y_pred > 0.5).astype(int)

print(classification_report(test_data.classes, y_pred))


# In[33]:


print(train_data.class_indices)


# In[35]:


classification_report(test_data.classes, y_pred)


# In[37]:


test_data.reset()

y_pred = model.predict(test_data, verbose=0)
y_pred = (y_pred > 0.5).astype(int).flatten()

y_true = test_data.classes

print(classification_report(y_true, y_pred))


# In[41]:


print(train_data.class_indices)


# In[43]:


for layer in base_model.layers:
    layer.trainable = True   


# In[45]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[47]:


from sklearn.utils import class_weight
import numpy as np

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))
print(class_weights)


# In[49]:


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    class_weight=class_weights
)


# In[51]:


test_data.reset()

y_pred = model.predict(test_data, verbose=0)
y_pred = (y_pred > 0.5).astype(int).flatten()

y_true = test_data.classes

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))


# In[57]:


test_data = test_gen.flow_from_directory(
    os.path.join(base_path, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False   
)


# In[59]:


test_data.reset()


# In[61]:


y_pred = model.predict(test_data, verbose=0)
y_pred = (y_pred > 0.5).astype(int).flatten()

y_true = test_data.classes


# In[63]:


from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))


# In[67]:


from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt") 


# In[73]:


def detect_face(frame):
    results = yolo_model(frame)[0]
    
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
        
        # class 0 = person (we use upper body/face area)
        if cls == 0 and conf > 0.5:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            h = y2 - y1
            face = frame[y1:y1 + int(h * 0.5), x1:x2]  # upper half = face
            return face
    
    return None


# In[75]:


import cv2
import numpy as np

IMG_SIZE = 96

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    face = detect_face(frame)
    
    if face is not None:
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        
        pred = model.predict(face, verbose=0)[0][0]
        
        label = "DROWSY" if pred > 0.5 else "ALERT"
        
        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
    
    cv2.imshow("YOLO Drowsiness Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np
import math
import mediapipe as mp
from tensorflow.keras.models import load_model


cnn_model = load_model("light_model.h5")


from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions
)


base_options = BaseOptions(model_asset_path="face_landmarker.task")

options = FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

detector = FaceLandmarker.create_from_options(options)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_ear(eye_points):
    if len(eye_points) != 6:
        return 0

    A = distance(eye_points[1], eye_points[5])
    B = distance(eye_points[2], eye_points[4])
    C = distance(eye_points[0], eye_points[3])

    if C == 0:
        return 0

    return (A + B) / (2.0 * C)


def preprocess(frame):
    img = cv2.resize(frame, (96, 96))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


drowsy_frames = 0

#  Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        left_eye_pts = []
        right_eye_pts = []

        for idx in LEFT_EYE:
            lm = landmarks[idx]
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            left_eye_pts.append((x, y))

        for idx in RIGHT_EYE:
            lm = landmarks[idx]
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            right_eye_pts.append((x, y))

        # 🔹 EAR
        ear = (compute_ear(left_eye_pts) + compute_ear(right_eye_pts)) / 2.0

        # 🔹 CNN
        cnn_input = preprocess(frame)
        cnn_pred = cnn_model.predict(cnn_input, verbose=0)[0][0]

        # 🔥 HYBRID DECISION
        if ear < 0.25 or cnn_pred > 0.5:
            drowsy_frames += 1
        else:
            drowsy_frames = 0

        if drowsy_frames > 10:
            label = "DROWSY"
            color = (0, 0, 255)
        else:
            label = "ALERT"
            color = (0, 255, 0)

        # 🔹 Display
        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.putText(frame, f"CNN: {cnn_pred:.2f}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # 🔹 Draw eyes
        for (x, y) in left_eye_pts + right_eye_pts:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("Hybrid Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[121]:


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# In[131]:


import math

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])*2 + (p1[1]-p2[1])*2)

def compute_ear(eye_points):
    A = distance(eye_points[1], eye_points[5])
    B = distance(eye_points[2], eye_points[4])
    C = distance(eye_points[0], eye_points[3])

    return (A + B) / (2.0 * C)


# In[2]:


from tensorflow.keras.models import load_model

cnn_model = load_model("light_model.h5")


# In[145]:


def preprocess_eye(frame):
    img = cv2.resize(frame, (96, 96))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# In[ ]:




