import pandas as pd
import tensorflow as tf
import tf_keras
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification

# Loading dataset and label
df_fake = pd.read_csv("C:/Users/gorek/resume_projects/AI_Fake_News_detector/dataset/Fake.csv")
df_true = pd.read_csv("C:/Users/gorek/resume_projects/AI_Fake_News_detector/dataset/True.csv")

df_fake["label"] = 1
df_true["label"] = 0

# Making both datasets into one DataFrame and only use title
df = pd.concat([df_fake, df_true], ignore_index=True)
df["input"] = df["title"].fillna("") + " " + df["text"].fillna("")
df = df[["input", "label"]]

# Train-test split
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df["input"].tolist(), df["label"].tolist(), test_size=0.3, random_state=42
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors="tf")

# Create Tensorflow Datasets (training and testing)
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).shuffle(1000).batch(8)
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels)).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels)).batch(8)

# Loading dataset
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Compile the model
model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=3e-5), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(train_dataset, validation_data=val_dataset, epochs=3)

# Save Model
model.save_pretrained("./fake-news-model-tf")
tokenizer.save_pretrained("./fake-news-model-tf")

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
