# AI_Fake_News_Detector

First test I got this:

3929/3929 [==============================] - 14455s 4s/step - loss: 0.6977 - accuracy: 0.5118 - val_loss: 0.6931 - val_accuracy: 0.5275
Epoch 2/5
3929/3929 [==============================] - 14452s 4s/step - loss: 0.6931 - accuracy: 0.5215 - val_loss: 0.6931 - val_accuracy: 0.5275
Epoch 3/5
3929/3929 [==============================] - 14656s 4s/step - loss: 0.6931 - accuracy: 0.5215 - val_loss: 0.6931 - val_accuracy: 0.5275
Epoch 4/5
3929/3929 [==============================] - 13819s 4s/step - loss: 0.6931 - accuracy: 0.5215 - val_loss: 0.6931 - val_accuracy: 0.5275
Epoch 5/5
3929/3929 [==============================] - 13734s 3s/step - loss: 0.6931 - accuracy: 0.5215 - val_loss: 0.6931 - val_accuracy: 0.5275
842/842 [==============================] - 916s 1s/step - loss: 0.6931 - accuracy: 0.5253
Test Accuracy: 52.53%
Test Loss: 0.6931

second test where i did title + text as well as setting a max_length for the encoders and lower the learning rate to 3e-5 and epoch to 3

3929/3929 [==============================] - 27038s 7s/step - loss: 0.5479 - accuracy: 0.6773 - val_loss: 0.6931 - val_accuracy: 0.4725
Epoch 2/3
3929/3929 [==============================] - 25918s 7s/step - loss: 0.6931 - accuracy: 0.4782 - val_loss: 0.6931 - val_accuracy: 0.4725
Epoch 3/3
3929/3929 [==============================] - 25209s 6s/step - loss: 0.6931 - accuracy: 0.4785 - val_loss: 0.6931 - val_accuracy: 0.4725
842/842 [==============================] - 1857s 2s/step - loss: 0.6931 - accuracy: 0.4747
Test Accuracy: 47.47%
Test Loss: 0.6931