#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.datasets import imdb
(train_data, train_labels),(test_data,test_labels)= imdb.load_data(num_words=10000)


# In[3]:


train_data[0]


# In[4]:


train_labels[0]


# In[5]:


max([max (sequence) for sequence in train_data])


# In[6]:


word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# In[7]:


import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# In[8]:


x_train[0]


# In[9]:


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[10]:


from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[11]:


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[12]:


from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[13]:


from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])


# In[14]:


x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[15]:


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train, 
                    epochs=25,
                    batch_size=512, 
                    validation_data=(x_val, y_val))


# In[16]:


history_dict = history.history
history_dict.keys()


# In[18]:


import matplotlib.pyplot as plt

acc = history.history['acc']
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[19]:


plt.clf()
acc_values = history_dict['acc']
val_acc = history_dict['val_acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[20]:


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[21]:


results


# In[44]:


model.predict(x_test)


# In[45]:


from keras.preprocessing.text import text_to_word_sequence
text_file = open("endgame.txt","r")
lines = text_file.readlines()
print (lines)


# In[46]:


file = []
for c in lines:
    
    words = c.split(" ")
    print(words)
    #words = lines.split()
    #for words in lines:
        #file.append(words)
file = words


# In[47]:


file_size = len(file)
print(file_size)
type(file)
file=''.join(file)
type(file)


# In[48]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick
t=Tokenizer()
t.fit_on_texts(file)

input = hashing_trick(file,500, hash_function = 'md5')


# In[56]:


input[0]


# In[49]:


encoded_docs =  t.texts_to_matrix(file,mode = 'count')
print(encoded_docs)


# In[30]:


encoded_docs[0]


# In[50]:


word_index = imdb.get_word_index()


# In[51]:


import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_test = vectorize_sequences(input)


# In[31]:


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train, 
                    epochs=25,
                    batch_size=512, 
                    validation_data=(x_val, y_val))


# In[32]:


history_dict = history.history
history_dict.keys()


# In[33]:


import matplotlib.pyplot as plt

acc = history.history['acc']
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[34]:


plt.clf()
acc_values = history_dict['acc']
val_acc = history_dict['val_acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[35]:


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[36]:


results


# In[37]:


model.predict(x_test)


# In[38]:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_loss_values = history_dict['val_acc']
loss_values = history_dict['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:




