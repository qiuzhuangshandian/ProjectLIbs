import tensorflow as tf
import numpy as np
import time

class Model(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, units):
    super(Model, self).__init__()
    self.units = units

    # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    if tf.test.is_gpu_available():
      self.gru = tf.keras.layers.CuDNNGRU(self.units,
                                          return_sequences=True,
                                          recurrent_initializer='glorot_uniform',
                                          stateful=True)
    else:
      # self.lstm = tf.keras.layers.lstm()
      self.gru = tf.keras.layers.GRU(self.units,
                                     return_sequences=True,
                                     recurrent_activation='sigmoid',
                                     recurrent_initializer='glorot_uniform',
                                     stateful=True)

    self.fc = tf.keras.layers.Dense(vocab_size)

  def call(self, x):
    # embedding = self.embedding(x)

    # output at every time step
    # output shape == (batch_size, seq_length, hidden_size)
    output = self.gru(x)

    # The dense layer will output predictions for every time_steps(seq_length)
    # output shape after the dense layer == (seq_length * batch_size, vocab_size)
    prediction = self.fc(output)
    
    # states will be used to pass at every step to the model while training
    return prediction


if __name__=="__main__":
    # Length of the vocabulary in chars
    # vocab_size = len(vocab)
    vocab_size = 100
    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    units = 1024
    model = Model(vocab_size, embedding_dim, units)
    # Using adam optimizer with default arguments
    optimizer = tf.train.AdamOptimizer()
    # Using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
    def loss_function(real, preds):
        return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)

    X = np.array([[[1,2]],[[3,4]],[[5,6]]],dtype=np.float32)
    X = tf.convert_to_tensor(X)
    y = model(X)
    print(y)
    print("ok!")


    # Training loop
  for epoch in range(EPOCHS):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()

    for (batch, (inp, target)) in enumerate(dataset):
          with tf.GradientTape() as tape:
              # feeding the hidden state back into the model
              # This is the interesting step
              predictions = model(inp)
              loss = loss_function(target, predictions)

          grads = tape.gradient(loss, model.variables)
          optimizer.apply_gradients(zip(grads, model.variables))

          if batch % 100 == 0:
              print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,
                                                            batch,
                                                            loss))
    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
      model.save_weights(checkpoint_prefix)

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))