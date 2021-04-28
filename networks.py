import tensorflow as tf
import numpy as np
from tensorflow import keras
import itertools
import sys

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.debugging.set_log_device_placement(True)

# Compute probability distributions
def prob_dist(constraints, choices):
    sums = []
    for choice in choices:
        choice_sum = 0
        for constraint in constraints:
            choice_sum += np.dot(choice, constraint)
        sums.append(choice_sum)
    probs = tf.nn.softmax(tf.convert_to_tensor(np.array(sums), dtype=tf.double))
    return probs.numpy()

# Policy network
class Embedder(object):
    
    def __init__(self, obssize, embedsize, lr, decay=False):
        """
        obssize: size of the states
        embedsize: size of the embeddings
        """
        self.model = tf.keras.models.Sequential([
                  #TODO 
                  #input layer of size obssize
                  #intermediate layers
                  #output layer of size embedsize
                    #######tf.keras.Input(shape=(obssize,)),
                    #######tf.keras.layers.LSTM(61, input_shape=(2, 61)),
                    tf.keras.layers.Dense(128, activation='tanh'),
                    tf.keras.layers.Dense(64, activation='tanh'),
                    tf.keras.layers.Dense(embedsize, activation='tanh')
                ])
        
        # DEFINE THE OPTIMIZER
        if decay:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=90,
                decay_rate=0.70
            )
            #lr_decay = (lr/iters) * 7
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)#, decay=lr_decay)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # RECORD HYPER-PARAMS
        self.obssize = obssize
        self.embedsize = embedsize
    
    def compute_embeddings(self, states):
        """
        compute prob distribution over all actions given state: pi(s)
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples, embedsize]
        """
        #print(states.shape)
        embed_choices = tf.cast(self.model(states), tf.double)
        #print(len(prob))
        #print(prob.numpy().size)
        return embed_choices.numpy()
    
    def train(self, cons_states, choice_states, actions, actsize, Qs):
        """
        states: numpy array (states)
        actions: numpy array (actions)
        Qs: numpy array (estimated Q values)
        """
        with tf.GradientTape() as tape:
            
            # Get probs for each state, action pair
            prob_selected = []
            for ind, (const, choice) in enumerate(zip(cons_states, choice_states)):
                const_embedding = self.model(const)
                cho_embedding = self.model(choice)

                sums = tf.math.reduce_sum(tf.matmul(cho_embedding, const_embedding, transpose_b = True), axis=1)
                prob = tf.cast(tf.nn.softmax(sums, axis=-1), tf.double)

                action = actions[ind]
                prob_selected.append(prob[action] + 1e-8)

            # Compute loss
            loss = 0
            for i in range(len(prob_selected)):
                loss += Qs[i] * tf.math.log(prob_selected[i])
            loss *= -1
            loss /= len(prob_selected)

            print(loss)
            # BACKWARD PASS
            gradients = tape.gradient(loss, self.model.trainable_variables)  
            #print(gradients)
            # UPDATE
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
        return loss.numpy()
