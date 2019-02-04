from keras import backend as K
from keras import layers, models, optimizers

class Actor:
    def __init__(self, state_size, action_size, low, high):
        self.state_size = state_size
        self.action_size = action_size
        self.low = low
        self.high = high
        self.range = high - low
        self.build_model()

    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')

        L2 = 0.01
        net = layers.Dense(units=512, kernel_regularizer=layers.regularizers.l2(L2))(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(units=256, kernel_regularizer=layers.regularizers.l2(L2))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net) 

        out = layers.Dense(units=self.action_size, activation='sigmoid', name='out', kernel_initializer=layers.initializers.RandomUniform(minval=-0.03, maxval=0.03))(net)

        actions = layers.Lambda(lambda x: (x * self.range) + self.low, name='actions')(out)

        self.model = models.Model(inputs=states, outputs=actions)

        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        optimizer = optimizers.Adam(lr=L2)
        updates_op = optimizer.get_updates(loss, self.model.trainable_weights)
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], outputs=[], updates=updates_op)