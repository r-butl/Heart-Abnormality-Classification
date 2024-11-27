import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, conv_configs, fc_configs, num_classes, dropout_rate, training):
        super(CNN, self).__init__()

        self.training = training
        self.conv_layers = []
        self.pool_layers = []
        
        # Build convolutional and pooling layers from conv_configs
        for config in conv_configs:
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=config['filters'], 
                    kernel_size=config['kernel_size'], 
                    strides=config['strides'], 
                    padding=config['padding'], 
                    activation=tf.nn.relu,
                    kernel_initializer=tf.compat.v1.glorot_normal_initializer()
                )
            )
            if 'pool_size' in config and 'pool_strides' in config:
                self.pool_layers.append(
                    tf.keras.layers.MaxPooling2D(
                        pool_size=config['pool_size'], 
                        strides=config['pool_strides'], 
                        padding=config.get('pool_padding', 'VALID')
                    )
                )
            else:
                self.pool_layers.append(None)
        
        # Build fully connected layers from fc_configs
        self.fc_layers = []
        for units in fc_configs:
            self.fc_layers.append(
                tf.keras.layers.Dense(
                    units=units, 
                    activation=tf.nn.relu, 
                    kernel_initializer=tf.compat.v1.glorot_normal_initializer()
                )
            )
            self.fc_layers.append(tf.keras.layers.Dropout(dropout_rate))
        
        # Output layer
        self.out = tf.keras.layers.Dense(
            units=num_classes, 
            activation='sigmoid',
            kernel_initializer=tf.compat.v1.glorot_normal_initializer()
        )

    def call(self, x):
        # Forward pass through convolutional layers
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            if pool:
                x = pool(x)
        
        # Flatten before fully connected layers
        x = tf.keras.layers.Flatten()(x)

        # Forward pass through fully connected layers
        for layer in self.fc_layers:
            x = layer(x) if not isinstance(layer, tf.keras.layers.Dropout) or self.training else x
        
        # Output layer
        x = self.out(x)
        return x