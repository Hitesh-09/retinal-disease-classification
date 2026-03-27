import tensorflow as tf
from src.model import build_efficientnet_b4_classifier, build_densenet_classifier

# Define the weighted loss
def weighted_loss(y_true, y_pred):
    pos_weights = tf.constant([1.9570, 2.0435, 15.9470, 15.5092, 19.0628, 31.8971, 20.6135, 3.0673], dtype=tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    loss = -(pos_weights * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)

# Build models
eff_model = build_efficientnet_b4_classifier()
den_model = build_densenet_classifier()

# Compile with the loss
eff_model.compile(optimizer='adam', loss=weighted_loss, metrics=['AUC'])
den_model.compile(optimizer='adam', loss=weighted_loss, metrics=['AUC'])

# Save models with the required names
eff_model.save('models/efficientnet_b4_weighted.keras')
den_model.save('models/densenet_best.keras')

print("Models saved with correct names and loss function.")