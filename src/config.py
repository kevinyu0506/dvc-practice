import os

data_dir = "data"
model_dir = "model"
feature_dir = os.path.join(model_dir, 'feature')
log_dir = "log"

train_data_path = os.path.join(data_dir, 'train')
validation_data_path = os.path.join(data_dir, 'validation')
test_data_path = os.path.join(data_dir, 'test')

top_model_weights_file = os.path.join(model_dir, 'model.h5')
top_model_json_file = os.path.join(model_dir, 'model.json')

train_feature_file = os.path.join(feature_dir, 'bottleneck_features_train.npy')
valid_feature_file = os.path.join(feature_dir, 'bottleneck_features_validation.npy')
test_feature_file = os.path.join(feature_dir, 'bottleneck_features_test.npy')

metrics_file = os.path.join(log_dir, 'metrics.csv')
evaluation_json_file = 'evaluation.json'

# dimensions of our images.
img_width, img_height = 150, 150

nb_train_samples = 1000
nb_validation_samples = 800
nb_test_samples = 200
batch_size = 10
epochs = 10
