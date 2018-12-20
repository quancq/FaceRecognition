import keras
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.layers import Lambda, Input
from keras.models import Model, Sequential


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def get_siamese_model(model_base):
    input_shape = model_base.layers[0].get_input_shape_at(0)[1:]
    print("\nInput shape : ", input_shape)
    input_a = Input(input_shape)
    input_b = Input(input_shape)

    processed_a = model_base(input_a)
    processed_b = model_base(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    return model
