from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from netCDF4 import Dataset
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam



def Guass_label(data):
    scale = MinMaxScaler()
    data_a = data.reshape(-1, 1)
    scale.fit(data_a)
    data_guass = scale.transform(data_a)
    data_original = scale.inverse_transform(data_guass)
    # data_guass[data_a == 0] = 0
    data_guass = data_guass.reshape((data.shape))
    data_original = data_original.reshape((data.shape))
    return data_guass,data_original

def Guass4(data, sample_size):
    # 分开都标准化到（0，1）范围

    inpv1_guass = np.zeros((sample_size, 226, 164, 3))
    inpv1_inverse = np.zeros((sample_size, 226, 164, 3))

    for i in range(3):
        scale = MinMaxScaler()
        data1 = data[:, :, :, i:i+1]
        data_a1 = data1.reshape(-1, 1)
        scale.fit(data_a1)
        data_guass1 = scale.transform(data_a1)
        data_original1 = scale.inverse_transform(data_guass1)
        data_guass1 = data_guass1.reshape((data1.shape))
        data_original1 = data_original1.reshape((data1.shape))
        inpv1_guass[:, :, :, i:i+1] = data_guass1

    return inpv1_guass, inpv1_inverse


def load_train_datasat(sample_size, plot_index):      # 读取训练数据集

    SSHA_train_file = Dataset('/train/train1.nc', 'r')  # 读入需要的数据
    SSTA_train_file = Dataset('/train/train2.nc', 'r')
    SST_train_file = Dataset('/train/train3.nc', 'r')
    ST50_train_file = Dataset('/train/train4.nc', 'r')

    lat = (SSHA_train_file.variables['lat'][:])
    lon = (SSHA_train_file.variables['lon'][:])

    inpv1 = np.zeros((sample_size, 226, 164, 3))
    inpv1[:, :, :, 0:1] = (SSHA_train_file.variables['SSHA'][0:sample_size, :, :]).reshape(sample_size, 226, 164,
                                                                                           1)  # (2847, 226, 164, 1)
    inpv1[:, :, :, 1:2] = (SSTA_train_file.variables['SSTA'][0:sample_size, :, :]).reshape(sample_size, 226, 164, 1)
    inpv1[:, :, :, 2:3] = (SST_train_file.variables['SST'][0:sample_size, :, :]).reshape(sample_size, 226, 164, 1)
    data_train_gass, data_train_gass_inverse = Guass4(inpv1, sample_size)

    label = np.zeros((sample_size, 226, 164, 1))
    label[:, :, :, :] = (ST50_train_file.variables['ST'][0:sample_size, :, :]).reshape(sample_size, 226, 164, 1)
    label_train_gass, label_train_gass_inverse = Guass_label(label)

    input = np.zeros((sample_size, 226, 164, 4))
    input[:, :, :, 0:3] = data_train_gass
    input[:, :, :, 3:4] = label_train_gass
    print(input.shape)

    return input, lon, lat


def load_test_datasat(test_size, plot_index):

    SSHA_test_file = Dataset('/test/test1.nc', 'r')
    SSTA_test_file = Dataset('/test/test2.nc', 'r')
    SST_test_file = Dataset('/test/test3.nc', 'r')
    ST50_test_file = Dataset('/test/test50.nc', 'r')

    lat = (SSHA_test_file.variables['lat'][:])
    lon = (SSHA_test_file.variables['lon'][:])

    inpv1 = np.zeros((test_size, 226, 164, 3))
    inpv1[:, :, :, 0:1] = (SSHA_test_file.variables['SSHA'][0:test_size, :, :]).reshape(test_size, 226, 164,
                                                                                           1)  # (2847, 226, 164, 1)
    inpv1[:, :, :, 1:2] = (SSTA_test_file.variables['SSTA'][0:test_size, :, :]).reshape(test_size, 226, 164, 1)
    inpv1[:, :, :, 2:3] = (SST_test_file.variables['SST'][0:test_size, :, :]).reshape(test_size, 226, 164, 1)
    data_train_gass, data_train_gass_inverse = Guass4(inpv1, test_size)

    label = np.zeros((test_size, 226, 164, 1))
    label[:, :, :, :] = (ST50_test_file.variables['ST'][0:test_size, :, :]).reshape(test_size, 226, 164, 1)
    label_train_gass, label_train_gass_inverse = Guass_label(label)

    input = np.zeros((test_size, 226, 164, 4))
    input[:, :, :, 0:3] = data_train_gass
    input[:, :, :, 3:4] = label_train_gass

    return input


def downsample(filters, size, stride, padding_flag=True, apply_batchnorm=True):
    #    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    if padding_flag:
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same',
                                   use_bias=False))
    else:
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=stride, padding='valid',
                                   use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, stride, padding_flag=True, apply_dropout=False):

    result = tf.keras.Sequential()

    if padding_flag:
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
                                            padding='same',
                                            use_bias=False))
    else:
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
                                        padding='valid',
                                        use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def build_vgg():

    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
    vgg = VGG19(weights="imagenet")

    optimizer = Adam(0.0002, 0.5)
    vgg.outputs = [vgg.layers[9].output]
    vgg.trainable = False
    vgg.compile(loss='mse',
                optimizer=optimizer,
                metrics=['accuracy'])

    img = Input(shape=[224, 224, 3])

    #vgg_size_img = re_size(img)

    # Extract image features
    img_features = vgg(img)

    return tf.keras.Model(img, img_features)


def Generator(OUTPUT_CHANNELS):    # Unet
    # downsample(filters, size, stride, padding_flag=True, apply_batchnorm=True)
    # padding_flag 为 True 时 padding='same'

    inputs = tf.keras.layers.Input(shape=[226, 164, 3])
    # 下采样

    down_stack = [

        downsample(filters=64, size=4, stride=2, padding_flag=True, apply_batchnorm=False),
        downsample(filters=128, size=(2,3), stride=1, padding_flag=False),
        downsample(filters=256, size=4, stride=2, padding_flag=True),
        downsample(filters=512, size=4, stride=2, padding_flag=True),
        downsample(filters=512, size=4, stride=2, padding_flag=True),
        downsample(filters=512, size=4, stride=2, padding_flag=True),
        downsample(filters=512, size=(6,4), stride=2, padding_flag=False),

        ]


    up_stack = [
        upsample(filters=512, size=(7,5), stride=2, padding_flag=False, apply_dropout=True),
        upsample(filters=512, size=4, stride=2, padding_flag=True, apply_dropout=True),
        upsample(filters=512, size=4, stride=2, padding_flag=True, apply_dropout=True),
        upsample(filters=256, size=4, stride=2, padding_flag=True),
        upsample(filters=128, size=4, stride=2, padding_flag=True),
        upsample(filters=64, size=(2,3), stride=1, padding_flag=False),

    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           activation='tanh')

    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    #x = tf.cast(x, tf.float64)

    return tf.keras.Model(inputs=inputs, outputs=x)

def penalize_term(gen_output, target):
    aa = tf.square(gen_output)
    print(aa)
    bb = tf.reduce_mean(aa, axis=0)
    print(bb)


def generator_loss(disc_generated_output, gen_output, target, LAMBDA, loss_object, vgg_feature_gan, vgg_feature_real):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    target = tf.cast(target, tf.float32)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    feature_loss = tf.reduce_mean(tf.abs(vgg_feature_gan - vgg_feature_real))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss) + feature_loss

    return total_gen_loss, gan_loss, l1_loss


def Discriminator():

    inp = tf.keras.layers.Input(shape=[226, 164, 3], name='input_data')
    tar = tf.keras.layers.Input(shape=[226, 164, 1], name='target_data')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 226, 164, channels = 3+1)

    down1 = downsample(filters=64, size=4, stride=2, padding_flag=True, apply_batchnorm=False)(x)
    down2 = downsample(filters=128, size=(2, 3), stride=1, padding_flag=False)(down1)
    down3 = downsample(filters=256, size=4, stride=2, padding_flag=True)(down2)
    down4 = downsample(filters=512, size=4, stride=2, padding_flag=True)(down3)

    conv = downsample(filters=512, size=4, stride=1, padding_flag=True)(down4)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    last = tf.keras.layers.Conv2D(1, 2, strides=1, padding='same')(leaky_relu)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output, loss_object):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
















