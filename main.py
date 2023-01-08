from train_function import *
import tensorflow as tf
import numpy as np

sample_size = 5478
test_size = 363
BATCH_SIZE = 128
BUFFER_SIZE = 2000


input_train, lon, lat = load_train_datasat(sample_size, plot_index)
train = tf.data.Dataset.from_tensor_slices(input_train)
train_dataset = train.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_input = load_test_datasat(test_size, plot_index_test)
test = tf.data.Dataset.from_tensor_slices(test_input)
test_dataset = test.batch(test_size)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

OUTPUT_CHANNELS = 1
LAMBDA = 10
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = Generator(OUTPUT_CHANNELS)
discriminator = Discriminator()

vgg = build_vgg()
vgg.trainable = False

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


EPOCHS = 200
@tf.function
def train_step(input_image, target, epoch, n, vgg_feature_gan, vgg_feature_real):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target, LAMBDA, loss_object, vgg_feature_gan, vgg_feature_real)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output, loss_object)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    return gen_total_loss, disc_loss

def fit(train_ds, epochs, test_ds, BATCH_SIZE):
    for epoch in range(epochs+1):
        if epoch%1 == 0:
            for example_test in test_ds.take(1):
                example_input = tf.convert_to_tensor(np.zeros((test_size, 226, 164, 3)))
                example_target = tf.convert_to_tensor(np.zeros((test_size, 226, 164, 1)))
                example_input = example_test[:, :, :, 0:3]
                example_target = example_test[:, :, :, 3]
                example_input = tf.cast(example_input, tf.float32)
                example_target = tf.cast(example_target, tf.float32)

        print("Epoch: ", epoch)

        for n, input_data in train_ds.enumerate():
            if n%1 == 0:
                print('.', end='')
            input = input_data[:, :, :, 0:3]
            label = input_data[:, :, :, 3]
            label = tf.expand_dims(label, axis=3)
            input = tf.cast(input,tf.float32)
            label = tf.cast(label, tf.float32)
            prediction = generator(input, training=True)
            label_np = label.numpy()
            prediction_np = prediction.numpy()
            aa = prediction_np.shape
            temp_label = np.zeros((aa[0], 224, 224, 3))
            temp_prediction = np.zeros((aa[0], 224, 224, 3))
            temp_label[:, :, 0:164, :] = label_np[:,0:224,:,:]
            temp_prediction[:, :, 0:164, :] = prediction_np[:,0:224,:,:]

            vgg_feature_gan = vgg.predict(temp_prediction)
            vgg_feature_real = vgg.predict(temp_label)

            gen_total_loss, disc_loss = train_step(input, label, epoch, n, vgg_feature_gan, vgg_feature_real)
            print('see loss', 'gen_total_loss : ', gen_total_loss.numpy(), 'disc_loss : ', disc_loss.numpy())
        print()

fit(train_dataset, EPOCHS, test_dataset, BATCH_SIZE)



