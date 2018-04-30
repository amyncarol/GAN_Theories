import os
import numpy as np
import shutil
import sys
from PIL import Image
import scipy.misc
from glob import glob
import tensorflow as tf 
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FOLDERS = ['/Users/yao/large_data_file_no_sync/imaterials/train_chunk0', 
            '/Users/yao/large_data_file_no_sync/imaterials/train_chunk1',
            '/Users/yao/large_data_file_no_sync/imaterials/train_chunk2',
            '/Users/yao/large_data_file_no_sync/imaterials/train_chunk3',
            '/Users/yao/large_data_file_no_sync/imaterials/train_chunk4',
            '/Users/yao/large_data_file_no_sync/imaterials/train_chunk5',
            '/Users/yao/large_data_file_no_sync/imaterials/train_chunk6',
            '/Users/yao/large_data_file_no_sync/imaterials/valid']

PATH = '/Users/yao/large_data_file_no_sync/imaterials'

IMAGE_SIZE = 64

def get_folder_size(path):
    return sum([len(files) for r, d, files in os.walk(path)]) - 1

def get_class_size(i):
    counter = 0
    for folder in FOLDERS:
        subfolder = os.path.join(folder, str(i))
        counter += len(os.listdir(subfolder))
    return counter

def largest_n_class(n):
    counter = np.zeros(128)
    for i in range(1, 129):
        counter[i-1]=get_class_size(i)
    index = np.argsort(counter)
    return index[-n:][::-1]+1

def generate_class_folder(i):
    os.mkdir(os.path.join(PATH, str(i)))
    for folder in FOLDERS:
        files = os.listdir(os.path.join(folder, str(i)))
        for file in files:
            file_path = os.path.join(folder, str(i)+'/'+file)
            if os.path.isfile(file_path):
                shutil.copy(file_path, os.path.join(PATH, str(i)))

def resize_images_in_folder(folder, folder_resized, resize_h=64):
    imgs_name = [i for i in os.listdir(folder) if i.endswith('jpeg')]

    if not os.path.exists(folder_resized):
        os.makedirs(folder_resized)

    counter = 0
    for img_name in imgs_name:
        img = scipy.misc.imread(os.path.join(folder, img_name)).astype(np.float)
        resize_w = resize_h
        resized_image = scipy.misc.imresize(img,[resize_h, resize_w])
        scipy.misc.imsave(os.path.join(folder_resized, img_name), resized_image)

        counter += 1
        if counter % 100 == 0:
            print(counter)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecord(class_i, tfrecord_folder):
    start_time = time.time()
    counter = 0
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_folder, str(class_i)+'.tfrecord'))
    for folder in FOLDERS:
        images = glob(os.path.join(folder, str(class_i)+'/*.jpeg'))
        for image in images:
            img = Image.open(image)
            img = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
            label = class_i
            feature = {'label': _int64_feature(label),
                'image': _bytes_feature(img.tostring())}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            counter += 1
            if counter % 100 == 0:
                print('processed {} image, which takes {} sec'.format(counter, time.time()-start_time))
    writer.close()

def parser(record):
    """
    Use `tf.parse_single_example()` to extract data from a `tf.Example`
    protocol buffer, and perform any additional per-record preprocessing.
    """
    keys_to_features = {
        'image': tf.FixedLenFeature([], tf.string, default_value=""),
        'label': tf.FixedLenFeature([], tf.int64,
                                default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    image = tf.decode_raw(parsed["image"], tf.uint8)
    image = tf.cast(tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3]), tf.float32)/255.0
    label = tf.cast(parsed["label"], tf.int32)

    return image, label

def data2fig(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

if __name__=='__main__':
    # class_i = largest_n_class(5)
    # for i in class_i:
    #   generate_class_folder(i)
    #   print(get_class_size(i))
    #for i in [12, 42, 92, 125]:
    #   resize_images_in_folder('/Users/yao/Google Drive/projects_ml/GAN_Theories/Datas/imaterials_original/'+str(i),\
    #       '/Users/yao/Google Drive/projects_ml/GAN_Theories/Datas/imaterials/'+str(i))

    #write_tfrecord(20, './Datas/imaterials_tfrecord')

    filenames = ["./Datas/imaterials_tfrecord/20.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser)
    dataset = dataset.batch(16)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    with tf.Session() as sess:
        img, lab = sess.run([images, labels])
        print(img)
        fig = data2fig(img)
        plt.savefig('test.png', bbox_inches='tight')
        plt.close(fig)





    