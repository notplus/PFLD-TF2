import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
import sys

sys.path.insert(0, './')
import tfrecord


TRAIN_DATASET = None

def rep_data_gen():
    img_arrays = []
    for features in TRAIN_DATASET.take(100):
        input_array = features[0].numpy()[0]
        img_arrays.append(input_array)
    img_arrays = np.array(img_arrays)
    for input_value in tf.data.Dataset.from_tensor_slices(img_arrays).batch(1).take(100):
        yield [input_value]


def quant_model_to_int8(saved_model_dir):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()

    with open(saved_model_dir + '/model_integer_only_quant.tflite', 'wb') as f:
        f.write(tflite_quant_model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--saved_model_folder', help='The saved model folder dir.', required=True, type=str)
    parser.add_argument('--train_tfrec', help='The train tfrecord dir.', required=True, type=str)
    args = parser.parse_args()

    TRAIN_DATASET = tfrecord.get_dataset(args.train_tfrec, batch_size=1)
    quant_model_to_int8(args.saved_model_folder)
