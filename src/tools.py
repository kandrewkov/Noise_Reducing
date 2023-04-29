import tensorflow as tf

batching_size = 12000


def get_audio(path):
    print(path)
    audio, _ = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1)
    print(audio)
    # plt.plot(audio)
    # plt.show()
    return audio


def inference_preprocess(path):
    audio = get_audio(path)
    audio_len = audio.shape[0]
    batches = []
    for i in range(0, audio_len - batching_size, batching_size):
        batches.append(audio[i:i + batching_size])

    batches.append(audio[-batching_size:])
    diff = audio_len - (i + batching_size)
    return tf.stack(batches), diff

import matplotlib.pyplot as plt

def predict(model, path):
    test_data, diff = inference_preprocess(path)
    predictions = model.predict(test_data)
    final_op = tf.reshape(predictions[:-1], ((predictions.shape[0] - 1) * predictions.shape[1], 1))
    final_op = tf.concat((final_op, predictions[-1][-diff:]), axis=0)
    return final_op


import os


def wav_to_16_kHz(input_name, output_name, path):
    """
    Convert any audio format to wav format, need to install ffmpeeg
    """
    # output_name = output_name + '.wav'
    comand = f"ffmpeg -i {path}/{input_name} -ac 1 -ar 16000 {path}/{output_name}"

    # comand = "ffmpeg -i " + path +input_name + " -ac 1 -ar 16000" + " " + output_name

    if output_name in os.listdir(path):
        os.remove(os.path.join(path, output_name))
    # print(comand)
    os.system(comand)
    return output_name
