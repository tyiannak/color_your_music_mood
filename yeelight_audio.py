import sys
import time
import numpy
import cv2
import argparse
import scipy.io.wavfile as wavfile
from pyAudioAnalysis import MidTermFeatures as mF
from pyAudioAnalysis import audioTrainTest as aT
import datetime
import signal
import pyaudio
import struct
from yeelight import Bulb

global fs
global all_data
global outstr
fs = 8000
FORMAT = pyaudio.paInt16


def signal_handler(signal, frame):
    """
    This function is called when Ctr + C is pressed and is used to output the
    final buffer into a WAV file
    """
    # write final buffer to wav file
    if len(all_data) > 1:
        wavfile.write(outstr + ".wav", fs, numpy.int16(all_data))
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def record_audio(block_size, devices, use_yeelight_bulbs=False, fs=8000):

    # initialize the yeelight devices:
    bulbs = []
    if use_yeelight_bulbs:
        for d in devices:
            bulbs.append(Bulb(d))

    # inialize recording process
    mid_buf_size = int(fs * block_size)
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=1, rate=fs,
                     input=True, frames_per_buffer=mid_buf_size)

    mid_buf = []
    count = 0
    global all_data
    global outstr
    all_data = []
    # initalize counters etc
    time_start = time.time()
    outstr = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M%p")
    out_folder = outstr + "_segments"

    # load segment model
    [classifier, MEAN, STD, class_names,
     mt_win, mt_step, st_win, st_step, _] = aT.load_model("model")

    [clf_energy, MEAN_energy, STD_energy, class_names_energy,
     mt_win_energy, mt_step_energy, st_win_energy, st_step_energy, _] = aT.load_model("energy")

    [clf_valence, MEAN_valence, STD_valence, class_names_valence,
     mt_win_valence, mt_step_valence, st_win_valence, st_step_valence, _] = aT.load_model("valence")


    while 1:
        block = stream.read(mid_buf_size)
        count_b = len(block) / 2
        format = "%dh" % (count_b)
        shorts = struct.unpack(format, block)
        cur_win = list(shorts)
        mid_buf = mid_buf + cur_win
        del cur_win
        if len(mid_buf) >= 5 * fs:

            # time since recording started:
            e_time = (time.time() - time_start)
            # data-driven time
            data_time = (count + 1) * block_size
            x = numpy.int16(mid_buf)
            seg_len = len(x)

            # extract features
            # We are using the signal length as mid term window and step,
            # in order to guarantee a mid-term feature sequence of len 1
            [mt_feats, _, names] = mF.mid_feature_extraction(x, fs,
                                                             seg_len,
                                                             seg_len,
                                                             round(fs * st_win),
                                                             round(fs * st_step))
            # extract features for music mood
            [mt_feats_2, _, _] = mF.mid_feature_extraction(x, fs,
                                                          round(fs * mt_win_energy),
                                                          round(fs * mt_win_energy),
                                                          round(fs * st_win_energy),
                                                          round(fs * st_step_energy))
            [mt_feats_3, _, _] = mF.mid_feature_extraction(x, fs,
                                                          round(fs * mt_win_valence),
                                                          round(fs * mt_win_valence),
                                                          round(fs * st_win_valence),
                                                          round(fs * st_step_valence))


            cur_fv = (mt_feats[:, 0] - MEAN) / STD
            cur_fv_2 = (mt_feats_2[:, 0] - MEAN_valence) / STD_valence
            cur_fv_3 = (mt_feats_3[:, 0] - MEAN_valence) / STD_valence

            # classify vector:
            [res, prob] = aT.classifier_wrapper(classifier, "svm_rbf",
                                                cur_fv)
            win_class = class_names[int(res)]

            [res_energy, prob_energy] = aT.classifier_wrapper(clf_energy,
                                                              "svm_rbf",
                                                              cur_fv_2)
            win_class_energy = class_names_energy[int(res_energy)]

            [res_valence, prob_valence] = aT.classifier_wrapper(clf_valence,
                                                              "svm_rbf",
                                                              cur_fv_2)
            win_class_valence = class_names_valence[int(res_valence)]


            print(win_class, win_class_energy, win_class_valence)
            if use_yeelight_bulbs:
                for b in bulbs:
                    if win_class == "silence":
                        b.set_hsv(50, 50, 20)
                    elif win_class == "other":
                        b.set_hsv(140, 50, 20)
                    elif win_class == "speech":
                        b.set_hsv(320, 100, 100)
                    elif win_class == "music":
                        b.set_hsv(0, 100, 100)

            win_prob = prob[int(res)]
            all_data += mid_buf
            mid_buf = numpy.double(mid_buf)
            mid_buf = []

            ch = cv2.waitKey(10)
            count += 1


def parse_arguments():
    record_analyze = argparse.ArgumentParser(description="Real time "
                                                         "audio analysis")
    record_analyze.add_argument("-d", "--devices", nargs="+",
                                help="IPs to Yeelight device(s) to use")
    record_analyze.add_argument("-bs", "--blocksize",
                                  type=float, choices=[0.25, 0.5, 0.75, 1, 2, 5],
                                  default=1, help="Recording block size")
    record_analyze.add_argument("-fs", "--samplingrate", type=int,
                                  choices=[4000, 8000, 16000, 32000, 44100],
                                  default=8000, help="Recording block size")
    return record_analyze.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    fs = args.samplingrate
    if fs != 8000:
        print("Warning! Segment classifiers have been trained on 8KHz samples. "
              "Therefore results will be not optimal. ")
    record_audio(args.blocksize, args.devices, False, fs)
