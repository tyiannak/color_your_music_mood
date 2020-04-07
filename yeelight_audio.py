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


def record_audio(block_size, fs=8000):

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

    while 1:
        try:
            block = stream.read(mid_buf_size)
            count_b = len(block) / 2
            format = "%dh" % (count_b)
            shorts = struct.unpack(format, block)
            cur_win = list(shorts)
            mid_buf = mid_buf + cur_win
            del cur_win

            if 1:
                # time since recording started:
                e_time = (time.time() - time_start)
                # data-driven time
                data_time = (count + 1) * block_size
                x = numpy.int16(mid_buf)
                seg_len = len(x)

                # extract features
                # We are using the signal length as mid term window and step,
                # in order to guarantee a mid-term feature sequence of len 1
                [mt_feats, _, _] = mF.mid_feature_extraction(x, fs,
                                                             seg_len,
                                                             seg_len,
                                                             round(fs * st_win),
                                                             round(fs * st_step)
                                                             )
                cur_fv = (mt_feats[:, 0] - MEAN) / STD
                # classify vector:
                [res, prob] = aT.classifier_wrapper(classifier, "svm_rbf",
                                                    cur_fv)
                win_class = class_names[int(res)]
                win_prob = prob[int(res)]
                print(win_class, win_prob, x.shape)
                mid_buf = numpy.double(mid_buf)

                mid_buf = []
                ch = cv2.waitKey(10)
                count += 1
        except IOError:
            print("Error recording")


def parse_arguments():
    record_analyze = argparse.ArgumentParser(description="Real time "
                                                         "audio analysis")
    record_analyze.add_argument("-bs", "--blocksize",
                                  type=float, choices=[0.25, 0.5, 0.75, 1, 2],
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
    record_audio(args.blocksize, fs)
