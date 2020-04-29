import sys
import numpy
import argparse
import scipy.io.wavfile as wavfile
from pyAudioAnalysis import MidTermFeatures as mF
from pyAudioAnalysis import audioTrainTest as aT
import datetime
import signal
import pyaudio
import struct
from yeelight import Bulb
import cv2
import color_map_2d

global fs
global all_data
global outstr
fs = 8000
FORMAT = pyaudio.paInt16

"""
Load 2D image of the valence-arousal representation and define coordinates
of emotions and respective colors
"""
img = cv2.cvtColor(cv2.imread("music_color_mood.png"),
                   cv2.COLOR_BGR2RGB)
colors = {"coral": [255, 127, 80], "pink": [255, 192, 203],
          "orange": [255, 165, 0], "blue": [0, 0, 205],
          "green": [0, 205, 0], "red": [205, 0, 0], "yellow": [204, 204, 0]}
angry_pos = [-0.8, 0.5]
fear_pos = [-0.3, 0.8]
happy_pos = [0.6, 0.6]
calm_pos = [0.4, -0.5]
sad_pos = [-0.6, -0.4]
emo_map = color_map_2d.create_2d_color_map([angry_pos, fear_pos, happy_pos,
                                            calm_pos, sad_pos],
                                           [colors["red"], colors["yellow"],
                                            colors["orange"], colors["green"],
                                            colors["blue"]],
                                           img.shape[0], img.shape[1])
new_img = cv2.addWeighted(img, 0.4, emo_map, 0.8, 0)


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

    # initialize recording process
    mid_buf_size = int(fs * block_size)
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=1, rate=fs,
                     input=True, frames_per_buffer=mid_buf_size)

    mid_buf = []
    count = 0
    global all_data
    global outstr
    all_data = []
    outstr = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M%p")

    # load segment model
    [classifier, mu, std, class_names,
     mt_win, mt_step, st_win, st_step, _] = aT.load_model("model")

    [clf_energy, mu_energy, std_energy, class_names_energy,
     mt_win_en, mt_step_en, st_win_en, st_step_en, _] = \
        aT.load_model("energy")

    [clf_valence, mu_valence, std_valence, class_names_valence,
     mt_win_va, mt_step_va, st_win_va, st_step_va, _] = \
        aT.load_model("valence")

    while 1:
        block = stream.read(mid_buf_size)
        count_b = len(block) / 2
        format = "%dh" % (count_b)
        shorts = struct.unpack(format, block)
        cur_win = list(shorts)
        mid_buf = mid_buf + cur_win
        del cur_win
        if len(mid_buf) >= 5 * fs:
            # data-driven time
            x = numpy.int16(mid_buf)
            seg_len = len(x)

            # extract features
            # We are using the signal length as mid term window and step,
            # in order to guarantee a mid-term feature sequence of len 1
            [mt_f, _, _] = mF.mid_feature_extraction(x, fs,
                                                     seg_len,
                                                     seg_len,
                                                     round(fs * st_win),
                                                     round(fs * st_step))
            # extract features for music mood
            [mt_f_2, _, _] = mF.mid_feature_extraction(x, fs,
                                                       round(fs * mt_win_en),
                                                       round(fs * mt_step_en),
                                                       round(fs * st_win_en),
                                                       round(fs * st_step_en))
            [mt_f_3, _, _] = mF.mid_feature_extraction(x, fs,
                                                       round(fs * mt_win_va),
                                                       round(fs * mt_step_va),
                                                       round(fs * st_win_va),
                                                       round(fs * st_step_va))

            fv = (mt_f[:, 0] - mu) / std
            fv_2 = (mt_f_2[:, 0] - mu_energy) / std_energy
            fv_3 = (mt_f_3[:, 0] - mu_valence) / std_valence

            # classify vector:
            [res, prob] = aT.classifier_wrapper(classifier, "svm_rbf", fv)
            win_class = class_names[int(res)]

            [res_energy, p_en] = aT.classifier_wrapper(clf_energy, "svm_rbf",
                                                       fv_2)
            win_class_energy = class_names_energy[int(res_energy)]

            [res_valence, p_val] = aT.classifier_wrapper(clf_valence, "svm_rbf",
                                                         fv_3)
            win_class_valence = class_names_valence[int(res_valence)]

            soft_energy = p_en[class_names_energy.index("high")] - \
                          p_en[class_names_energy.index("low")]
            soft_valence = p_val[class_names_valence.index("positive")] - \
                           p_val[class_names_valence.index("negative")]

            print(win_class, win_class_energy, win_class_valence,
                  soft_valence, soft_energy)

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

            h, w, _ = img.shape
            y_center, x_center = int(h / 2), int(w / 2)
            x = x_center + int((w/2) * soft_valence)
            y = y_center - int((h/2) * soft_energy)

            new_img_2 = new_img.copy()
            new_img_2[y-10:y+10, x-10:x+10] = [200, 50, 10]


#            cv2.putText(img, "x", (y_center, x_center),
#                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

            cv2.imshow('Signal', new_img_2)
            ch = cv2.waitKey(10)
            count += 1


def parse_arguments():
    record_analyze = argparse.ArgumentParser(description="Real time "
                                                         "audio analysis")
    record_analyze.add_argument("-d", "--devices", nargs="+",
                                help="IPs to Yeelight device(s) to use")
    record_analyze.add_argument("-bs", "--blocksize",
                                  type=float,
                                choices=[0.25, 0.5, 0.75, 1, 2, 5],
                                  default=1, help="Recording block size")
    record_analyze.add_argument("-fs", "--samplingrate", type=int,
                                  choices=[4000, 8000, 16000, 32000, 44100],
                                  default=8000, help="Recording block size")
    return record_analyze.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    fs = args.samplingrate
    if fs != 8000:
        print("Warning! Segment classifiers have been trained on 8KHz samples."
              " Therefore results will be not optimal. ")
    record_audio(args.blocksize, args.devices, False, fs)
