#!/usr/bin/env python3
import websocket
import config
import json
import numpy as np
from src.data_methods import *
from src.leap_methods import collect_frame
from src.classes import *
import random
import tkinter as tk
import matplotlib.pyplot as plt
# import tensorflow as tf
from tensorflow_core.python import keras
from itertools import cycle

# dead bird from https://www.flickr.com/photos/9516941@N08/3180449008
# useful tutorials on blitting:
# https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot
# https://bastibe.de/2013-05-30-speeding-up-matplotlib.html
# Without blitting, matplotlib lags badly.
# The model predicting too often can also cause things to lag a bit. Predicting every 20 frames is safe at present, but could change, depending on the size of the model used.
# capture every nth frame, slowing input to roughly 25fps
n = 4

#### set up gui

blit = True
root = tk.Tk()
# Create another window for settings
settings_window = tk.Toplevel()
settings_window.geometry("600x400")
settings_gui = SettingsGUI(settings_window)

##### set up model and prediction

model = None
# load the prediction model
for file in os.scandir('models/prediction_model'):
    if file.path[-2:] == 'h5':
        model = keras.models.load_model(file)
assert model != None, 'No h5 file found in prediction model folder'

model_path = 'models/prediction_model/'
# mapping of gestures to integers: need this for decoding model output
gestures, g2idx, idx2g = get_gestures(version='prediction', path=model_path)
# set whether or not to derive features and drop unused VoI
derive_features = True
# which hands will be used in predicting?
hands = ['left', 'right']
# Get the VoI that are used as predictors
VoI = get_VoI(path=model_path)
if derive_features:
    VoI_drop = get_VoI_drop(path=model_path)
    VoI_predictors = [v for v in VoI if v not in VoI_drop]
    # use these to get VoI, labelled by hand
    VoI_predictors = [hand + '_' + v for v in VoI_predictors for hand in hands]
else:
    VoI = [hand + '_' + v for v in VoI for hand in hands]

# get dictionary with one and two handed derived variables to use in prediction
derived_feature_dict = get_derived_feature_dict(path=model_path)
# get mean and standard deviation dictionaries
# these are used for standardizing input to model
with open(model_path + 'means_dict.json', 'r') as f:
    means_dict = json.load(f)
with open(model_path + 'stds_dict.json', 'r') as f:
    stds_dict = json.load(f)

# no of frames to keep stored in memory for prediction
keep = model.input.shape[-2]
# set up circular buffer for storing model input data
model_input_data = CircularBuffer((model.input.shape[-2],model.input.shape[-1]))

# keep track of total no. of frames received
frames_total = 0
# no. captured continuously
frames_recorded = 0

# store previous frame, just in case a hand drops out, and we need to vill in values
previous_frame = None
# indicator the next successfully received frame will be the first
first_two_handed_frame = True

#### initialize variables for fury, angularity, and pred confidence calculations

raw_fury = 0
fury = 0
previous_fury = 0
angularity = 0
raw_angularity = 0
pred_confidence = 0
adjusted_raw_pred_confidence = 0

# set up storage for fury and angularity history
x_axis_range = settings_gui.settings['x axis range']
angularity_cb = CircularBuffer((x_axis_range,))
fury_cb = CircularBuffer((x_axis_range,))
# also need history of prediction confidence
confidence_cb = CircularBuffer((x_axis_range,))

colour = cycle('bgrcmk')
current_label = plt.text(x_axis_range - 3,0.0,'no_gesture', bbox={'facecolor': next(colour), 'alpha': 0.3, 'pad': 5})
# old labels that drift left
old_labels = []
# old labels that have finished drifting and are stationary
stationary_labels = []
gesture = 'no_gesture'
gesture_change = False

websocket_cache = {}

while True:
    frames_total += 1

    packed_frame = collect_frame(frames_total, n, websocket_cache)
    if len(packed_frame) > 0:
        frames_recorded += 1
        frame_index = (frames_recorded - 1) % keep
        # length of packed frame is ~350 if one hand present, ~700 for two
        if len(packed_frame) < 400 and previous_frame == None:
            print('need two hands to start')
        # if we have at least one hand, and a previous frame to supplement any missing hand data, then we can proceed
        else:
            # if a hand is missing, fill in the data from the previous frame
            if len(packed_frame) < 400:
                previous_frame.update(packed_frame)
                packed_frame = previous_frame.copy()
                if frames_total % 5 == 0:
                    print('Warning: a hand is missing')
                if not hand_missing:
                    hand_missing = True

            else:
                hand_missing = False
                previous_frame = packed_frame.copy()
            # get the derived features
            if derive_features:
                new_features = features.get_derived_features(packed_frame, derived_feature_dict)
                packed_frame.update(new_features)

            # if this is the first two handed frame, generate a sorted list of variables used for prediction, including derived features
            if first_two_handed_frame:
                first_two_handed_frame = False
                # for the first frame only, set previous complete frame to the current frame
                previous_complete_frame = packed_frame.copy()
                if derive_features:
                    all_predictors = VoI_predictors + [pred for pred in new_features.keys()]
                    all_predictors.sort()
                else:
                    all_predictors = sorted(VoI)
                #print(all_predictors)
                # create new gesture label if needed
                if gesture_change == True:
                    old_labels.append(current_label)
                    current_label = plt.text(x_axis_range - 3,pred_confidence,gesture, bbox={'facecolor': next(colour), 'alpha': 0.3, 'pad': 5})
                    gesture_change = False
                else:
                    current_label.set_position((x_axis_range - 3, pred_confidence))
                for old_l in old_labels:
                    pos = old_l.get_position()
                for stationary_l in stationary_labels:

                    stationary_l.remove()


            previous_complete_frame = packed_frame.copy()

            frame = np.empty(model.input.shape[-1])
            for i, p in enumerate(all_predictors):
                frame[i] = (packed_frame[p] - means_dict[p]) / stds_dict[p]
            if frames_total % settings_gui.settings['every nth frame to model'] == 0:
                model_input_data.add(frame)

            # make a prediction every pred_interval number of frames
            # but first ensure there is a complete training example's worth of consecutive frames
            if frames_recorded >= keep and frames_recorded % settings_gui.settings['prediction interval'] == 0:
                # feed example into model, and get a prediction
                pred = model.predict(np.expand_dims(model_input_data.get(), axis=0))
                # sometimes getting nan at the start. Need to find source of this.
                if idx2g[np.argmax(pred)] != gesture:
                    gesture_change = True
                    gesture = idx2g[np.argmax(pred)]
                if pred[0][np.argmax(pred)] > settings_gui.settings['min conf. to change image']:
                    if not hand_missing:
                        print(idx2g[np.argmax(pred)])
                    # gui.gesture.set(idx2g[np.argmax(pred)].replace('_', ' '))

