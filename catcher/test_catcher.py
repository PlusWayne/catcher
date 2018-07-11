import json
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from train_catcher import Catch
import time


if __name__ == "__main__":
    # Make sure this grid size matches the value used fro training
    grid_size = 10

    with open("model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("model.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    env = Catch(grid_size)
    c = 0
    for e in range(1000):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        c += 1
        while not game_over:
            time.sleep(.1)
            input_tm1 = input_t

            # get next action
            q = model.predict(input_tm1)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            c += 1
        print("Epoch {:03d}/999 | Reward {}".format(e, reward))
