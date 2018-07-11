import gym
from gym import space,logger
from gym.utils import seeding
import numpy as np

class Catcher(object):

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.viewer = None
        self.action_space = spaces.Discrete(2)
        # self.state init
        self.reset()
        self.seed()

    def reset(self):
        n = np.random.randint(0, self.grid_size-1, size=1)
        m = np.random.randint(1, self.grid_size-2, size=1)
        self.state = np.asarray([0, n, m])[np.newaxis]
        # f0 = 0 at the top

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _draw_state(self):
        im_size = (self.grid_size,)*2
        state = self.state[0]

        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket

        mode = 'rgb_array'
        scale = 50
        screen_width = self.grid_size * scale
        screen_height = self.grid_size * scale
        fruit_width = 1 * scale
        basket_width = 3 * scale

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #draw fruit
            fruit = rendering.make_circle(scale/2)
            self.fruittrans = rendering.Transform()
            fruit.add_attr(self.fruittrans)
            fruit.set_color(.5,.5,.8)
            self.viewer.add_geom(fruit)

            #draw basket
            l,r,t,b = -basket_width/3, basket_width*2/3, scale, 0
            basket = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            basket.set_color(.8,.6,.4)
            self.baskettrans = rendering.Transform()
            basket.add_attr(self.baskettrans)
            self.viewer.add_geom(basket)

            """
            for col in range(0, self.grid_size):
                self.track = rendering.Line((scale*col,0), (scale*col, screen_height))
                self.track.set_color(.9,.8,.7)
                self.viewer.add_geom(self.track)

            for row in range(0, self.grid_size):
                self.track = rendering.Line((0, scale*row), (screen_width, scale*row))
                self.track.set_color(.9,.8,.7)
                self.viewer.add_geom(self.track)
            """

        if self.state is None: return None

        fruit_x = (state[1] + 0.5) * scale
        fruit_y = (self.grid_size - 0.5 - state[0]) * scale
        
        basket_x = state[2] * scale
        basket_y = 0
        self.fruittrans.set_translation(fruit_x, fruit_y)
        self.baskettrans.set_translation(basket_x, basket_y)

        self.viewer.render(return_rgb_array = mode=='rgb_array')

        return canvas

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size-2)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out

    def step(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over, {}
    def observe(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0


def main():
	a = Catcher()

if __name__ == '__main__':
	main()