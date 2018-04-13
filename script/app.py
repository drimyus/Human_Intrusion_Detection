import os
from kivy import Config
from kivy.app import App


Config.read(os.path.expanduser('~/.kivy/config.ini'))
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '600')
Config.set('graphics', 'resizable', True)

from screens.start import StartScreen


class HIMD_APP(App):
    def build(self):
        self.title = "Human Intruders Detection App"
        return StartScreen()


if __name__ == '__main__':
    HIMD_APP().run()
