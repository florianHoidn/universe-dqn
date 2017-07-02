import pyglet
import numpy as np
from pyglet.window import key
from universe import spaces
from threading import Thread

class PygletController:
    """This class allows you to take control of an OpenAI Universe game manually. 
    Note that it is already possible in OpenAI Universe to play the games yourself, e.g., 
    by connecting a vnc viewer to OpenAI's docker containers. 
    This class, however, allows you to input your commands via the env's step function. 
    In this way, an off-policy RL-algorithm can watch you play and learn from it.
    Sometimes it's nevertheless usefull to also visit "http://localhost:15900/viewer/?password=openai" in the browser 
    to see what the OpenAI auto starter is doing, because the PygletController will only render the frames 
    that your agent actually gets to see."""       
    def __init__(self, act, width=512, height=384, nbrFramesStacked=4):
        self._act = act
        self.width, self.height = width, height
        self.nbrFramesStacked = nbrFramesStacked
        self.currentOpenAiInput = None
        self.openAiFrame = None
        self.frameUpdated = False
        self.keys = ['up', 'left', 'down', 'right', 'x']
        self.key_events = [[[spaces.KeyEvent.by_name(key, down=down_or_up)]] for key in self.keys for down_or_up in [True, False]]     
        self.action_indices = {str(self.key_events[i]):i for i in range(len(self.key_events))}
        
        Thread(target=self.runPygletWindowThread).start()
        
    def runPygletWindowThread(self):
        self.window = pyglet.window.Window(width=self.width, height=self.height, visible=True)      
        
        @self.window.event
        def on_key_press(symbol, modifiers):  
            if key.symbol_string(symbol).lower() == 'enter':
                print("Human input disabled.")
                self.currentOpenAiInput = None
            elif key.symbol_string(symbol).lower() in self.keys:
                self.currentOpenAiInput = [self.action_indices[str([[spaces.KeyEvent.by_name(key.symbol_string(symbol).lower(), down=True)]])]]
                
        @self.window.event
        def on_key_release(symbol, modifiers):
            if key.symbol_string(symbol).lower() == 'enter':
                print("Human input disabled.")
                self.currentOpenAiInput = None            
            elif key.symbol_string(symbol).lower() in self.keys:
                self.currentOpenAiInput = [self.action_indices[str([[spaces.KeyEvent.by_name(key.symbol_string(symbol).lower(), down=False)]])]]
        
        @self.window.event
        def on_draw():
            if self.frameUpdated and self.openAiFrame is not None: 
                self.frameUpdated = False
                self.window.clear()
                self.window.switch_to()
                frameBytes = np.array(list(reversed(self.openAiFrame))).tobytes()
                image = pyglet.image.ImageData(width=self.width, height=self.height, format='L', data=frameBytes, pitch=self.width) #'RGB', frameBytes) # 'L' stands for luminosity and can be used for grayscale rendering.
                image.blit(0,0)
                
        def scheduledDraw(dt):
            on_draw()                
        
        pyglet.clock.schedule_interval(scheduledDraw, 0.034)        
        pyglet.app.run()
    
    def __call__(self, *args, **kwargs):
        if args is not None and args[0] is not None:
            self.openAiFrame = args[0][0][:,:,self.nbrFramesStacked-1:]
            self.frameUpdated = True
        
        if self.currentOpenAiInput is None:
            return self._act(*args, **kwargs)
        else:
            return self.currentOpenAiInput