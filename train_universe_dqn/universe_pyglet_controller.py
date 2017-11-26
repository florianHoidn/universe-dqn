import pyglet
import numpy as np
from pyglet.window import key
from universe import spaces
from threading import Thread
from universe_env_wrapper import unwrap_openai_env

class PygletController:
    """This class allows you to take control of an OpenAI Universe game manually. 
    Note that it is already possible in OpenAI Universe to play the games yourself, e.g., 
    by connecting a vnc viewer to OpenAI's docker containers. 
    This class, however, allows you to input your commands via the env's step function. 
    In this way, an off-policy RL-algorithm can watch you play and learn from it.
    Sometimes it's nevertheless usefull to also visit "http://localhost:15900/viewer/?password=openai" in the browser 
    to see what the OpenAI auto starter is doing, because the PygletController will only render the frames 
    that your agent actually gets to see."""       
    def __init__(self, act, 
                 openai_env, 
                 width=512, 
                 height=384, 
                 nbr_frames_stacked=4):
        self._act = act
        self.width, self.height = width, height
        self.nbr_frames_stacked = nbr_frames_stacked
        self.openai_inputs_queue = []
        self.replace_last_in_queue = False # Due to the high input lag, one can't really store every input in the queue...
        self.last_input_was_mouse_motion = False #  ...especially mouse motions aren't processed fast enough for the game to remain playable.
        self.openai_frame = None
        self.frame_updated = False
        
        self.env = unwrap_openai_env(openai_env)
        
        self.pixel_skip = self.env.pixel_skip
        self.mouse_grid_stride = self.env.mouse_grid_stride        
        self.game_ulx, self.game_uly, self.game_lrx, self.game_lry = self.env.game_ulx, self.env.game_uly, self.env.game_lrx, self.env.game_lry
        
        self.keys = self.env.keys
        self.actions = self.env.actions      
        self.action_indices = {str(self.actions[i]):i for i in range(len(self.actions))}
               
        Thread(target=self.runPygletWindowThread).start()
        
    def runPygletWindowThread(self):
        self.window = pyglet.window.Window(width=self.width, height=self.height, visible=True)      
        
        @self.window.event
        def on_key_press(symbol, modifiers):
            if key.symbol_string(symbol).lower() == 'enter' and self._act is not None:
                print("Human input disabled.")
                self.openai_inputs_queue = []    
            elif key.symbol_string(symbol).lower() == 'plus':
                self.env.manual_reward += 10
                print("Increasing manual reward to "+str(self.env.manual_reward))
            elif key.symbol_string(symbol).lower() == 'minus':
                self.env.manual_reward -= 10
                print("Decreasing manual reward to "+str(self.env.manual_reward))
            elif key.symbol_string(symbol) in ['_0', 'NUM_0']:
                self.env.manual_reward = 0
                print("Resetting manual reward to 0")
            elif key.symbol_string(symbol).lower() in self.keys:
                next_openai_input = [self.action_indices[str([[spaces.KeyEvent.by_name(key.symbol_string(symbol).lower(), down=True)]])]]
                self.push_action_to_queue(next_openai_input)
            self.last_input_was_mouse_motion = False 
                
        @self.window.event
        def on_key_release(symbol, modifiers):
            if key.symbol_string(symbol).lower() == 'enter' and self._act is not None:
                print("Human input disabled.")
                self.openai_inputs_queue = []
            elif key.symbol_string(symbol).lower() in self.keys:
                next_openai_input = [self.action_indices[str([[spaces.KeyEvent.by_name(key.symbol_string(symbol).lower(), down=False)]])]]
                self.push_action_to_queue(next_openai_input)
            self.last_input_was_mouse_motion = False 
        
        @self.window.event
        def on_mouse_press(x, y, button, modifiers):
            if self.mouse_grid_stride != None:
                x_on_grid, y_on_grid = get_xy_on_grid(x, y)
                next_openai_input = [self.action_indices[str([[spaces.PointerEvent(x_on_grid, y_on_grid, 1)]])]] 
                self.push_action_to_queue(next_openai_input)
            self.last_input_was_mouse_motion = False 
                
        @self.window.event
        def on_mouse_motion(x, y, dx, dy):
            if self.mouse_grid_stride != None:
                x_on_grid, y_on_grid = get_xy_on_grid(x, y)                
                next_openai_input = [self.action_indices[str([[spaces.PointerEvent(x_on_grid, y_on_grid, 0)]])]] 
                self.replace_last_in_queue = self.last_input_was_mouse_motion
                self.push_action_to_queue(next_openai_input)
                self.replace_last_in_queue = False
            self.last_input_was_mouse_motion = True
        
        @self.window.event
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            self.env.manual_reward = self.env.manual_reward + scroll_y
                
        def get_xy_on_grid(x, y):
            x_scaled, y_scaled = x * self.pixel_skip, (self.height - y) * self.pixel_skip
            x_bounded = self.game_ulx if x_scaled <= self.game_ulx else self.game_lrx if x_scaled >= self.game_lrx else x_scaled
            y_bounded = self.game_uly if y_scaled <= self.game_uly else self.game_lry if y_scaled >= self.game_lry else y_scaled     
            x_on_grid = self.game_ulx + ((x_bounded - self.game_ulx)//self.mouse_grid_stride)*self.mouse_grid_stride
            y_on_grid = self.game_uly + ((y_bounded - self.game_uly)//self.mouse_grid_stride)*self.mouse_grid_stride            
            return x_on_grid, y_on_grid
            
        @self.window.event
        def on_draw():
            if self.frame_updated and self.openai_frame is not None: 
                self.frame_updated = False
                self.window.clear()
                self.window.switch_to()
                frameBytes = np.array(list(reversed(self.openai_frame))).tobytes()
                image = pyglet.image.ImageData(width=self.width, height=self.height, format='L', data=frameBytes, pitch=self.width) #'RGB', frameBytes) # 'L' stands for luminosity and can be used for grayscale rendering.
                image.blit(0,0)
                
        @self.window.event
        def on_close():
            self.window.close()
            
        def scheduledDraw(dt):
            on_draw()                
        
        pyglet.clock.schedule_interval(scheduledDraw, 0.034)        
        pyglet.app.run()
    
    def __call__(self, *args, **kwargs):
        if args is not None and args[0] is not None:
            self.openai_frame = args[0][0][:,:,self.nbr_frames_stacked-1:]
            self.frame_updated = True
        
        if not self.openai_inputs_queue and self._act is not None:
            return self._act(*args, **kwargs)
        else:
            return self.pop_action_from_queue() if self.openai_inputs_queue else 0  
            
    def push_action_to_queue(self, action):
        """Pushes the next next human input to the queue of actions that will be played next. 
        As it isn't necessary to input the same action multiple times, the action will only be pushed,
        if it is distinct from the last action that has been pushed."""
        if not self.openai_inputs_queue or self.openai_inputs_queue[-1] != action:
            if self.replace_last_in_queue:
                self.openai_inputs_queue[-1] = action
            else:
                self.openai_inputs_queue.append(action)
            
    def pop_action_from_queue(self):
        """Pops the next human input from the queue, unless it is the last input in the queue.
        Thus, the only way to empty the queue and thereby return control to the learning algo 
        is to hit return."""        
        if len(self.openai_inputs_queue)>1:
            return self.openai_inputs_queue.pop(0)
        else:
            return self.openai_inputs_queue[0]