import atexit
import os
import sys
import threading
import traceback

import cloudpickle
import gym
import numpy as np
import warnings
import time

import common


class GymWrapper:

  def __init__(self, env, obs_key='image', act_key='action'):
    self._env = env
    self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_is_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    if self._obs_is_dict:
      spaces = self._env.observation_space.spaces.copy()
    else:
      spaces = {self._obs_key: self._env.observation_space}
    return {
        **spaces,
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    if self._act_is_dict:
      return self._env.action_space.spaces.copy()
    else:
      return {self._act_key: self._env.action_space}

  def step(self, action):
    if not self._act_is_dict:
      action = action[self._act_key]
    obs, reward, done, info = self._env.step(action)
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = float(reward)
    obs['is_first'] = False
    obs['is_last'] = done
    obs['is_terminal'] = info.get('is_terminal', done)
    return obs

  def reset(self):
    obs = self._env.reset()
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = 0.0
    obs['is_first'] = True
    obs['is_last'] = False
    obs['is_terminal'] = False
    return obs


class DMC:
  # check suite.ALL_TASKS to see matching domains with tasks
  # For stacker stack_2:
    # observations: (arm_pos, arm_vel, touch, hand_pos, box_pos, box_vel, target_pos)
  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    self.fingertips = [13,14,17,18]
    self.boxes = [19,20,21,21]
    self.current_step = 0
    self.learning_phase = 'init'

    os.environ['MUJOCO_GL'] = 'egl'
    self.interesting_geom = [13,14,17,18]
    domain, task = name.split('_', 1)
    self.task = task
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if domain == 'manip':
      from dm_control import manipulation
      self._env = manipulation.load(task + '_vision')
    elif domain == 'locom':
      from dm_control.locomotion.examples import basic_rodent_2020
      self._env = getattr(basic_rodent_2020, task)()
    else:
      from dm_control import suite
      self._env = suite.load(domain, task)
    self._action_repeat = action_repeat
    self._size = size
    if camera in (-1, None):
      camera = dict(
          quadruped_walk=2, quadruped_run=2, quadruped_escape=2,
          quadruped_fetch=2, locom_rodent_maze_forage=1,
          locom_rodent_two_touch=1,
      ).get(name, 0)
    self._camera = camera
    self._ignored_keys = []
    for key, value in self._env.observation_spec().items():
      #print("possible observations:\n")
      #print(key)
      if value.shape == (0,):
        print(f"Ignoring empty observation key '{key}'.")
        self._ignored_keys.append(key)

  @property
  def obs_space(self):
    spaces = {
        'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'grab_reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'stacking_reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'target_pos_reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'log_contacts': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'log_contact_forces': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'log_box_pos_z_mean': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32)
    }
    for key, value in self._env.observation_spec().items():
      '''
      _env.observation_spec().items() returns that:
      key: position, value: Array(shape=(4,), dtype=dtype('float64'), name='position')
      key: velocity, value: Array(shape=(3,), dtype=dtype('float64'), name='velocity')
      key: touch, value: Array(shape=(2,), dtype=dtype('float64'), name='touch')
      key: target_position, value: Array(shape=(2,), dtype=dtype('float64'), name='target_position')
      key: dist_to_target, value: Array(shape=(), dtype=dtype('float64'), name='dist_to_target')
      '''
      if key in self._ignored_keys:
        continue
      if value.dtype == np.float64:
        spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
      elif value.dtype == np.uint8:
        spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
      else:
        raise NotImplementedError(value.dtype)
    return spaces

  @property
  def act_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return {'action': action}

  def calculate_contacts(self, number_contacts):
    n_boxes = int(self.task.split("_")[1])
    box_names = ['box' + str(b) for b in range(n_boxes)]
    _FAR = .065
    reward = 0
    contacts = 0
    contact_forces = 0
    sim = self._env.physics
    fingertips = self.fingertips
    boxes = self.boxes
    # fingertips = [13,14,17,18]
    # boxes = [19,20,21,21]
    touched_boxes = []
    fingers_involved = []
    
    for i in range(number_contacts):
        contact = sim.data.contact[i]
        con_object1 = contact.geom1
        con_object2 = contact.geom2

        # swap if needed
        if con_object2 in fingertips:
            con_object1, con_object2 = con_object2, con_object1

        # TODO do we need to check the contact_force
        # any(sim.data.contact_force(i)[0] > 0) and
        if (con_object1 in fingertips) and (con_object2 in boxes):
            #print('One finger and one box involved')
            contacts += 1
            contact_forces += np.sum(sim.data.contact_force(i)[0])
            # exactly one finger and one box is part of contact 
            # (we don't want fingers to touch each other)

            if len(fingers_involved) == 0:
                # save first finger and box

                fingers_involved.append(con_object1)
                # append also the other part of the finger
                if con_object1 + 1 in fingertips:
                    fingers_involved.append(con_object1 + 1)
                if con_object1 - 1 in fingertips:
                    fingers_involved.append(con_object1 - 1)

                touched_boxes.append(con_object2)
                
            else:
                # one finger is already involved
                if (con_object1 not in fingers_involved) and (con_object2 in touched_boxes):
                    # new finger is involved and box is already touched
                    box_name = sim.model.id2name(con_object2, 'geom')
                    box_pos_z = sim.named.data.geom_xpos[box_name, 'z']
                    box_pos_x = sim.named.data.geom_xpos[box_name, 'x']
                    if box_pos_z > 0.0655 and box_pos_z<0.3 and box_pos_x>(-0.682843+0.3) and box_pos_x<(0.682843-0.3):
                        # not touching other boxes
                        distance_other = [sim.site_distance(box_name, box2) for box2 in box_names if box2 != box_name]
                        if np.min(distance_other) > _FAR:
                            reward = 1
                            return reward, contacts, contact_forces

                elif con_object2 not in touched_boxes:
                    # new box is touched and we don't care about which finger is involved
                    touched_boxes.append(con_object2)

    return reward, contacts, contact_forces


  def calculate_grab_reward_contactbased(self, learn_force=False, learn_lift=False):
    # thumb start pos=".03 0 .045"
    # finger start pos="-.03 0 .045"
    # -> tumb finger left, thumb right -> but flip once because in the air 

    number_contacts = self._env.physics.data.ncon

    n_boxes = int(self.task.split("_")[1])
    box_names = ['box' + str(b) for b in range(n_boxes)]
    far = .1
    reward = 0
    contacts = 0
    contact_forces = 0
    sim = self._env.physics
    fingertips = self.fingertips
    boxes = self.boxes
    # fingertips = [13,14,17,18]
    # boxes = [19,20,21,21]
    touched_boxes = []
    fingers_involved = []
    
    for i in range(number_contacts):
        contact = sim.data.contact[i]
        con_object1 = contact.geom1
        con_object2 = contact.geom2

        # swap if needed
        if con_object2 in fingertips:
            con_object1, con_object2 = con_object2, con_object1

        # TODO do we need to check the contact_force
        # any(sim.data.contact_force(i)[0] > 0) and
        if (con_object1 in fingertips) and (con_object2 in boxes):
            #print('One finger and one box involved')
            contacts += 1
            contact_forces += np.sum(sim.data.contact_force(i)[0])
            # exactly one finger and one box is part of contact 
            # (we don't want fingers to touch each other)

            if len(fingers_involved) == 0:
                # save first finger and box
                fingers_involved.append(con_object1)
                # append also the other part of the finger
                if con_object1 + 1 in fingertips:
                    fingers_involved.append(con_object1 + 1)
                if con_object1 - 1 in fingertips:
                    fingers_involved.append(con_object1 - 1)
                touched_boxes.append(con_object2)
                
            else:
                # one finger is already involved
                if (con_object1 not in fingers_involved) and (con_object2 in touched_boxes):
                    box_name = sim.model.id2name(con_object2, 'geom')
                    box_pos_z = sim.named.data.geom_xpos[box_name, 'z']
                    box_pos_x = sim.named.data.geom_xpos[box_name, 'x']

                    hand_pos = sim.body_2d_pose('hand')[:2]
                    hand_pos_x = hand_pos[0]
                    distance_x = abs(box_pos_x - hand_pos_x)

                    thumb_pos = sim.body_2d_pose('thumbtip')[:2]
                    thumb_pos_x = thumb_pos[0]
                    finger_pos = sim.body_2d_pose('fingertip')[:2]
                    finger_pos_x = finger_pos[0]

                    if distance_x < 0.022 and ((thumb_pos_x >= box_pos_x and finger_pos_x <= box_pos_x) \
                      or (thumb_pos_x <= box_pos_x and finger_pos_x >= box_pos_x)):
                        if learn_lift:
                            # box has to be lifted to get reward
                            box_name = sim.model.id2name(con_object2, 'geom')
                            box_pos_z = sim.named.data.geom_xpos[box_name, 'z']
                            box_pos_x = sim.named.data.geom_xpos[box_name, 'x']
                            if box_pos_z > 0.04:
                                # not touching other boxes
                                # distance_other = [sim.site_distance(box_name, box2) for box2 in box_names if box2 != box_name]
                                # np.min(distance_other) > 0.1 and 
                                
                                reward = 1
                                return reward, contacts, contact_forces
                        else:
                            reward = 1
                            return reward, contacts, contact_forces

                elif con_object2 not in touched_boxes:
                    # new box is touched and we don't care about which finger is involved
                    touched_boxes.append(con_object2)

    return reward, contacts, contact_forces

  def finger_close_reward(self, close=.02, not_left_and_right=False):
    sim = self._env.physics
    n_boxes = int(self.task.split("_")[1])
    box_names = ['box' + str(b) for b in range(n_boxes)]
    box_pos = sim.body_2d_pose(box_names)[:,:2]
    box_pos_x = box_pos[:,0]
    hand_pos = sim.body_2d_pose('hand')[:2]
    hand_pos_x = hand_pos[0]
    hand_pos_z = hand_pos[1]

    thumb_pos = sim.body_2d_pose('thumbtip')[:2]
    thumb_pos_x = thumb_pos[0]
    finger_pos = sim.body_2d_pose('fingertip')[:2]
    finger_pos_x = finger_pos[0]
    
    distances_x = []
    for box1, id in zip(box_names, range(n_boxes)):
      # site: grasp, pinch
      # one finger left one finger right
      distance_x = np.abs([box_pos_x[id] - hand_pos_x])
      distances_x.append(distance_x)
    
      # pinch close to box and hand above box
      if sim.site_distance('pinch', box1) < close and distance_x < close:
          # finger to either side of box 
          if not_left_and_right or ((thumb_pos_x >= box_pos_x[id] and finger_pos_x <= box_pos_x[id]) \
            or (thumb_pos_x <= box_pos_x[id] and finger_pos_x >= box_pos_x[id])):
              reward = 1
              return reward, distance_x

    return 0.0, np.min(distances_x)


  def learn_to_grab_reward(self, current_step):
    if self.learning_phase == 'init':
        # learn to be close to box
        contacts = self.calculate_grab_reward_contactbased(learn_lift=False)
        output = self.finger_close_reward(close=.03, not_left_and_right=True)
        return (output[0], contacts[1], output[1])
    elif self.learning_phase == 'close':
        # learn to be close to box
        contacts = self.calculate_grab_reward_contactbased(learn_lift=False)
        output = self.finger_close_reward()
        return (output[0], contacts[1], output[1])
    elif self.learning_phase == 'grab':
        # learn contact with box
        return self.calculate_grab_reward_contactbased(learn_lift=False)
    elif self.learning_phase == 'lift':
        # learn lift box
        return self.calculate_grab_reward_contactbased(learn_lift=True)
    elif self.learning_phase == 'hover':
        return self.calculate_grab_reward_contactbased(learn_lift=True)
        # learn hover box
        # return self.calculate_box2target_reward(drop=False), 0.0, 0.0
    elif self.learning_phase == 'drop':
        # learn drop box
        return self.calculate_box2target_reward(drop=True), 0.0, 0.0
    else:
        raise ValueError('Unknown learning phase: {}'.format(self.learning_phase))
    

  def calculate_box2target_reward(self, drop=False):
    # check if box is above target
    sim = self._env.physics
    target_pos = sim.body_2d_pose('target')[:2]
    target_pos_z = target_pos[1]
    target_pos_x = target_pos[0]
    n_boxes = int(self.task.split("_")[1])
    box_names = ['box' + str(b) for b in range(n_boxes)]
    box_pos = sim.body_2d_pose(box_names)[:,:2]
    box_pos_z = box_pos[:,1]
    box_pos_x = box_pos[:,0]

    for box1, id in zip(box_names, range(n_boxes)):
      if box_pos_z[id] > target_pos_z and np.abs(box_pos_x[id] - target_pos_x) < 0.02:
        
        if drop:
          # check that no contact between box and finger
          ncon = sim.data.ncon
          for i in range(ncon):
            con = self._env.physics.data.contact[i]
            con_object1 = con.geom1
            con_object2 = con.geom2
            if con_object1 in self.fingertips or con_object2 in self.fingertips:
              return 0
        else:
          return 1

    return 0


  def calculate_grab_reward_distance(self):
    # ideas: increase distance to wall, increase distance to other boxes
    _CLOSE = .03    # (Meters) Distance below which a thing is considered close.
    _FAR = .065       # (Meters) Distance above which a thing is considered far.
    # site: grasp, pinch
    sim = self._env.physics
    n_boxes = int(self.task.split("_")[1])
    #print("Box Pos Z Values:")
    box_names = ['box' + str(b) for b in range(n_boxes)]
    box_pos = sim.body_2d_pose(box_names)[:,:2]
    box_pos_z = box_pos[:,1]
    box_pos_x = box_pos[:,0]

    for box1, id in zip(box_names, range(n_boxes)):
      if sim.site_distance('pinch', box1) < _CLOSE \
        and box_pos_z[id] > 0.0655 and box_pos_z[id]<0.3 and box_pos_x[id]>(-0.682843+0.3) and box_pos_x[id]<(0.682843-0.3):
          # not close to any other box
          distance_other = [sim.site_distance(box1, box2) for box2 in box_names if box2 != box1]
          if np.min(distance_other) > _FAR:
              reward = 1
              return reward

    return 0.0

  def calculate_grab_reward(self):
    # ideas: increase distance to wall, increase distance to other boxes
    _CLOSE = .03    # (Meters) Distance below which a thing is considered close.
    _FAR = .065       # (Meters) Distance above which a thing is considered far.
    # site: grasp, pinch
    sim = self._env.physics
    n_boxes = int(self.task.split("_")[1])
    #print("Box Pos Z Values:")
    box_names = ['box' + str(b) for b in range(n_boxes)]
    box_pos = sim.body_2d_pose(box_names)[:,:2]
    box_pos_z = box_pos[:,1]
    box_pos_x = box_pos[:,0]

    for box1, id in zip(box_names, range(n_boxes)):
      if sim.site_distance('pinch', box1) < _CLOSE \
        and box_pos_z[id] > 0.0655 and box_pos_z[id]<0.3 and box_pos_x[id]>(-0.682843+0.3) and box_pos_x[id]<(0.682843-0.3):
          # not close to any other box
          distance_other = [sim.site_distance(box1, box2) for box2 in box_names if box2 != box1]
          if np.min(distance_other) > _FAR:
              reward = 1
              return reward

    return 0.0

  def check_no_contact(self, sim, box_name):
      
      number_contacts = sim.data.ncon
      fingertips = self.fingertips
      boxes = self.boxes
      for i in range(number_contacts):
          contact = sim.data.contact[i]
          con_object1 = contact.geom1
          con_object2 = contact.geom2

          # swap if needed
          if con_object2 in fingertips:
              con_object1, con_object2 = con_object2, con_object1

          if (con_object1 in fingertips) and (con_object2 in boxes):
              #print('One finger and one box involved')
              # exactly one finger and one box is part of contact 
              # (we don't want fingers to touch each other)
              box_name_2 = sim.model.id2name(con_object2, 'geom')
              box_pos_z = sim.named.data.geom_xpos[box_name_2, 'z']
              box_pos_x = sim.named.data.geom_xpos[box_name_2, 'x']
              #print("Box2: "+str(box_pos_x)+","+str(box_pos_z))
              #print("Box2: "+box_name_2)
              #print("Box1: "+box_name)
              box_pos_z_1 = sim.named.data.geom_xpos[box_name, 'z']
              box_pos_x_1 = sim.named.data.geom_xpos[box_name, 'x']
              #print("Box1: "+str(box_pos_x_1)+","+str(box_pos_z_1))
              if box_name == box_name_2:
                  return False
              
      return True


  def calculate_box_pos(self, prev_ts_box_pos, prev_ts):
    box_height_threshold = 0.001
    reward = 0
    sim = self._env.physics
    # fingertips = [13,14,17,18]
    # boxes = [19,20,21,21]
    n_boxes = int(self.task.split("_")[1])
    #print("Box Pos Z Values:")
    box_names = ['box' + str(b) for b in range(n_boxes)]
    box_pos = sim.body_2d_pose(box_names)[:,:2]
    box_pos_z = box_pos[:,1]
    box_pos_x = box_pos[:,0]
    prev_box_pos_z = prev_ts_box_pos[:, :2][:,1]
    
    for i in range(n_boxes):
        
        #if box_pos_z[i] > 0.022: # 0.022 box height on ground
        # x pos in between -.382843 and .382843 to not touch wall
        # (values show wall center at x-pos)
        #print("Prev: "+str(prev_box_pos_z))
        #print(box_pos_z)
        
        # Stacking reward version 1: 
        if (box_pos_z[i] > 0.065) and (box_pos_z[i]<0.18) and (box_pos_x[i]>(-0.682843+0.3)) and (box_pos_x[i]<(0.682843-0.3)): # total box height ca. 0.044 -> ca. 0.065 for box stacked on other box
            # box height higher than before
            if (box_pos_z[i] > (prev_box_pos_z[i] + box_height_threshold)):
                reward += 1
            # box no contact with finger
            elif (self.check_no_contact(sim=sim, box_name=box_names[i])):
                # box above other box given custom range --> falling on other box
                for j in range(n_boxes):
                    if (i!=j) and ((abs(box_pos_x[i] - box_pos_x[j])<0.023) and (box_pos_z[i]>box_pos_z[j]+0.04)):
                        #print("Z-Pos: "+str(box_pos_z[i]) + ',' + str(box_pos_z[j]))
                        #print("X-Pos: "+str(box_pos_x[i]) + ',' + str(box_pos_x[j]))
                        if (box_pos_z[i]) < prev_box_pos_z[i]: # avoid reward for same position
                            reward += 1
            # subtract reward when falling and not above other box
            elif (box_pos_z[i] + box_height_threshold) < prev_box_pos_z[i]:
                reward -= 1
                
        '''
        # Stacking reward version 2:
        # add reward if height is according and other box is below and in range (fulfilling stacking condition)
        # discard if out of bounds from walls --> no proper stack
        if (box_pos_z[i] > 0.065) and box_pos_z[i]<0.18:
            for j in range(n_boxes):
                if box_pos_z[i] > box_pos_z[j] + 0.04: #ca. one box height
                    if sim.site_distance('box'+str(j), 'box'+str(i)) < 0.05:
                        if (box_pos_x[i]>(-0.682843+0.3)) and (box_pos_x[i]<(0.682843-0.3)):
                            reward += 1
        # - reward if cond. was met before and isnt in new ts
        elif (prev_box_pos_z[i] > 0.065 and prev_box_pos_z[i]<0.18):
            for j in range(n_boxes):
                if prev_box_pos_z[i] > prev_box_pos_z[j] + 0.04: #ca. one box height
                    if prev_ts.site_distance('box'+str(j), 'box'+str(i)) < 0.05:
                        if (box_pos_x[i]>(-0.682843+0.3)) and (box_pos_x[i]<(0.682843-0.3)):
                            reward -= 1
        '''
        
    return reward, box_pos, box_pos_z


  def target_box_pos_sparse_reward(self):
    box_height_threshold = 0.001
    reward = 0
    sim = self._env.physics
    n_boxes = int(self.task.split("_")[1])
    #print("Box Pos Z Values:")
    box_names = ['box' + str(b) for b in range(n_boxes)]
    box_pos = sim.body_2d_pose(box_names)[:,:2]
    target_pos_x = sim.named.model.body_pos['target', 'x']
    #print("Target:")
    #print(target_pos_x)
    box_pos_z = box_pos[:,1]
    box_pos_x = box_pos[:,0]
    
    for i in range(n_boxes):
        # box no contact with finger
        #if (self.check_no_contact(sim=sim, box_name=box_names[i])):
        # box on target box x pos:
        if ((abs(box_pos_x[i] - target_pos_x)<0.023)):
            reward += 1
            # max 1 reward for frame!!!
            #return reward, box_pos, box_pos_z
        
    return reward, box_pos, box_pos_z

  def target_box_pos_dense_reward(self, only_x=True, field_method=True):
    distance_threshold = 0.35
    distance_field_1 = 0.08
    distance_field_2 = 0.175
    distance_field_3 = 0.35
    reward = 0
    sim = self._env.physics
    n_boxes = int(self.task.split("_")[1])
    #print("Box Pos Z Values:")
    box_names = ['box' + str(b) for b in range(n_boxes)]
    box_pos = sim.body_2d_pose(box_names)[:,:2]
    target_pos_x = sim.named.model.body_pos['target', 'x']
    #print("Target:")
    #print(target_pos_x)     
    box_pos_z = box_pos[:,1]
    box_pos_x = box_pos[:,0]
      
    for i in range(n_boxes):
        # box no contact with finger
        #if (self.check_no_contact(sim=sim, box_name=box_names[i])):
        # reward distance to target box:
        if only_x:
            distance = abs(target_pos_x-box_pos_x[i])
        else:
            distance = sim.site_distance('box'+str(i), 'target')
        if field_method:
            if distance<distance_field_1:
                reward += (0.5 + ((distance_field_1-distance)/(2*distance_field_1)))
            elif distance<distance_field_2:
                reward += (0.0 + (((distance_field_2-distance)*0.92)/distance_field_2))
            elif distance<distance_field_3:
                reward += (-0.5 + ((distance_field_3-distance)/distance_field_3))
            else:
                reward -= (0.15 + distance)
            
        elif distance<distance_threshold:
            reward += ((distance_threshold-distance)/distance_threshold)**2
            
    # Norm reward by boxes:
    reward = reward/n_boxes
    
    return reward, box_pos, box_pos_z

  def step(self, action):
    #print("Current Step: ", self.current_step)
    assert np.isfinite(action['action']).all(), action['action']
    reward = 0.0
    grab_rewards = 0.0
    contacts = 0
    contact_forces = 0
    stacking_rewards = 0.0
    target_pos_rewards = 0.0
    box_pos_z_mean = 0.0
    box_pos_z_total = []
    
    for i in range(self._action_repeat):
      n_boxes = int(self.task.split("_")[1])
      box_names = ['box' + str(b) for b in range(n_boxes)]
      previous_timestep_box_pos = self._env.physics.body_2d_pose(box_names)
      prev_ts = self._env.physics
      time_step = self._env.step(action['action'])
      reward += time_step.reward or 0.0
      
      # calculate contact reward
      ncon = self._env.physics.data.ncon
      # # grab_reward, contact, contact_force = self.calculate_contacts(ncon)
      grab_reward, contact, contact_force = self.learn_to_grab_reward(self.current_step)
      #grab_reward = self.calculate_grab_reward()
      stacking_reward, box_pos, box_pos_z = self.calculate_box_pos(previous_timestep_box_pos, prev_ts)
      # target_pos_reward
      #target_pos_reward,_,_ = self.target_box_pos_sparse_reward()
      target_pos_reward,_,_ = self.target_box_pos_dense_reward(only_x=True, field_method=True)

      grab_rewards += grab_reward
      stacking_rewards += stacking_reward
      target_pos_rewards += target_pos_reward
      contacts += contact
      contact_forces += contact_force
      #box_pos_z_mean += np.mean(box_pos_z)
      #box_pos_z_total.append(box_pos_z)
      
          
      if time_step.last():
        break
    
    assert time_step.discount in (0, 1)
    obs = {
        'reward': reward,
        'grab_reward': grab_rewards,
        'stacking_reward': stacking_rewards,
        'target_pos_reward': target_pos_rewards,
        'is_first': False,
        'is_last': time_step.last(),
        'is_terminal': time_step.discount == 0,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
        'log_contacts': contacts,
        'log_contact_forces': contact_forces,
        'log_box_pos_z_mean': box_pos_z_mean,
        #'last_box_pos_z': box_pos_z_total,
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return obs

  

  def reset(self):
    time_step = self._env.reset()
    obs = {
        'reward': 0.0,
        'grab_reward': 0.0,
        'stacking_reward': 0.0,
        'target_pos_reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
        'log_contacts': 0,
        'log_contact_forces': 0,
        'log_box_pos_z_mean': 0,
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return obs

  def set_current_step(self, step):
      self.current_step = step

  def set_learning_phase(self, learning_phase):
      self.learning_phase = learning_phase


class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky=True, all_actions=False):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    with self.LOCK:
      env = gym.envs.atari.AtariEnv(
          game=name, obs_type='image', frameskip=1,
          repeat_action_probability=0.25 if sticky else 0.0,
          full_action_space=all_actions)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    self._env = gym.wrappers.AtariPreprocessing(
        env, noops, action_repeat, size[0], life_done, grayscale)
    self._size = size
    self._grayscale = grayscale

  @property
  def obs_space(self):
    shape = self._size + (1 if self._grayscale else 3,)
    return {
        'image': gym.spaces.Box(0, 255, shape, np.uint8),
        'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    return {'action': self._env.action_space}

  def step(self, action):
    image, reward, done, info = self._env.step(action['action'])
    if self._grayscale:
      image = image[..., None]
    return {
        'image': image,
        'ram': self._env.env._get_ram(),
        'reward': reward,
        'is_first': False,
        'is_last': done,
        'is_terminal': done,
    }

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    return {
        'image': image,
        'ram': self._env.env._get_ram(),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }

  def close(self):
    return self._env.close()


class Crafter:

  def __init__(self, outdir=None, reward=True, seed=None):
    import crafter
    self._env = crafter.Env(reward=reward, seed=seed)
    self._env = crafter.Recorder(
        self._env, outdir,
        save_stats=True,
        save_video=False,
        save_episode=False,
    )
    self._achievements = crafter.constants.achievements.copy()

  @property
  def obs_space(self):
    spaces = {
        'image': self._env.observation_space,
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'log_reward': gym.spaces.Box(-np.inf, np.inf, (), np.float32),
    }
    spaces.update({
        f'log_achievement_{k}': gym.spaces.Box(0, 2 ** 31 - 1, (), np.int32)
        for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {'action': self._env.action_space}

  def step(self, action):
    image, reward, done, info = self._env.step(action['action'])
    obs = {
        'image': image,
        'reward': reward,
        'is_first': False,
        'is_last': done,
        'is_terminal': info['discount'] == 0,
        'log_reward': info['reward'],
    }
    obs.update({
        f'log_achievement_{k}': v
        for k, v in info['achievements'].items()})
    return obs

  def reset(self):
    obs = {
        'image': self._env.reset(),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'log_reward': 0.0,
    }
    obs.update({
        f'log_achievement_{k}': 0
        for k in self._achievements})
    return obs


class Dummy:

  def __init__(self):
    pass

  @property
  def obs_space(self):
    return {
        'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    return {'action': gym.spaces.Box(-1, 1, (6,), dtype=np.float32)}

  def step(self, action):
    return {
        'image': np.zeros((64, 64, 3)),
        'reward': 0.0,
        'is_first': False,
        'is_last': False,
        'is_terminal': False,
    }

  def reset(self):
    return {
        'image': np.zeros((64, 64, 3)),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs = self._env.step(action)
    self._step += 1
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
      self._step = None
    return obs

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.act_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})


class OneHotAction:

  def __init__(self, env, key='action'):
    assert hasattr(env.act_space[key], 'n')
    self._env = env
    self._key = key
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    shape = (self._env.act_space[self._key].n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.n = shape[0]
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    index = np.argmax(action[self._key]).astype(int)
    reference = np.zeros_like(action[self._key])
    reference[index] = 1
    if not np.allclose(reference, action[self._key]):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step({**action, self._key: index})

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.act_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class ResizeImage:

  def __init__(self, env, size=(64, 64)):
    self._env = env
    self._size = size
    self._keys = [
        k for k, v in env.obs_space.items()
        if len(v.shape) > 1 and v.shape[:2] != size]
    print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
    if self._keys:
      from PIL import Image
      self._Image = Image

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    spaces = self._env.obs_space
    for key in self._keys:
      shape = self._size + spaces[key].shape[2:]
      spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
    return spaces

  def step(self, action):
    obs = self._env.step(action)
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def reset(self):
    obs = self._env.reset()
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def _resize(self, image):
    image = self._Image.fromarray(image)
    image = image.resize(self._size, self._Image.NEAREST)
    image = np.array(image)
    return image


class RenderImage:

  def __init__(self, env, key='image'):
    self._env = env
    self._key = key
    self._shape = self._env.render().shape

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    spaces = self._env.obs_space
    spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
    return spaces

  def step(self, action):
    obs = self._env.step(action)
    obs[self._key] = self._env.render('rgb_array')
    return obs

  def reset(self):
    obs = self._env.reset()
    obs[self._key] = self._env.render('rgb_array')
    return obs


class Async:

  # Message types for communication via the pipe.
  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _CLOSE = 4
  _EXCEPTION = 5

  def __init__(self, constructor, strategy='thread'):
    self._pickled_ctor = cloudpickle.dumps(constructor)
    if strategy == 'process':
      import multiprocessing as mp
      context = mp.get_context('spawn')
    elif strategy == 'thread':
      import multiprocessing.dummy as context
    else:
      raise NotImplementedError(strategy)
    self._strategy = strategy
    self._conn, conn = context.Pipe()
    self._process = context.Process(target=self._worker, args=(conn,))
    atexit.register(self.close)
    self._process.start()
    self._receive()  # Ready.
    self._obs_space = None
    self._act_space = None

  def access(self, name):
    self._conn.send((self._ACCESS, name))
    return self._receive

  def call(self, name, *args, **kwargs):
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      pass  # The connection was already closed.
    self._process.join(5)

  @property
  def obs_space(self):
    if not self._obs_space:
      self._obs_space = self.access('obs_space')()
    return self._obs_space

  @property
  def act_space(self):
    if not self._act_space:
      self._act_space = self.access('act_space')()
    return self._act_space

  def step(self, action, blocking=False):
    promise = self.call('step', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking=False):
    promise = self.call('reset')
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except (OSError, EOFError):
      raise RuntimeError('Lost connection to environment worker.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, conn):
    try:
      ctor = cloudpickle.loads(self._pickled_ctor)
      env = ctor()
      conn.send((self._RESULT, None))  # Ready.
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print('Error in environment process: {}'.format(stacktrace))
      conn.send((self._EXCEPTION, stacktrace))
    finally:
      try:
        conn.close()
      except IOError:
        pass  # The connection was already closed.
