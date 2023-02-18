import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
import time
from collections import deque
import pickle

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common



def main():
  configs = yaml.safe_load((
      pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = common.Flags(config).parse(remaining)

  logdir = pathlib.Path(config.logdir).expanduser()
  # easier download from gcloud
  logdir_downloads = pathlib.Path(config.logdir + "/downloads").expanduser()
  logdir_downloads.mkdir(parents=True, exist_ok=True)
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir_downloads / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  # deque over last 10 episodes(len(episod)=1000)
  MAX_SIZE = 10
  global queue, learning_phase
  change_config = common.Once()
  if (logdir / 'learn_lift.pkl').exists():
      with open(pathlib.Path(logdir / 'learn_lift.pkl'), 'rb') as f:
          pickle_output = pickle.load(f)
      if len(pickle_output) == 2:
          queue, learning_phase = pickle_output
          # legacy support for old pickles
          if type(learning_phase) == dict:
              for key, value in learning_phase.items():
                  if value():
                      learning_phase = str(key)
                      break
                  else:
                      learning_phase = 'init'
      elif len(pickle_output) == 3:
          queue, should_grab_now, should_lift_now = pickle_output
          if should_lift_now():
              if should_grab_now():
                  learning_phase = 'grab'
              elif should_lift_now():
                  learning_phase = 'lift'
              else:
                  learning_phase = 'init'
      else:
          raise ValueError('Invalid pickle output')
      
      print('Loaded metadata from learn_lift.pkl')
      print('queue', queue)
      print('learning_phase', learning_phase)
  else:
      queue = deque(maxlen=MAX_SIZE)
      learning_phase = 'init'

  import tensorflow as tf
  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
      capacity=config.replay.capacity // 10,
      minlen=config.dataset.length,
      maxlen=config.dataset.length))
  step = common.Counter(train_replay.stats['total_steps'])
  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir_downloads),
      common.TensorBoardOutput(logdir_downloads),
  ]
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video_train = common.Every(config.eval_every)
  should_video_eval = common.Every(config.eval_every)
  # I think this was a bug in the original code.
  should_expl = common.Until(config.expl_until)
  print('should_expl', should_expl._until)

  def make_env(mode):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
      env = common.DMC(
          task, config.action_repeat, config.render_size, config.dmc_camera)
      env = common.NormalizeAction(env)
    elif suite == 'atari':
      env = common.Atari(
          task, config.action_repeat, config.render_size,
          config.atari_grayscale)
      env = common.OneHotAction(env)
    elif suite == 'crafter':
      assert config.action_repeat == 1
      outdir = logdir / 'crafter' if mode == 'train' else None
      reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
      env = common.Crafter(outdir, reward)
      env = common.OneHotAction(env)
    else:
      raise NotImplementedError(suite)
    env = common.TimeLimit(env, config.time_limit)
    return env

  def per_episode(ep, mode):
    global queue, learning_phase
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    grab_reward = float(ep['grab_reward'].astype(np.float64).sum())

    if mode == 'train' and config.meta_learn:
      print('\n')
      queue.append(grab_reward)
      print('Last grab rewards', list(queue))
      print('learning_phase', learning_phase)
      if len(queue) == MAX_SIZE:
        if learning_phase == 'init' and np.average(queue) > 300:
          learning_phase = 'close'
          queue = deque(maxlen=MAX_SIZE)
          print('Activating grab now')
        elif learning_phase == 'close' and np.average(queue) > 300:
          learning_phase = 'grab'
          queue = deque(maxlen=MAX_SIZE)
          print('Activating grab now')
        elif learning_phase == 'grab' and np.average(queue) > 300:
          learning_phase = 'lift'
          queue = deque(maxlen=MAX_SIZE)
          print('Activating lift now')
        elif config.meta_learn_hover and learning_phase == 'lift' and np.min(queue) > 100:
          learning_phase = 'hover'
          queue = deque(maxlen=MAX_SIZE)
          print('Activating hover now')
          # save pretrained model
          logdir = pathlib.Path(config.logdir).expanduser()
          # easier download from gcloud
          logdir_pretrained = pathlib.Path(config.logdir + "/pretrained_grab_100thres").expanduser()
          logdir_pretrained.mkdir(parents=True, exist_ok=True)
          agnt.save(logdir_pretrained / 'variables.pkl')
          with open(pathlib.Path(logdir_pretrained / 'learn_lift.pkl'), 'wb') as f:
              pickle.dump((queue, learning_phase), f)
              print('Saved learn_lift.pkl')
        elif config.meta_learn_drop and learning_phase == 'hover' and np.average(queue) > 300:
          learning_phase = 'drop'
          queue = deque(maxlen=MAX_SIZE)
          print('Activating drop now')

    stacking_reward = float(ep['stacking_reward'].astype(np.float64).sum())
    # contacts
    contacts = ep['log_contacts'].astype(np.uint32).sum()
    contact_force_sum = float(ep['log_contact_forces'].astype(np.float64).sum())
    contact_force_mean = float(ep['log_contact_forces'].astype(np.float64).mean())
    # Box Pos:
    box_pos_z_mean = float(ep['log_box_pos_z_mean'].astype(np.float64).mean())

    print(f'{mode.title()} episode has {length} steps and return {score:.1f}, grab reward {grab_reward:.1f} and stacking reward {stacking_reward:.1f}.')
    print(f'Episode has {contacts} contacts and contact force sum {contact_force_sum:.1f} and mean {contact_force_mean:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_grab_reward', grab_reward)
    logger.scalar(f'{mode}_stacking_reward', stacking_reward)
    logger.scalar(f'{mode}_length', length)
    # contacts
    logger.scalar(f'{mode}_contacts', contacts)
    logger.scalar(f'{mode}_contact_force_sum', contact_force_sum)
    logger.scalar(f'{mode}_contact_force_mean', contact_force_mean)
    # box pos z
    logger.scalar(f'{mode}_box_pos_z_mean', box_pos_z_mean)
    
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      for key in config.log_keys_video:
        logger.video(f'{mode}_policy_{key}', ep[key])
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()

  print('Create envs.')
  num_eval_envs = min(config.envs, config.eval_eps)
  if config.envs_parallel == 'none':
    train_envs = [make_env('train') for _ in range(config.envs)]
    eval_envs = [make_env('eval') for _ in range(num_eval_envs)]
  else:
    make_async_env = lambda mode: common.Async(
        functools.partial(make_env, mode), config.envs_parallel)
    train_envs = [make_async_env('train') for _ in range(config.envs)]
    eval_envs = [make_async_env('eval') for _ in range(eval_envs)]
  act_space = train_envs[0].act_space
  obs_space = train_envs[0].obs_space
  train_driver = common.Driver(train_envs)
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)

  prefill = max(0, config.prefill - train_replay.stats['total_steps'])
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(act_space)
    train_driver(random_agent, steps=prefill, episodes=1)
    eval_driver(random_agent, episodes=1)
    train_driver.reset()
    eval_driver.reset()

  print('Create agent.')
  train_dataset = iter(train_replay.dataset(**config.dataset))
  report_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  agnt = agent.Agent(config, obs_space, act_space, step)

  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(train_dataset))
  if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      train_agent(next(train_dataset))
  # test if 1k or 500
  if config.multi_agent:
      train_policy = lambda *args: agnt.policy(
          *args, mode='explore' if should_expl(step) or ((step.value / 500) % 2 == 0) else 'train')  
  else:
      train_policy = lambda *args: agnt.policy(
          *args, mode='explore' if should_expl(step) else 'train')
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  def train_step(tran, worker):
    if should_train(step):
      for _ in range(config.train_steps):
        mets = train_agent(next(train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agnt.report(next(report_dataset)), prefix='train')
      logger.write(fps=True)
  train_driver.on_step(train_step)

  while step < config.steps:
    #print(f'Step {step.value}')
    #print('should_expl', should_expl(step), should_expl._until)

    [env.set_current_step(step.value) for env in train_envs]
    [env.set_current_step(step.value) for env in eval_envs]
    [env.set_learning_phase(learning_phase) for env in train_envs]
    [env.set_learning_phase(learning_phase) for env in eval_envs]

    if learning_phase == 'hover' and change_config():
        config = config.update({
          'grab_reward_weight': 0.2,
          'stacking_reward_weight': 0.8,
        })
        print('grab_reward_weight:', config.grab_reward_weight, 'stacking_reward_weight', config.stacking_reward_weight)


    if step >= config.start_external_reward and False:
      # linear fade-in from grab to stacking reward
      config = config.update({
          'reward_weight': 1.0,
          'grab_reward_weight': (1.0 - (step.value - config.start_external_reward)\
                                / (config.steps - config.start_external_reward)),
          'stacking_reward_weight': (0.0 + (step.value - config.start_external_reward)\
                                / (config.steps - config.start_external_reward))
      })

    logger.write()
    print('Start evaluation.')
    logger.add(agnt.report(next(eval_dataset)), prefix='eval')
    eval_driver(eval_policy, episodes=config.eval_eps)
    print('Start training.')
    #for i in range(10):
    #  print('Train driver', i)
    #  print('queue', queue, 'learning_phase', learning_phase)
    #  train_driver(train_policy, steps=config.eval_every)
    train_driver(train_policy, steps=config.eval_every)
    agnt.save(logdir / 'variables.pkl')
    with open(pathlib.Path(logdir / 'learn_lift.pkl'), 'wb') as f:
        pickle.dump((queue, learning_phase), f)
        print('Saved learn_lift.pkl')

  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  main()
