import tensorflow as tf
from tensorflow_probability import distributions as tfd

import agent
import common


class Random(common.Module):

  def __init__(self, config, act_space, wm, tfstep, reward):
    self.config = config
    self.act_space = self.act_space

  def actor(self, feat):
    shape = feat.shape[:-1] + self.act_space.shape
    if self.config.actor.dist == 'onehot':
      return common.OneHotDist(tf.zeros(shape))
    else:
      dist = tfd.Uniform(-tf.ones(shape), tf.ones(shape))
      return tfd.Independent(dist, 1)

  def train(self, start, context, data):
    return None, {}


class Plan2Explore(common.Module):

  def __init__(self, config, act_space, wm, tfstep, reward):
    # self.obs = {'contact_reward': None}
    self.config = config
    self.reward = reward
    self.wm = wm
    self.ac = agent.ActorCritic(config, act_space, tfstep)
    self.actor = self.ac.actor
    stoch_size = config.rssm.stoch
    if config.rssm.discrete:
      stoch_size *= config.rssm.discrete
    size = {
        'embed': 32 * config.encoder.cnn_depth,
        'stoch': stoch_size,
        'deter': config.rssm.deter,
        'feat': config.rssm.stoch + config.rssm.deter,
    }[self.config.disag_target]
    self._networks = [
        common.MLP(size, **config.expl_head)
        for _ in range(config.disag_models)]
    self.opt = common.Optimizer('expl', **config.expl_opt)
    self.extr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
    self.intr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
    self.guide_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)

    self.contact_reward = lambda seq: self.wm.heads['contact_reward'](seq['feat']).mode()

  def set_mode(self, mode):
    self._mode = mode
    self.ac.set_mode(mode)

  def train(self, start, context, data):
    #print('train in expl.py')
    # in data is contact_reward
    metrics = {}
    stoch = start['stoch']
    if self.config.rssm.discrete:
      stoch = tf.reshape(
          stoch, stoch.shape[:-2] + (stoch.shape[-2] * stoch.shape[-1]))
    target = {
        'embed': context['embed'],
        'stoch': stoch,
        'deter': start['deter'],
        'feat': context['feat'],
    }[self.config.disag_target]
    inputs = context['feat']
    if self.config.disag_action_cond:
      action = tf.cast(data['action'], inputs.dtype)
      inputs = tf.concat([inputs, action], -1)
    metrics.update(self._train_ensemble(inputs, target))
    metrics.update(self.ac.train(
        self.wm, start, data['is_terminal'], self._intr_reward, self.contact_reward))
    return None, metrics

  def _intr_reward(self, seq):
    # batch: 10, length: 10 = 100 daher kommt die 100
    inputs = seq['feat']
    #print('inputs', inputs)
    # shape=(16, 100, 2048), dtype=float16)
    if self.config.disag_action_cond:
      action = tf.cast(seq['action'], inputs.dtype)
      #print('action', action)
      # shape=(16, 100, 5), dtype=float16)
      inputs = tf.concat([inputs, action], -1)
      #print('inputs after concate', inputs)
      # shape=(16, 100, 2053), dtype=float16)
    
    preds = [head(inputs).mode() for head in self._networks]
    #print('preds', preds[0].shape)
    # shape=(16, 100, 1024), dtype=float32 
    # len(preds) 10 = config.disag_models
    #print('tf.tensor(preds)', tf.tensor(preds).shape, 
    #'+ std(0)', tf.tensor(preds).std(0).shape,
    #'+ mean(-1)', tf.tensor(preds).std(0).mean(-1).shape)
    # tf.tensor(preds) (10, 16, 100, 1024) + std(0) (16, 100, 1024) + mean(-1) (16, 100)
    disag = tf.tensor(preds).std(0).mean(-1)
    if self.config.disag_log:
      disag = tf.math.log(disag)
    #print('disag', disag, 'disag.shape', disag.shape)
    #print('self.intr_rewnorm(disag)[0]', self.intr_rewnorm(disag)[0])
    # shape=(16, 100), dtype=float32) disag.shape (16, 100)
    reward = self.config.expl_intr_scale * self.intr_rewnorm(disag)[0]
    if self.config.expl_guide_scale and False:
      # passt noch nicht
      # print('We are in in guide scale')
      # self.obs[contact_reward] tf.Tensor([0.], shape=(1,), dtype=float32)
      #print('contact_reward', contact_reward)
      # print('norm', self.guide_rewnorm(contact_reward)[0])
      # reward += self.config.expl_guide_scale * contact_reward
      pass
    if self.config.expl_extr_scale:
      reward += self.config.expl_extr_scale * self.extr_rewnorm(
          self.reward(seq))[0]
    return reward

  def _train_ensemble(self, inputs, targets):
    if self.config.disag_offset:
      targets = targets[:, self.config.disag_offset:]
      inputs = inputs[:, :-self.config.disag_offset]
    targets = tf.stop_gradient(targets)
    inputs = tf.stop_gradient(inputs)
    with tf.GradientTape() as tape:
      preds = [head(inputs) for head in self._networks]
      loss = -sum([pred.log_prob(targets).mean() for pred in preds])
    metrics = self.opt(tape, loss, self._networks)
    return metrics


class ModelLoss(common.Module):

  def __init__(self, config, act_space, wm, tfstep, reward):
    self.config = config
    self.reward = reward
    self.wm = wm
    self.ac = agent.ActorCritic(config, act_space, tfstep)
    self.actor = self.ac.actor
    self.head = common.MLP([], **self.config.expl_head)
    # self.head shape = [], {layers: 4, units: 400, act: elu, norm: none, dist: mse}
    self.opt = common.Optimizer('expl', **self.config.expl_opt)

  def train(self, start, context, data):
    metrics = {}
    target = tf.cast(context[self.config.expl_model_loss], tf.float32)
    with tf.GradientTape() as tape:
      loss = -self.head(context['feat']).log_prob(target).mean()
    metrics.update(self.opt(tape, loss, self.head))
    metrics.update(self.ac.train(
        self.wm, start, data['is_terminal'], self._intr_reward))
    return None, metrics

  def _intr_reward(self, seq):
    reward = self.config.expl_intr_scale * self.head(seq['feat']).mode()
    if self.config.expl_extr_scale:
      reward += self.config.expl_extr_scale * self.reward(seq)
    return reward
