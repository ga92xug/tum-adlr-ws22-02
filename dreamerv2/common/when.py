class Every:

  def __init__(self, every):
    self._every = every
    self._last = None

  def __call__(self, step):
    step = int(step)
    if not self._every:
      return False
    if self._last is None:
      self._last = step
      return True
    if step >= self._last + self._every:
      self._last += self._every
      return True
    return False


class Once:

  def __init__(self):
    self._once = True

  def __call__(self):
    if self._once:
      self._once = False
      return True
    return False


class Activated:
    def __init__(self):
        self.activated = False

    def activate(self):
        self.activated = True
    
    def deactivate(self):
        self.activated = False

    def __call__(self):
        return self.activated

    def __str__(self) -> str:
        return f'Activated: {self.activated}'




class Until:

  def __init__(self, until):
    self._until = until

  def __call__(self, step):
    step = int(step)
    if not self._until:
      return True
    return step < self._until
