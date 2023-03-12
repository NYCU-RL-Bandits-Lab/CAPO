from torch import optim, nn

from ..policy import Parametric

class PGAlgorithm(object):
    ''' Generic base class for Policy Gradient algorithms. '''

    def __init__(self, agent, network, reccurrent=False, optimizer='adam', optim_kwargs={ }, cyclic=False):
        super().__init__()
        assert isinstance(network, Parametric), "network must be an instance of 'policy.Parametric'"
        assert optimizer in ['adam', 'rmsprop'], "optimizer must be a either 'adam' or 'rmsprop'"

        self.agent = agent # Track the agent
        self.network = network
        self.reccurrent = reccurrent

        # optimizer instance
        self.OptimClass = optim.Adam if optimizer == 'adam' else optim.RMSprop
        # self.optimizer = self.OptimClass(self.network.parameters(), **optim_kwargs)
        lr = optim_kwargs['lr']
        self.optimizer = self.OptimClass([
            {'params' : self.network.affine.parameters(), 'lr': lr}, 
            {'params' : self.network.thetas.parameters(), 'lr': lr}, 
            {'params' : self.network.tabular_thetas, 'lr': lr}, 
            {'params' : self.network.value.parameters(), 'lr': lr*10}
                            ])

    def zero_grad(self):
        ''' Similar to PyTorch's boilerplate 'optim.zero_grad()' '''
        self.optimizer.zero_grad()

    def step(self, clip=None):
        ''' Similar to PyTorch's boilerplate 'optim.step()' '''
        
        # Optional gradient clipping
        if clip is not None:
            nn.utils.clip_grad_norm_(self.network.parameters(), clip)
        
        self.optimizer.step()
