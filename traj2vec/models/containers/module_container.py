class ModuleContainer:
    def __init__(self):
        self.modules = []

    def get_params(self):
        params = []
        for m in self.modules:
            params.extend(m.parameters())
        return params


    def get_params_flat(self):
        return torch.cat([p.view(-1) for p in self.get_params()])

    def set_params_flat(self, params_flat):
        count = 0
        for p in self.get_params():
            size = p.numel()
            p.data[:] = params_flat[count:count+size].contiguous()
            count += size

    def get_modules(self):
        return self.modules

    def apply(self, init):
        for m in self.get_modules():
            m.apply(init)

    def cuda(self):
        [m.cuda() for m in self.get_modules()]


    def state_dict_lst(self):
        return [m.state_dict() for m in self.get_modules()]

    def load_state_dict(self, sd):
        for m, x in zip(self.modules, sd):
            m.load_state_dict(x)

    def zero_grad(self):
        for p in self.get_params():
            if p.grad is not None:
                p.grad.data.zero_()

    def recurrent(self):
        return False