"""Optional CUDA Graph acceleration with one complete SAC update per eligible transition."""
import torch


class CudaPolicy:
    def __init__(self, agent):
        self.agent=agent
        self.state=torch.zeros((1,agent.state_dim),device=agent.device)
        rng=torch.cuda.get_rng_state(agent.device)
        stream=torch.cuda.Stream(device=agent.device)
        stream.wait_stream(torch.cuda.current_stream(agent.device))
        with torch.no_grad(),torch.cuda.stream(stream):
            for _ in range(3):agent._scale_to_env(agent.actor(self.state).rsample())
        torch.cuda.current_stream(agent.device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(self.graph):
            self.action=agent._scale_to_env(agent.actor(self.state).rsample())
        # Capture policy inference to reduce small-kernel dispatch for individual states.
        # Warm-up leaves weights unchanged; restoring RNG avoids consuming extra training randomness.
        torch.cuda.set_rng_state(rng,agent.device)

    @torch.no_grad()
    def sample_action(self,state):
        self.state.copy_(torch.as_tensor(state,dtype=torch.float32,device=self.agent.device).reshape(1,-1))
        self.graph.replay()
        return self.action.squeeze(0).cpu().numpy()


class CudaUpdate:
    def __init__(self, agent, batch):
        if agent.device.type != 'cuda' or not agent.optimized:
            raise ValueError('CUDA Graph requires optimized CUDA agent')
        self.agent = agent
        self.keys = ['states','actions','rewards','next_states','dones']
        widths = [batch[key].shape[1] for key in self.keys]
        self.storage = torch.empty((batch['states'].shape[0],sum(widths)),device=agent.device)
        self.batch = dict(zip(self.keys,self.storage.split(widths,dim=1)))
        self._copy_batch(batch)
        networks = [agent.actor,agent.critic1,agent.critic2,agent.target_critic1,agent.target_critic2]
        initial = [[p.detach().clone() for p in net.parameters()] for net in networks]
        initial_temperature = agent.log_temperature.detach().clone()
        rng_state = torch.cuda.get_rng_state(agent.device)
        optimizers = [agent.actor_optimizer,agent.critic1_optimizer,
                      agent.critic2_optimizer,agent.temperature_optimizer]

        def restore_initial():
            # Graph warm-up/capture executes updates; restore weights, temperature and all Adam state in place.
            # Otherwise extra training occurs. Do not replace tensor addresses already referenced by the graph.
            with torch.no_grad():
                for net,snapshot in zip(networks,initial):
                    for p,value in zip(net.parameters(),snapshot):p.copy_(value)
                agent.log_temperature.copy_(initial_temperature)
                for optimizer in optimizers:
                    for state in optimizer.state.values():
                        for value in state.values():
                            if torch.is_tensor(value):value.zero_()

        # Preallocate optimizer state and temporary memory; include initialization in benchmark timing.
        stream = torch.cuda.Stream(device=agent.device)
        stream.wait_stream(torch.cuda.current_stream(agent.device))
        with torch.cuda.stream(stream):
            for _ in range(3):agent.update(self.batch,return_tensors=True)
        torch.cuda.current_stream(agent.device).wait_stream(stream)
        restore_initial()
        torch.cuda.synchronize(agent.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.metrics = agent.update(self.batch,return_tensors=True)
        restore_initial()
        torch.cuda.set_rng_state(rng_state,agent.device)

    def _copy_batch(self, batch):
        # Write a fixed input buffer with one cat, avoiding five small copies and graph reconstruction.
        torch.cat([batch[key] for key in self.keys],dim=1,out=self.storage)

    def update(self, batch):
        self._copy_batch(batch)
        self.graph.replay()
        return self.metrics
