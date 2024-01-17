from pipeline.scheduler.base import SchedulerBase


class Scheduler_AE(SchedulerBase):
    def __init__(self,s_enc,s_dec,s_gen,s_disc) -> None:
        self.Scheduler_enc = s_enc
        self.Scheduler_dec = s_dec
        self.Scheduler_gen = s_gen
        self.Scheduler_disc = s_disc

    def step(self):
        self.Scheduler_enc.step()
        self.Scheduler_dec.step()
        self.Scheduler_gen.step()
        self.Scheduler_disc.step()
    
    def load(self,param):
        try:
            self.Scheduler_enc.load_state_dict(param['enc'])
            self.Scheduler_dec.load_state_dict(param['dec'])
            self.Scheduler_gen.load_state_dict(param['gen'])
            self.Scheduler_disc.load_state_dict(param['disc'])
        except:
            print("Unable to load scheduler parameters ...")

    def get_param(self):
        return{
            'enc':self.Scheduler_enc.state_dict(),
            'dec':self.Scheduler_dec.state_dict(),
            'gen':self.Scheduler_gen.state_dict(),
            'disc':self.Scheduler_disc.state_dict()}

