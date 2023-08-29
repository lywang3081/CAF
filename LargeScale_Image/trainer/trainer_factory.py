import copy
from arguments import get_args
args = get_args()
import torch

class TrainerFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(myModel, args, optimizer, evaluator, taskcla, model_emp = None):
        if args.trainer == 'ewc':
            import trainer.ewc as trainer
        elif args.trainer == 'ewc_caf':
            import trainer.ewc_caf as trainer
        elif args.trainer == 'mas':
            import trainer.mas as trainer
        elif args.trainer == 'mas_caf':
            import trainer.mas_caf as trainer
        elif args.trainer == 'rwalk':
            import trainer.rwalk as trainer
        elif args.trainer == 'si':
            import trainer.si as trainer
        elif args.trainer == 'gs':
            import trainer.gs as trainer
        return trainer.Trainer(myModel, args, optimizer, evaluator, taskcla)
    
class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self, model, args, optimizer, evaluator, taskcla, model_emp = None):
        
        self.model = model
        self.model_emp = model_emp
        self.args = args
        self.optimizer = optimizer
        self.evaluator=evaluator
        self.taskcla=taskcla
        self.model_fixed = copy.deepcopy(self.model)
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.current_lr = args.lr
        self.ce=torch.nn.CrossEntropyLoss()
        self.model_single = copy.deepcopy(self.model)
        self.optimizer_single = None
        self.model.s_gate = args.s_gate