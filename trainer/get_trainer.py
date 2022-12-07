from . import SMF_trainer



def get_trainer(name):
    if name == 'SMF-UL':
        TrainFramework = SMF_trainer.TrainFramework
    else:
        raise NotImplementedError(name)

    return TrainFramework
