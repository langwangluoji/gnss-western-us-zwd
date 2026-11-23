import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from unittest.mock import patch

from residual_edm_v1.training import Cursor, train_updates, load_checkpoint, study_remaining


class TinyData:
    train_indices = np.arange(37)
    manifest = {'test': 'synthetic'}
    def batch(self, ids, targets=True):
        x = torch.tensor(ids, dtype=torch.float32)[:, None] / 37
        return {'x': x, 'residual': 2 * x + .2}


class TinyModel(torch.nn.Module):
    arm = 'det'
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
    def loss(self, batch, **kw):
        return (self.layer(batch['x']) - batch['residual']).square().mean()


class TrainingTests(unittest.TestCase):
    def test_sqlite_interruption_resumes_trial_and_does_not_duplicate(self):
        import optuna
        from residual_edm_v1.training import search
        data=TinyData();data.device='cpu';data.screen_indices=np.arange(4)
        interrupted=[False]
        opened=[]
        create=optuna.create_study
        def capture(**kwargs):
            study=create(**kwargs);opened.append(study);return study
        def execute(model,data,directory,config,seed,updates,**kwargs):
            trial=kwargs['trial']
            if not interrupted[0]:
                interrupted[0]=True
                raise KeyboardInterrupt()
            if trial.number==1: raise optuna.TrialPruned()
            if trial.number==2: raise ValueError('synthetic failure')
            return {'best':float(trial.number),'checkpoint':str(directory/'best.pt')}
        with tempfile.TemporaryDirectory() as temp, patch('optuna.create_study',side_effect=capture), patch('residual_edm_v1.model.ResidualModel',side_effect=lambda arm:TinyModel()), patch('residual_edm_v1.training.train_updates',side_effect=execute):
            with self.assertRaises(KeyboardInterrupt): search(data,Path(temp),'det')
            study=search(data,Path(temp),'det')
            self.assertEqual(len(study.trials),12)
            self.assertEqual(study.trials[0].state,optuna.trial.TrialState.COMPLETE)
            self.assertEqual(study.trials[1].state,optuna.trial.TrialState.PRUNED)
            self.assertEqual(study.trials[2].state,optuna.trial.TrialState.FAIL)
            self.assertEqual(len(search(data,Path(temp),'det').trials),12)
            # Explicitly release SQLAlchemy pools before Windows removes the
            # temporary database, including the interrupted study's connection.
            for item in opened:
                storage=item._storage._backend
                storage.remove_session()
                storage.engine.dispose()

    def test_cursor_resume_exact(self):
        a = Cursor(np.arange(37), 42)
        a.take(80)
        state = a.state_dict()
        b = Cursor(np.arange(37), 0)
        b.load_state_dict(state)
        np.testing.assert_array_equal(a.take(120), b.take(120))

    def test_checkpoint_resume_matches_continuous(self):
        torch.set_num_threads(1)
        config = {'lr': 1e-3, 'weight_decay': .01, 'ema': .99}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            torch.manual_seed(42)
            a = TinyModel()
            train_updates(a, TinyData(), root/'a', config, 42, 200, eval_every=1000)
            torch.manual_seed(42)
            b = TinyModel()
            train_updates(b, TinyData(), root/'b', config, 42, 100, eval_every=1000)
            train_updates(TinyModel(), TinyData(), root/'b', config, 42, 200, eval_every=1000)
            sa = load_checkpoint(root/'a'/'last.pt')
            sb = load_checkpoint(root/'b'/'last.pt')
            for group in ('model', 'ema'):
                for key in sa[group]:
                    torch.testing.assert_close(sa[group][key], sb[group][key], rtol=0, atol=0)
            self.assertEqual(sb['step'], 200)

    def test_trial_budget_counts_failures_and_pruning(self):
        import optuna
        study = optuna.create_study()
        study.add_trial(optuna.trial.create_trial(value=1.))
        study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
        study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED))
        self.assertEqual(study_remaining(study, 12), 9)


if __name__ == '__main__':
    unittest.main()
