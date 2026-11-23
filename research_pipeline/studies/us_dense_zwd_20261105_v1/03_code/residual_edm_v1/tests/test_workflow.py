import unittest
from residual_edm_v1.workflow import gate, mc_stable, validate_promotion, validate_evaluation


class WorkflowTests(unittest.TestCase):
    def test_registered_budget_cannot_be_bypassed(self):
        import optuna
        study=optuna.create_study()
        study.add_trial(optuna.trial.create_trial(value=1.))
        with self.assertRaises(ValueError): validate_promotion(study,'stage1')

    def test_retrospective_requires_terminal_replication(self):
        selection={'gate':{'passed':True}}
        with self.assertRaises(ValueError): validate_evaluation(selection,{})
        with self.assertRaises(ValueError): validate_evaluation(selection,{'status':'not_promoted'})
        validate_evaluation({'gate':{'passed':False}}, {'status':'not_promoted'})
    def test_gate_requires_both_improvements(self):
        self.assertTrue(gate(8., [7.97], [7.94])['passed'])
        self.assertFalse(gate(8., [7.94], [7.94])['passed'])
        self.assertFalse(gate(8., [8.], [7.98])['passed'])

    def test_gate_rejects_bad_seed(self):
        self.assertFalse(gate(8., [8.,8.,8.], [7.7,7.7,8.06])['passed'])

    def test_mc_requires_stable_difference_and_decision(self):
        self.assertTrue(mc_stable(8., 7.97, 7.94, 7.941))
        self.assertFalse(mc_stable(8., 7.97, 7.949, 7.951))
        self.assertFalse(mc_stable(8., 7.97, 7.94, 7.946))


if __name__=='__main__': unittest.main()
