"""Explicit process-local adapter: GraphAffine model, original strict experiment IO."""
from pathlib import Path
from . import native_process_experiment as runner
from .native_graph_gaussian import GraphAffineVelocity
from .data import sha


def main():
    original_codes=runner.codes;original_train=runner.train

    def codes():
        return {**original_codes(),**{name:sha(Path(__file__).with_name(name)) for name in
            ('native_graph_gaussian.py','native_graph_gaussian_experiment.py')}}

    def factory(mode):
        if mode!='full':raise ValueError('GraphAffine control requires the same full whitening as Full')
        return GraphAffineVelocity()

    def train(args):
        args.family='graph_affine';original_train(args)

    # All three bindings are local to this fresh Python process. Historical code
    # files, active workers, data arrays and scientific locks are unchanged.
    runner.codes=codes;runner.make_process_model=factory;runner.train=train
    runner.main()


if __name__=='__main__':main()
