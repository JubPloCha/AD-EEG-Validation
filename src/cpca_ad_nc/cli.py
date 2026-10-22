from __future__ import annotations
import argparse
from threadpoolctl import threadpool_limits
from .io import config,prepare,prepare_external,extract_features


def main():
    p=argparse.ArgumentParser(description='AD-NC EEG: parallel EEG/PCA/cPCA branches and frozen external validation')
    p.add_argument('phase',choices=['features','prepare','internal','freeze','predict','external','interpret','cognition','stability','report','all'])
    p.add_argument('--config',required=True,help='TOML configuration; relative paths resolve beside this file')
    p.add_argument('--cohort',choices=['development','external'],default='development',help='EEG cohort for features phase')
    args=p.parse_args();c=config(args.config)
    stages=['prepare','internal','freeze','external','interpret','report'] if args.phase=='all' else [args.phase]
    with threadpool_limits(limits=1):
        for stage in stages:
            if stage=='features':extract_features(c,args.cohort)
            elif stage=='prepare':prepare(c)
            elif stage=='internal':
                from .workflow import internal
                internal(c)
            elif stage=='freeze':
                from .workflow import freeze
                freeze(c)
            elif stage=='predict':
                from .external import predict_external
                predict_external(c)
            elif stage=='external':
                from .external import evaluate_external
                prepare_external(c,require_labels=True);evaluate_external(c)
            elif stage=='interpret':
                from .interpret import run_interpretation
                run_interpretation(c)
            elif stage=='cognition':
                from .cognition import cognition
                from .interpret import feature_dictionary
                from .io import schema
                out=c['_root']/'private/interpretability';out.mkdir(parents=True,exist_ok=True)
                feature_dictionary(schema(c)).to_csv(out/'feature_electrode_region_dictionary.csv',index=False)
                cognition(c)
            elif stage=='stability':
                from .stability import run_stability
                (c['_root']/'private/interpretability').mkdir(parents=True,exist_ok=True);run_stability(c)
            elif stage=='report':
                from .report import create_report
                create_report(c)


if __name__=='__main__':main()

