FINAL_YAMLS = {
    'cheetah'   : 'results/MOCheetah/final/config.yaml',
    'hopper'    : 'results/MOHopper/clean-front/back-to-small-network/config.yaml',
    'walker'    : 'results/MOWalker/lower-lr-shorter/config.yaml',
    'humanoid'  : 'config/mohumanoid.yaml',
    'ant'       : 'config/moant.yaml',
    'bruce5D'   : 'bruce-results/5D/jt-debugging/test/config.yaml',
    # 'bruce5D'   : 'bruce-results/5D/jt-debugging/tighter-sigmas/config.yaml',
    'bruce6D'   : 'bruce-results/6D/normal-sigmas/tuned-energy/config.yaml',
    'bruce6D+DR': 'bruce-results/6D/dr-first-try/longer-more-aggressive/config.yaml'
}

HYPER_PARETOS = {
    'cheetah'   : 'ral/pareto_logs/HyperCheetah.csv',
    'hopper'    : 'ral/pareto_logs/HyperHopper.csv',
    'walker'    : 'ral/pareto_logs/HyperWalker.csv',
    'humanoid'  : 'ral/pareto_logs/HyperHumanoid.csv',
    'ant'       : 'ral/pareto_logs/HyperAnt.csv',
}

HYPER_TIMES = {
    'cheetah'   : 'ral/time_logs/HyperCheetah',
    'hopper'    : 'ral/time_logs/HyperHopper',
    'walker'    : 'ral/time_logs/HyperWalker',
    'humanoid'  : 'ral/time_logs/HyperHumanoid',
    'ant'       : 'ral/time_logs/HyperAnt',
}

BRUCE_TRADEOFFS = {
    'swing_arms': [0.57260025, 0.00301419, 0.32290676, 0.00075819, 0.06009477, 0.04062585],
    'smooth':     [5.8893696e-04, 7.2646540e-03, 2.4472056e-02, 6.3891990e-04, 9.6675140e-01, 2.8411314e-04],
    'rigid_arms': [0.5130773 , 0.01274788, 0.00119235, 0.28102297, 0.1855738 , 0.00638578],
}