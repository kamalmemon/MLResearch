import os

LOCAL_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

log_config = {
    'app': 'research_ml_email_mkt',
    'logdir': LOCAL_ROOT_DIR + '/logs',
    'backupcount': 2,
    'filerotate': 'W0'
}