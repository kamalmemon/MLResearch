import logging
import logging.handlers as handlers
import time
import os
from config import log_config
import sys
import uuid

config = log_config
app = config['app']
logdir = config['logdir']
logger = logging.getLogger(app)
logger.setLevel(logging.INFO)
identifier = { 'uuid': uuid.uuid4() }

# Formatter
formatter = logging.Formatter('%(asctime)s - %(uuid)s - %(name)s - %(module)s - %(levelname)s - %(message)s')

if not os.path.exists(logdir):
        os.mkdir(logdir)

log_path = logdir + '/' + app + '.log'

logHandler = handlers.TimedRotatingFileHandler(log_path, when=config['filerotate'], interval=1, backupCount=config['backupcount'])
logHandler.setLevel(logging.INFO)
logHandler.setFormatter(formatter)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)

logger.addHandler(logHandler)
logger.addHandler(consoleHandler)
logger = logging.LoggerAdapter(logger, identifier)
