import logging

def config_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(sh_fmt)
    root_logger.addHandler(sh)

    infra_logger = logging.getLogger('infra')
    fh = logging.FileHandler('infra.log', mode='w')
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_fmt)
    infra_logger.addHandler(fh)

    ml_logger = logging.getLogger('ml')

