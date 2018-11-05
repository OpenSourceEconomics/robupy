import warnings
import socket

# This turns of the RuntimeWarnings during production so that they do not clutter the log files.
if 'heracles' not in socket.gethostname():
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')