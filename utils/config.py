MAX_SAMPLES = 50000
BATCH_SIZE = 16
BUFFER_SIZE = 50

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

THRESHOLD = 2
BUCKETS = [(8, 10), (12, 14), (16, 19)]
TESTSET_SIZE = 25000
# parameters for processing the dataset
DATA_PATH = '../cornell_movie_dialogs_corpus'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = './utils/processed'

# Hyper-parameters
NUM_LAYERS = 6  # they use 6
D_MODEL = 512  # they use 512
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1
ENC_VOCAB = 24455
DEC_VOCAB = 24666
