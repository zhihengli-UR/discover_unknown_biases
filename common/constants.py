# Strings
LOSS = 'loss'
ACCURACY = 'acc'
ITERATION = 'iteration'
INPUT_IMAGE = 'input_image'
RECON_IMAGE = 'recon_image'
RECON = 'recon'
FIXED = 'fixed'
SQUARE = 'square'
ELLIPSE = 'ellipse'
HEART = 'heart'
TRAVERSE = 'traverse'
RANDOM = 'random'
TEMP = 'tmp'
GIF = 'gif'
JPG = 'jpg'
FACTORVAE = 'FactorVAE'
DIPVAEI = 'DIPVAEI'
DIPVAEII = 'DIPVAEII'
BetaTCVAE = 'BetaTCVAE'
INFOVAE = 'InfoVAE'
TOTAL_VAE = 'total_vae'
TOTAL_VAE_EPOCH = 'total_vae_epoch'
LEARNING_RATE = 'learning_rate'

# Algorithms
ALGS = ('AE', 'VAE', 'BetaVAE', 'CVAE', 'IFCVAE')
LOSS_TERMS = (FACTORVAE, DIPVAEI, DIPVAEII, BetaTCVAE, INFOVAE)

# Datasets
DATASETS = ('celebA', 'dsprites_full', 'dsprites_noshape', 'color_dsprites', 'noisy_dsprites', 'scream_dsprites',
            'smallnorb', 'cars3d', 'shapes3d',
            'mpi3d_toy', 'mpi3d_realistic', 'mpi3d_real')
DEFAULT_DATASET = DATASETS[1]
TEST_DATASETS = DATASETS[0:2]  # celebA, dsprites_full

# Architectures
DISCRIMINATORS = ('SimpleDiscriminator', 'SimpleDiscriminatorConv64')
TILERS = ('MultiTo2DChannel',)
DECODERS = ('SimpleConv64', 'ShallowLinear', 'DeepLinear')
ENCODERS = ('SimpleConv64', 'SimpleGaussianConv64', 'PadlessConv64', 'PadlessGaussianConv64',
            'ShallowGaussianLinear', 'DeepGaussianLinear')

# Evaluation Metrics
EVALUATION_METRICS = ('dci', 'factor_vae_metric', 'sap_score', 'mig', 'irs', 'beta_vae_sklearn')

# Schedulers
LR_SCHEDULERS = ('ReduceLROnPlateau', 'StepLR', 'MultiStepLR', 'ExponentialLR',
                 'CosineAnnealingLR', 'CyclicLR', 'LambdaLR')
SCHEDULERS = ('LinearScheduler', )

# GENFORCE_OBJECT_CLASSES = {'car', 'tvmonitor', 'boat', 'bottle', 'diningtable', 'train', 'cow', 'chair', 'pottedplant', 'bus', 'sofa', 'person', 'bicycle', 'bird', 'airplane', 'dog', 'sheep', 'motorbike', 'cat', 'horse'}
# GENFORCE_SCENE_CLASSES = {'bedroom', 'bridge', 'church', 'classroom', 'conferenceroom', 'diningroom', 'kitchen', 'livingroom', 'restaurant', 'tower'}
GENFORCE_OBJECT_CLASSES = {'car': [751], 'boat': [554, 625, 814], 'bottle': [440, 737, 989, 907], 'train': [466], 'chair': [423, 559, 765], 'bicycle': [671], 'dog': [153, 200, 229, 230, 235, 238, 239, 245, 248, 251, 252, 254, 256, 275], 'motorbike': [670], 'cat': [281, 282, 283, 284, 285]}
GENFORCE_SCENE_CLASSES = {'bedroom': [52], 'bridge': [66], 'church': [91], 'classroom': [92], 'conferenceroom': [102], 'diningroom': [121], 'kitchen': [203], 'livingroom': [215], 'restaurant': [284], 'tower': [334]}


SYNTHETIC_ATTRIBUTE_NAME = {
    'dsprites_full': ['shape', 'scale', 'orientation', 'position_x', 'position_y'],
    'smallnorb': ['categories', 'elevations', 'azimuth', 'lighting_conditions'],
    'mpi3d_real': ['color', 'shape', 'size', 'camera_height', 'background_color', 'first_DOF_horizontal', 'second_DOF_vertical']
}
