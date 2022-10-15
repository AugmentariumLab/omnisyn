import configargparse

from helpers import my_helpers


def parse_training_options():
  parser = configargparse.ArgumentParser(description='Train pipeline3')
  parser.add('-c', '--config', required=True, is_config_file=True,
             help='config file path')
  parser.add_argument('--script-mode',
                      type=str,
                      help='Mode of the script',
                      choices=['train_depth_pose',
                               'eval_depth_pose',
                               'train_inpainting',
                               'run_inpainting',
                               'run_inpainting_single',
                               'eval_inpainting',
                               'eval_inpainting_testset',
                               'eval_depth_test',
                               'dump_examples',
                               'train',
                               'run_example',
                               'visualize_point_cloud'],
                      required=True)
  parser.add_argument(
    '--gsv-path',
    type=str,
    help='path to GSV',
    default=
    "")
  parser.add_argument(
    '--carla-path',
    type=str,
    help='path to carla',
    default=
    "")
  parser.add_argument('--model-name',
                      type=str,
                      choices=['\'\''],
                      help='name of the model to use',
                      default='\'\'')
  parser.add_argument('--checkpoints-dir',
                      type=str,
                      help='checkpoints directory',
                      default="")
  parser.add_argument('--inpaint-checkpoints-dir',
                      type=str,
                      help='checkpoints directory',
                      default="")
  parser.add_argument("--batch-size",
                      type=int,
                      help="batch size for training",
                      default=2)
  parser.add_argument("--epochs",
                      type=int,
                      help="epochs for training",
                      default=999999)
  parser.add_argument("--learning-rate",
                      type=float,
                      help="learning rate",
                      default=0.00006)
  parser.add_argument("--width", type=int, help="width", default=256)
  parser.add_argument("--height", type=int, help="height", default=256)
  parser.add_argument("--opt-beta1", type=float, help="beta1", default=0.5)
  parser.add_argument("--opt-beta2", type=float, help="beta2", default=0.999)
  parser.add_argument("--train-tensorboard-interval",
                      type=int,
                      help="train tensorboard interval",
                      default=0)
  parser.add_argument("--validation-interval",
                      type=int,
                      help="validation interval",
                      default=200)
  parser.add_argument("--checkpoint-interval",
                      type=int,
                      help="checkpoint interval",
                      default=200)
  parser.add_argument("--smoothness-loss-lambda",
                      type=float,
                      help="smoothness loss lambda",
                      default=1)
  parser.add_argument("--checkpoint-count",
                      type=int,
                      help="how many checkpoints to keep",
                      default=3)
  parser.add_argument("--point-radius",
                      type=float,
                      help="point radius",
                      default=0.004)
  parser.add_argument("--device",
                      type=str,
                      help="preferred device",
                      default="cuda")
  parser.add_argument("--verbose",
                      type=bool,
                      help="print debugging statements",
                      default="true")
  parser.add_argument("--patch-loss-patch-size",
                      type=int,
                      help="size for patch loss",
                      default=5)
  parser.add_argument("--patch-loss-stride",
                      type=int,
                      help="stride for patch loss",
                      default=1)
  parser.add_argument("--patch-loss-stride-dist",
                      type=int,
                      help="dist for patch loss",
                      default=3)
  parser.add_argument("--inpaint-use-residual",
                      type=my_helpers.str2bool,
                      help="Use residual",
                      default=False)
  parser.add_argument("--inpaint-wrap-padding",
                      type=my_helpers.str2bool,
                      help="Wrap padding (CCNN)",
                      default=False)
  parser.add_argument("--loss", type=str, help="Loss function", default="l1")
  parser.add_argument("--inpaint-use-batchnorm",
                      type=my_helpers.str2bool,
                      help="Batch Norm",
                      default=True)
  parser.add_argument("--upscale-point-cloud",
                      type=my_helpers.str2bool,
                      help="Upscale point cloud",
                      default=True)
  parser.add_argument("--inpaint-one-conv",
                      type=my_helpers.str2bool,
                      help="Use 1x1 conv on last inpaint",
                      default=False)
  parser.add_argument("--carla-min-dist",
                      type=float,
                      help="Carla Min Distance",
                      default=8.0)
  parser.add_argument("--carla-max-dist",
                      type=float,
                      help="Carla Min Distance",
                      default=8.0)
  parser.add_argument("--mesh-width", type=int, help="Mesh Width",
                      default=512)
  parser.add_argument("--mesh-height", type=int, help="Mesh Height",
                      default=512)
  parser.add_argument("--inpaint-resolution", type=int, help="Mesh render resolution",
                      default=-1)
  parser.add_argument("--threshold-depth",
                      type=my_helpers.str2bool,
                      help="If depth > 65, set to 1000",
                      default=False)
  parser.add_argument("--interpolation-mode",
                      type=str,
                      help="Interpolation mode",
                      default="bilinear")
  parser.add_argument("--add-depth-noise",
                      type=float,
                      help="ABS of depth noise to add",
                      default=0)
  parser.add_argument("--use-pred-depth",
                      type=my_helpers.str2bool,
                      help="Train using predicted depth",
                      default=False)
  parser.add_argument("--train-pred-depth",
                      type=my_helpers.str2bool,
                      help="Train using predicted depth",
                      default=False)
  parser.add_argument("--use-blending",
                      type=my_helpers.str2bool,
                      help="Do blending instead of inpainting",
                      default=False)
  parser.add_argument("--inpaint-final-convblock",
                      type=my_helpers.str2bool,
                      help="Use final convblock for inpainting",
                      default=True)
  parser.add_argument("--inpaint-model-size",
                      type=int,
                      help="Size of the inpainting model",
                      default=4)
  parser.add_argument("--inpaint-model-layers",
                      type=int,
                      help="Layers in the inpainting model",
                      default=7)
  parser.add_argument("--depth-input-uv",
                      type=my_helpers.str2bool,
                      help="Input UVs into depth",
                      default=False)
  parser.add_argument("--normalize-depth",
                      type=my_helpers.str2bool,
                      help="Normalize depth values in loss",
                      default=False)
  parser.add_argument("--predict-zdepth",
                      type=my_helpers.str2bool,
                      help="Predict z-depth",
                      default=False)
  parser.add_argument("--cost-volume",
                      type=str,
                      help="Use cost volume instead of monocular",
                      default="")
  parser.add_argument("--clamp-depth-to",
                      type=float,
                      help="Clamp max depth to",
                      default=1000)
  parser.add_argument("--use-depth-mask",
                      type=my_helpers.str2bool,
                      help="Input depth mask instead of raw depths",
                      default=False)
  parser.add_argument("--inpaint-model-version",
                      type=str,
                      help="Version of the inpainting model",
                      default="v1")
  parser.add_argument("--rot-loss-lambda",
                      type=float,
                      help='Rotation loss factor',
                      default=1)
  parser.add_argument("--trans-loss-lambda",
                      type=float,
                      help='Translation loss factor',
                      default=1)
  parser.add_argument("--depth-range-loss-lambda",
                      type=float,
                      help='Depth range loss factor',
                      default=1)
  parser.add_argument("--clip-grad-value",
                      type=float,
                      help='Clip Gradients by value',
                      default=0)
  parser.add_argument("--model-use-v-input",
                      type=my_helpers.str2bool,
                      help='Input v coords into each conv block.',
                      default=False)
  parser.add_argument("--inpaint-depth",
                      type=my_helpers.str2bool,
                      help='Inpaint the depth also',
                      default=False)
  parser.add_argument("--inpaint-depth-factor",
                      type=float,
                      help='Inpaint the depth also',
                      default=0.01)
  parser.add_argument("--inpaint-depth-scale",
                      type=float,
                      help='Scale for depth input',
                      default=0.5)
  parser.add_argument("--debug-mode",
                      type=my_helpers.str2bool,
                      help='Debug mode',
                      default=False)
  parser.add_argument("--debug-one-batch",
                      type=my_helpers.str2bool,
                      help='Debug mode, use one batch',
                      default=False)
  parser.add_argument('--dataset',
                      type=str,
                      help='which dataset to use',
                      default="carla",
                      choices=["carla", "m3d", "gsv"])
  parser.add_argument('--min-depth',
                      type=float,
                      help='Minimum depth',
                      default=2)
  parser.add_argument('--max-depth',
                      type=float,
                      help='Maximum depth',
                      default=100)
  parser.add_argument('--turbo-cmap-min',
                      type=float,
                      help='Turbo colormap min depth',
                      default=2)
  parser.add_argument('--m3d-dist',
                      type=float,
                      help='Matterport3d Distance',
                      default=1)
  parser.add_argument('--mesh-removal-threshold',
                      type=float,
                      help='mesh triangle removal threshold',
                      default=2.0)
  parser.add_argument('--representation',
                      type=str,
                      help='3d representation, either pointcloud or mesh',
                      default='mesh',
                      choices=['mesh', 'pointcloud'])
  parser.add_argument('--depth-type',
                      type=str,
                      help='depth type',
                      default='one_over')
  parser.add_argument('--inpaint-loss',
                      type=str,
                      help='loss for inpainting',
                      default='l1')
  parser.add_argument('--depth-loss',
                      type=str,
                      help='loss for inpainting',
                      default='l1')
  parser.add_argument('--inpaint-upscale',
                      type=my_helpers.str2bool,
                      help='make inpainting do upscaling',
                      default=False)
  parser.add_argument('--predict-meshcuts',
                      type=my_helpers.str2bool,
                      help='predict discontinuities for mesh cutss',
                      default=False)
  parser.add_argument('--train-depth-steps',
                      type=int,
                      help='',
                      default=90600)
  parser.add_argument('--train-inpainting-steps',
                      type=int,
                      help='',
                      default=45200)
  return parser.parse_args()
