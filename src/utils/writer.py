import tensorflow as tf
import numpy as np
import os 
import time
from multiprocessing import Process, Value, Array, Manager
import multiprocessing
from glob import glob
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(seq, images):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      "sequence": _bytes_feature(str(seq["seq_name"]).encode('utf-8')),
      'height': _int64_feature(int(seq["height"])),
      'width': _int64_feature(int(seq["width"])),
      'frames': _int64_feature(seq["2d_poses"].shape[1]),
      'image_raw': _bytes_feature(images.tobytes()), # np.uint8
      'pmask': _bytes_feature(seq["pmask"][:, :, 0].astype(np.uint8).tobytes()), # np.uint8
      '2d_norm_poses': _bytes_feature(seq["2d_norm_poses"].astype(np.float32).tobytes()), # np.float32
      '2d_root_joints': _bytes_feature(seq["2d_root_joints"].astype(np.int32).tobytes()), # np.int32 otherwise precision errors
      '2d_poses': _bytes_feature(seq["2d_poses"].astype(np.int32).tobytes()), # np.int32 otherwise precision errors
      '3d_poses': _bytes_feature(seq["3d_poses"].astype(np.float32).tobytes()), # np.float32
      'trans': _bytes_feature(seq["trans"].astype(np.float32).tobytes()), # np.float32
      "intrinsics": _bytes_feature(seq["intrinsics"].astype(np.float32).tobytes()), # np.float32
      "extrinsics": _bytes_feature(seq["extrinsics"].astype(np.float32).tobytes()), # np.float32
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

class TFWriter:
  
    def __init__(self) -> None:
       pass

    def serialize_seq2tfrecord(self, seq_path):

        cwd = os.getcwd()
        seq_path_splitted = os.path.split(seq_path)
        datasetDir = os.path.join(cwd, 'data', '3DPWPreprocessed')
        img_dir = os.path.join(datasetDir, "imageFiles")
        sequence_file = seq_path_splitted[-1].split(".")[0]
        
        seq = np.load(seq_path, allow_pickle=True)
        
        folder_name = os.path.split(seq_path_splitted[0])[-1]

        
        img_sequence_dir = os.path.join(img_dir, sequence_file)    
        img_names = os.listdir(img_sequence_dir)
        
        print(f"++++ Serializing Sequence: {seq['seq_name']} to file: {folder_name}")

        # with tf.io.TFRecordWriter(os.path.join(tfrecord_path, folder_name) + ".tfrecord") as writer:
        imgs = []
        for i, name in enumerate(img_names):
            image_string = open(os.path.join(img_sequence_dir, name), 'rb').read()
            # x = tf.image.decode_jpeg(image_string)
            imgs.append(image_string)
        tf_example = serialize_example(seq, np.array(imgs))
        

        print(f"---- Serialization Completed: {seq['seq_name']}")
        
        return tf_example
    
    def cvt_seq2tfrecord(self):
      
        cwd = os.getcwd()
        datasetDir = os.path.join(cwd, 'data', '3DPWPreprocessed')
        
        seq_dir = os.path.join(datasetDir, "sequenceFiles") 
        seq_folders = glob(os.path.join(seq_dir, "*"))
        out_path = os.path.join(cwd, 'data', '3DPWPreprocessed', 'tfrecords_combined')
        folder_names = []
        seq_paths = []
        writers = []
        for folder in seq_folders:
            _folder_name = os.path.split(folder)[-1]
            _new_writer = tf.io.TFRecordWriter(os.path.join(out_path, _folder_name) + ".tfrecord")
            writers.append(_new_writer)
            folder_names.append(_folder_name)
            seq_paths.append(glob(os.path.join(folder, "*.npz")))
        # seq_paths = glob(os.path.join(seq_dir, "*", "*.npz"))
        print(f"====================== Started ======================")
        tic = time.time()
        # with multiprocessing.Pool() as pool:
        #     # call the function for each item in parallel
        #     result = pool.map(self.write_seq2tfrecord, seq_paths)
        for path, writer, folder_name in zip(seq_paths, writers, folder_names):
            with multiprocessing.Pool() as pool:
                result = pool.map(self.serialize_seq2tfrecord, path)
                
                print(f"++ Writing to {folder_name}")
                for res in result:
                    writer.write(res)
                writer.close()
                print(f"-- Writing Complete")
            
        toc = time.time()

        print('Done in {:.4f} seconds'.format(toc-tic))
        
        print(f"====================== Finished ======================")

    
    
    

if __name__ == '__main__':
    writer = TFWriter()
    writer.cvt_seq2tfrecord()
    pass