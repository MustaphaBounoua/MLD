# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute FAD between two multivariate Gaussian."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from src.eval_metrics.fad import fad_utils
from src.eval_metrics.fad.create_embeddings_main import compute_embeddings
import pathlib
import csv
import os
import glob

from src.utils import clean_folder

flags.DEFINE_string("background_stats", None,
                    "Tf record containing the background stats (mu sigma).")
flags.DEFINE_string("test_stats", None,
                    "Tf record containing the test stats (mu sigma).")

flags.mark_flags_as_required(["background_stats", "test_stats"])

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.
  mu_bg, sigma_bg = fad_utils.read_mean_and_covariances(FLAGS.background_stats)
  mu_test, sigma_test = fad_utils.read_mean_and_covariances(FLAGS.test_stats)
  fad = fad_utils.frechet_distance(mu_bg, sigma_bg, mu_test, sigma_test)
  
  print("okay")
  print("FAD: %f" % fad)


def get_fad(background_stats, output_stats):
    mu_bg, sigma_bg = fad_utils.read_mean_and_covariances(background_stats)
    mu_test, sigma_test = fad_utils.read_mean_and_covariances(output_stats)
    fad = fad_utils.frechet_distance(mu_bg, sigma_bg, mu_test, sigma_test)
    return fad


def get_fad_given_paths(path_1,path_2,clean_after=False ):
  
  file_list_1 = get_files_list(path_1)
  file_list_2 = get_files_list(path_2)
  
  
  stat_1 = os.path.join(path_1,"stat1")
  stat_2 = os.path.join(path_2,"stat2")
  
  compute_embeddings(file_list_1, output_path= stat_1)
  compute_embeddings(file_list_2, output_path= stat_2)
  
  fad_score = get_fad(stat_1,stat_2)
  
  # if clean_after:
  #   clean_folder(path_1)
  #   clean_folder(path_2)
  return fad_score



def get_files_list(path_files):
   
    files = glob.glob(path_files + '/*.wav')
    list_file = os.path.join(path_files,"list_of_paths.cvs")
    write_to_file(files, list_file )
    
    return list_file
    
def write_to_file(data_list, list_file):

    with open(list_file, "w") as outfile:
      outfile.write("\n".join(data_list))


if __name__ == "__main__":
  app.run(main)
