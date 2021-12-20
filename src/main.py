# -*- coding: utf-8 -*-
"""
    Main Module for the project
    This is the driver of the whole project,
    execution of all the steps is performed here
"""

from constants import *
from train_test_model import *

def main():
  """
    This is the main method of the code, the execution brings
    from here. It basically intializes the network and 
    train it
    Args:
      (None)
    Returns:
      (None)
  """
  #choose the netwrok
  if base_model:  net_type = 'baseline'
  else:  net_type = 'proposed'

  #initialize training
  run_training(net_type)
  


if __name__ == "__main__":
  #run
  main()
