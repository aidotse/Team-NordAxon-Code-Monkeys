#!/bin/bash
# File              : run.sh
# Author            : Sheetal Reddy <sheetal.reddy@ai.se>
# Date              : 23.10.2020
# Last Modified Date: 23.10.2020
# Last Modified By  : Sheetal Reddy <sheetal.reddy@ai.se>

ROOT_DIR=$PWD/../..
DATA_DIR=$ROOT_DIR/astra_data_readonly
CODE_DIR=$ROOT_DIR/Code

nvidia-docker  run   \
	-v $DATA_DIR:/astra_data_readonly \
	-v $CODE_DIR:/Code \
	--shm-size="8G" \
	-it nordaxon_code_monkeys \
	bash



