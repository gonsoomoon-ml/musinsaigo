#!/bin/bash
cd /opt/ml/code
eval "accelerate launch ${LAUNCH_ARGS} /opt/ml/code/${CODE_FILENAME} ${SCRIPT_ARGS}"