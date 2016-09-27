##Find the current location of the MAJICK directory
export MAJICK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
##Add it to the PATH to find scripts easily, and add to the PYTHONPATH to
##easily discover libraries
export PATH=$MAJICK_DIR:$PATH
export PYTHONPATH=$MAJICK_DIR:$PYTHONPATH
export PYTHONPATH=$MAJICK_DIR/imager_lib:$PYTHONPATH
