#!/bin/bash
MODEL_NAME=$1

PACKAGE_FOLDER=packages/$MODEL_NAME
ARTIFACTS_SRC=../shared/assets/artifacts/$MODEL_NAME
MODEL_SRC=../shared/assets/models/$MODEL_NAME.onnx

echo "[PACKAGING]"
rm -r $PACKAGE_FOLDER 
mkdir -p $PACKAGE_FOLDER  # Ensure the package folder exists
cp -r $ARTIFACTS_SRC $PACKAGE_FOLDER
mv $PACKAGE_FOLDER/$MODEL_NAME $PACKAGE_FOLDER/artifacts  
rm -r $PACKAGE_FOLDER/artifacts/tempDir
cp $MODEL_SRC $PACKAGE_FOLDER
mv $PACKAGE_FOLDER/$MODEL_NAME.onnx $PACKAGE_FOLDER/model.onnx
cp scripts/*.py $PACKAGE_FOLDER
echo "Done."

echo "[DEPLOYING]"
scp -r $PACKAGE_FOLDER beagle:/opt/model_zoo
echo "Done."
