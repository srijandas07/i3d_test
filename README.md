Code for Testing I3D model.
By Srijan Das (srijan.das@inria.fr)

Requirements:
1. python 2.7
2. Keras 2.1.5
3. Tensorflow 1.4.1
4. cuda/8.0
5. cudnn/5.1-cuda-8.0

For help - use python test.py -h

Input - ../data/rgb - path of the video frames (input to the script)
        ../split/test_${dataset_name}.txt - Videos to be tested
        

Output -
         pre-trained models are stored in ../models folder
         prediction results are stored in results.txt within output folder 

For testing - 

./job.sh Path to the video_dir
example - .job.sh ../data/rgb/NTU/




