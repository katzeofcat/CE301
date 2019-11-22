First download everything
To use it first download the object detection API from tensorflow's github page and follow the installation instraction:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#add-libraries-to-pythonpath

Then use 
https://github.com/tzutalin/labelImg to make a custom dataset with label and bounding boxes which is in a XML format

Then use the program convert xml to csv to covert (before using it the content must be change to your label and directory) 
and next with the output data use Tfrecord.py to generate a tfrecord which is used to train the network.

Now edit the pb.txt file to your classes and for first class is 1 and proceading is 2 and etc..

Next edit the configure file with the number of classes and the file directory of .pbtxt file and the pretrain model's checkpoint
(The pre train model and its checkpoint can be download from tensorflow page in model zoo link that given above or search from internet.)

After everything done, put all data in a file then put under tensorflow/models/research/object_detection directory and use the following code under tensorflow/models/research/object_detection using cmd to start training:
python3 train.py --logtostderr --train_dir=your file directory(the pre train model should be inside) --pipeline_config_path= your configure file directiory 

After training is complete (when the loss curve is stable stop training) there will be servel checkpoint file, choose the latest one and run the following code under the tensorflow/models/research/object_detection using cmd
python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path your configure file \
    --trained_checkpoint_prefix your checkpoint\
    --output_directory your desired directory
    
Then finally run the video.py file with change the .pb directory and the lable map directory to the data that generated in above file to run. so you can start use object reconization 


OR

YOU CAN USE the data from pbfile if you don't want to train a custom dataset

If want to connect to robot first you must download a aria robot enviroment for python from aria robot homepage
Then install a maper tool to make a map with robot that fit to run by mobilesim
Open mobilesim and run the map. Then run the program video1.py to start control the robot. in this progrm only bowl and cup can use to control robot turn left or right. 
The degree of turning can be change via the veloctiy inside the left.py or right.py