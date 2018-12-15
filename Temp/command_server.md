* Script calculate and save face encoding
	* ``cd FaceRecognition``
	* ``python3 -m model.baseline1 --dataset_dir /home/datasets/msceleb/extracted_images --save_dir ./Dataset/Process``
* Create subset clean data from original dataset
	* ``cd FaceRecognition``
	* ``python3 -m model.baseline1 --face_encoding_dir /home/quan/FaceRecognition/Dataset/Process/face_encodings --src_dataset_dir /home/datasets/msceleb/extracted_images --dst_dataset_dir /home/quan/FaceRecognition/Dataset/Version2``
	
* Split dataset to train - test dataset
	* ``cd FaceRecognition``
	* ``python3 -m preprocess.prepare_dataset --src_dataset_dir /home/quan/FaceRecognition/Dataset/Version2 --dst_dataset_dir /home/quan/FaceRecognition/Dataset/Split_Version2 --test_size 0.1``
	
* Train baseline1
	* ``cd FaceRecognition``
	* ``python3 -m train --training_data_dir ./Dataset/Split_Version2/Train --test_data_dir ./Dataset/Split_Version2/Test --face_encoding_dir ./Dataset/Process/face_encodings --mid_name_path ./Dataset/Process/MID_Name.json --model_names "KNN-RandomForest"``
	
* Evaluate baseline1
	* ``cd FaceRecognition``
	* ``python3 -m evaluate --training_data_dir ./Dataset/Split_Version2/Train --test_data_dir ./Dataset/Split_Version2/Test --face_encoding_dir ./Dataset/Process/face_encodings --mid_name_path ./Dataset/Process/MID_Name.json --model_dir ./Experiment/2018-11-19_22-56-34/Model --model_names "KNN"``
	
* Reszie images
	* ``python3 -m utils_dir.project_utils --src_image_dir ./Dataset/Resized_Split_Version2 --size 160``
	
* Train pretrained
	* ``python3 src/classifier.py TRAIN ../FaceRecognition/Dataset/Resized_Split_Version2/Train ./model/pretrained_1/20170512-110547.pb ./model/my_model/my_cls.pkl --batch_size 128``

* Train pretrained resnet
	* ``python3 -m model.pretrained --dataset_dir ./Dataset/Split_Version3/ --num_epochs 50``