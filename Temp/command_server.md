* Script calculate and save face encoding
	* ``cd FaceRecognition``
	* ``python3 -m model.baseline1 --dataset_dir /home/datasets/msceleb/extracted_images --save_dir ./Dataset/Process``
* Create subset clean data from original dataset
	* ``cd FaceRecognition``
	* ``python3 -m model.baseline1 --face_encoding_dir /home/quan/Dataset/Process/face_encodings --src_dataset_dir /home/datasets/msceleb/extracted_images --dst_dataset_dir /home/quan/Dataset/Version2``